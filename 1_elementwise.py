import os
import subprocess
import sys

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.jit
def show_layout_theory_jit():
    m = 2  # rows
    n = 1024  # cols
    tile_m = 2  # rows
    tile_n = 256  # cols 
    # tile_m, tile_n = (2, 256)
    
    tile_col_idx = 2

    # Start from a 2D row-major layout that matches the real tensor demo:
    #
    #   logical shape  : (2, 1024)
    #   physical stride: (1024, 1)
    #
    # So moving by +1 row jumps 1024 elements, while moving by +1 column
    # jumps 1 element.
    gmem = cute.make_layout((m, n), stride=(n, 1))
    print(f"Global layout: {gmem}")
    # -> (2,1024):(1024,1)
    #
    # Read this as:
    #   shape  = (rows, cols)   = (2, 1024)
    #   stride = (row_stride, col_stride) = (1024, 1)

    # Tile the global layout into blocks of shape (2, 256).
    #
    # Because the full tensor is (2, 1024), this produces:
    #   - one full tile in the row dimension
    #   - four tiles in the column dimension
    blocked = cute.zipped_divide(gmem, (tile_m, tile_n))
    print(f"After block divide: {blocked}")
    # -> ((2,256),(1,4)):((1024,1),(0,256))
    #
    # Read this as:
    #   first tuple  ((2,256), (1,4))
    #     - (2,256) is the per-block tile shape
    #     - (1,4)   tells us there are 1 x 4 such tiles in the full tensor
    #
    #   second tuple ((1024,1), (0,256))
    #     - (1024,1) is the stride inside one block tile
    #     - (0,256)  is the stride for moving between block tiles
    #         * moving to the next tile row changes nothing (0), because there
    #           is only one tile in the row dimension
    #         * moving to the next tile column jumps by 256 elements

    # Pick the third tile in the tile-column dimension: tile (0, 2).
    # This corresponds to columns [512:768).
    blk_layout, blk_offset = cute.slice_and_offset(((None, None), (0, tile_col_idx)), blocked)
    print(f"Block (0, {tile_col_idx}) layout: {blk_layout}")
    print(f"Block (0, {tile_col_idx}) base offset: {blk_offset}")
    # -> ((2,256)):((1024,1))
    # -> 512
    #
    # Read this as:
    #   - the chosen block still has local shape (2, 256)
    #   - inside that block, row-major addressing is unchanged
    #   - the block starts at linear offset 512 in the original tensor

    # Now divide the chosen block into "one thread handles one full row slice
    # of 256 contiguous elements". This is not the exact TV layout used in
    # part 2, but it gives an easy-to-read stepping stone:
    #
    #   block    : (2, 256)
    #   thread tile: (1, 256)
    #
    # so we get 2 thread-tiles total, one per row.
    threaded = cute.zipped_divide(blk_layout, (1, 256))
    print(f"After thread divide: {threaded}")
    # -> ((1,256),(2,1)):((0,1),(1024,0))
    #
    # Read this as:
    #   - (1,256) is the work owned by one logical thread tile
    #   - (2,1) means there are 2 such thread tiles inside the block
    #   - the second tile stride (1024,0) shows how to move between
    #     thread tiles:
    #       * moving to the next thread-tile row jumps by 1024 elements
    #       * moving in the tile-column dimension changes nothing because
    #         there is only one tile there

    # Show the first few column offsets in row 0 of the chosen block.
    print("  first few addresses inside block row 0:")
    print(f"    col 0 -> global offset {blk_offset + blk_layout((0, 0))}")
    print(f"    col 1 -> global offset {blk_offset + blk_layout((0, 1))}")
    print(f"    col 2 -> global offset {blk_offset + blk_layout((0, 2))}")
    print(f"    col 3 -> global offset {blk_offset + blk_layout((0, 3))}")


def show_layout_theory():
    print("== Part 1: Layout theory ==")
    show_layout_theory_jit()


@cute.kernel
def tiled_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Slice out one thread-block tile from each tiled tensor.
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    # Compose block-local tensors with the thread/value layout so that
    # (thread_idx, value_idx) maps directly to a memory address.
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    # Slice out the fragment owned by one thread.
    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    thrC[None] = thrA.load() + thrB.load()


@cute.jit
def tiled_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    coalesced_ldst_bytes = 16

    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    # A smaller TV layout that covers a (2, 256) tile:
    # threads  : (1, 32)
    # values   : (2, 16B) -> recast to (2, 8) for fp16
    # combined : (2, 256)
    thr_layout = cute.make_ordered_layout((1, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((2, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    print(f"Tiler: {tiler_mn}")
    print(f"TV layout: {tv_layout}")
    print(f"Tiled gA: {gA.type}")

    tiled_elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run_simple_kernel_demo():
    print("\n== Part 2: Real gmem tensors with CuTe tiling ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    # Use a small real tensor that lines up with four (2, 256) CuTe tiles.
    # The physical torch stride stays non-overlapping row-major: (1024, 1).
    m = 2
    n = 1024

    # Keep values small so the float16 correctness check is easy to read.
    a = (torch.arange(m * n, device="cuda", dtype=torch.int32) % 16).to(torch.float16)
    b = (100 + (torch.arange(m * n, device="cuda", dtype=torch.int32) % 16)).to(torch.float16)
    a = a.reshape(m, n)
    b = b.reshape(m, n)
    c = torch.zeros_like(a)

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mB shape: {mB.shape}, stride: {mB.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}")

    expected = a + b

    tiled_add = cute.compile(tiled_elementwise_add, mA, mB, mC)
    tiled_add(mA, mB, mC)
    torch.cuda.synchronize()

    print("input A, first row:", a[0, :8].tolist())
    print("input B, first row:", b[0, :8].tolist())
    print("expected, first row:", expected[0, :8].tolist())
    print("output C, first row:", c[0, :8].tolist())
    print("matches torch:", torch.equal(c, expected))


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_layout_theory()
            return
        if mode == "--kernel":
            run_simple_kernel_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    # Run the two demos in separate Python processes. In this environment,
    # combining the MLIR layout path and runtime tensor path in one process
    # can corrupt memory during execution/shutdown.
    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    # Work around a shutdown bug in this CUTLASS DSL build after MLIR context use.
    sys.stdout.flush()
    os._exit(0)
