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
    #   physical stride: (1024, 1)  # 1 is fastest moving dimension..cols
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
    print(f"After zipped divide by tile (2,256): {blocked}")
    # -> ((2,256),(1,4)):((1024,1),(0,256))
    #
    # Read this as:
    #   first tuple  ((2,256), (1,4))
    #     - (2,256) is the per-block tile shape  WITHIN TILE
    #     - (1,4)   tells us there are 1 x 4 such tiles in the full tensor ... WHICH TILE
    #
    #   second tuple ((1024,1), (0,256))
    #     - (1024,1) is the stride inside one block tile
    #     - (0,256)  is the stride for moving between block tiles
    #         * moving to the next tile row changes nothing (0), because there
    #           is only one tile in the row dimension
    #         * moving to the next tile column jumps by 256 elements
    #    The 1024 means:
    #
    #    if you stay in the same tile and move from row 0 to row 1, the linear offset increases by 1024
    #    if you stay in the same row and move from column j to j+1, the linear offset increases by 1
    
    #  One tile is thus:
    #  (2,256):(1024,1)  

    # `blocked` has hierarchical coordinates:
    #     WITHIN TILE                                        WHICH TILE
    #   ((coord_inside_tile_row, coord_inside_tile_col), (which_tile_row, which_tile_col))
    #
    # So in `((None, None), (0, tile_col_idx))`:
    #   - `(None, None)` means "keep all coordinates inside the chosen tile"
    #   - `(0, tile_col_idx)` means "pick tile row 0 and tile column `tile_col_idx`"
    #
    # Since `tile_col_idx = 2`, this picks the 3rd tile across the columns,
    # which corresponds to columns [512:768).
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
    #   - the first stride tuple (0,1) is the stride -- inside one thread tile --:
    #       * moving along the thread-tile row does nothing because its extent is 1
    #       * moving along the thread-tile columns advances by 1 element
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

    # `gA`, `gB`, and `gC` are already tiled by `zipped_divide(..., tiler_mn)` in
    # the host JIT function. So their coordinate structure is:
    #
    #   ((coord_inside_tile_m, coord_inside_tile_n), which_tile)
    #
    # For this demo there are 4 tiles total across the N dimension, so `which_tile`
    # is effectively the block index.
    #
    # Slice out one thread-block tile from each tiled tensor.
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    # After this slice, each `blk*` is just the local tile owned by this block.
    # In this example that tile has shape (2, 256), so conceptually:
    #
    #   blkA : (tile_row, tile_col) -> global-memory address
    #
    # The block no longer needs to mention "which tile" because `bidx` already
    # selected it.

    # `tv_layout` is the thread/value layout built on the host:
    #
    #   (thread_idx, value_idx) -> (tile_row, tile_col)
    #
    # `composition(blkA, tv_layout)` plugs that mapping into the block tile:
    #
    #   blkA     : (tile_row, tile_col) -> address
    #   tv_layout: (thread_idx, value_idx) -> (tile_row, tile_col)
    #   -----------------------------------------------------------
    #   tidfrgA  : (thread_idx, value_idx) -> address
    #
    # So composition rewrites the block-local tensor in thread-centric terms.
    # Instead of asking "which element of the tile is this?", we can ask
    # "which value owned by which thread is this?"
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    # Now select one thread's fragment.
    #
    # `tidx` chooses the thread, and `None` means "keep all values owned by
    # that thread". So each `thr*` is a small per-thread view:
    #
    #   thrA : (value_idx) -> address
    #
    # In other words, `thrA` is the fragment of A that this one thread will
    # load, `thrB` is the fragment of B, and `thrC` is where this thread will
    # store its results.
    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    # Load the thread-local fragments, add them elementwise, then store the
    # results back through the thread-local output view.
    thrC[None] = thrA.load() + thrB.load()


@cute.jit
def tiled_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    coalesced_ldst_bytes = 16
    # 16 bytes = 128 bits, so this asks CuTe to build the value layout around
    # one 128-bit memory transaction per contiguous load/store lane.
    #
    # For fp16 elements (16 bits each), 16 bytes corresponds to 8 contiguous
    # elements. That is why the byte-oriented value layout later gets recast
    # into an element-oriented layout with 8 fp16 values along that dimension.

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
    # `a` repeats 0..15, while `b` is a constant 100 everywhere, so the
    # result reads naturally as 100, 101, 102, ...
    a = (torch.arange(m * n, device="cuda", dtype=torch.int32) % 16).to(torch.float16)
    b = torch.full((m * n,), 100, device="cuda", dtype=torch.float16)
    a = a.reshape(m, n)
    b = b.reshape(m, n)
    c = torch.zeros_like(a)

    # `assumed_align=16` means CuTe can treat these gmem pointers as 16-byte
    # aligned, which matches the 128-bit load/store shape used above.
    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)
    print(f"---- Starting shapes ----")
    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mB shape: {mB.shape}, stride: {mB.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}\n")

    expected = a + b

    tiled_add = cute.compile(tiled_elementwise_add, mA, mB, mC)
    # call our compiled kernel
    tiled_add(mA, mB, mC)
    # ensure results are ready with a cpu synch
    torch.cuda.synchronize()

    print(f"\n---- Results: ----")
    print("input A, first row:", a[0, :8].tolist())
    print("input B, first row:", b[0, :8].tolist())
    print(f"")
    print("expected, first row:", expected[0, :8].tolist())
    print("output C, first row:", c[0, :8].tolist())
    print(f"\nmatches torch:", torch.equal(c, expected))


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
