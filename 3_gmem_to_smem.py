import os
import subprocess
import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


M = 64
N = 64
BLK_M = 32
BLK_N = 32
N_THR = 128
COPIES_PER_THR = (BLK_M * BLK_N) // N_THR  # 1024 / 128 = 8


@cute.jit
def show_gmem_to_smem_theory_jit():
    block_row_idx = 1
    block_col_idx = 1

    # Start from a 2D row-major global-memory layout.
    #
    # Shape  (64, 64) means 64 rows and 64 columns.
    # Stride (64, 1) means:
    #   - moving down one row jumps by 64 elements
    #   - moving across one column jumps by 1 element
    
    gmem = cute.make_layout((M, N), stride=(N, 1))
    print(f"Global layout: {gmem}")
    # -> (64,64):(64,1)

    # Tile the matrix into 32x32 thread-block tiles.
    blocked = cute.zipped_divide(gmem, (BLK_M, BLK_N))
    print(f"After block tiling by (32,32): {blocked}")
    # -> ((32,32),(2,2)):((64,1),(2048,32))
    #
    # Read this as:
    #   - each block tile has shape (32, 32)
    #   - there are (2, 2) such tiles in the full matrix
    #   - inside one tile, row-major stride is still (64, 1)
    #   - moving to the next tile row jumps by 32*64 = 2048 elements
    #   - moving to the next tile col jumps by 32 elements

    # Pick block tile (1,1), i.e. the bottom-right tile.
    blk_layout, blk_offset = cute.slice_and_offset(
        ((None, None), (block_row_idx, block_col_idx)),
        blocked,
    )
    print(f"Block ({block_row_idx}, {block_col_idx}) layout: {blk_layout}")
    print(f"Block ({block_row_idx}, {block_col_idx}) base offset: {blk_offset}")
    # -> (32,32):(64,1)
    # -> 2080
    #
    # 2080 = 1*2048 + 1*32. In matrix terms, this block starts at row 32, col 32.

    # Shared memory usually stores the same logical tile shape as the GMEM tile.
    # Here that staged tile is just another 32x32 row-major layout.
    sA_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))
    print(f"SMEM tile layout: {sA_layout}")
    # -> (32,32):(32,1)
    #
    # Read this as:
    #   - the staged shared-memory tile also has shape (32, 32)
    #   - inside SMEM, moving to the next row jumps by 32 elements
    #   - moving to the next column jumps by 1 element

    # The tile has 32*32 = 1024 elements.
    # With 128 threads, each thread copies 1024 / 128 = 8 elements.
    #
    # One simple assignment is:
    #   copy_slot = thread_idx + 128 * iter_idx
    #
    # That means thread 5 gets copy slots:
    #   5, 133, 261, 389, 517, 645, 773, 901
    #
    # But conceptually what matters is the 2D tile coordinates those slots map to.
    print("Thread 5 copies these tile coordinates:")
    print(f"  iter 0 -> coord {sA_layout.get_hier_coord(5)}")
    print(f"  iter 1 -> coord {sA_layout.get_hier_coord(133)}")
    print(f"  iter 2 -> coord {sA_layout.get_hier_coord(261)}")
    print(f"  iter 3 -> coord {sA_layout.get_hier_coord(389)}")
    print(f"  iter 4 -> coord {sA_layout.get_hier_coord(517)}")
    print(f"  iter 5 -> coord {sA_layout.get_hier_coord(645)}")
    print(f"  iter 6 -> coord {sA_layout.get_hier_coord(773)}")
    print(f"  iter 7 -> coord {sA_layout.get_hier_coord(901)}")

    # So thread 5 walks down the tile at a fixed column:
    #   (0,5), (4,5), (8,5), ...
    #
    # Meanwhile, in the same iteration, neighboring threads copy neighboring columns:
    print("Iteration 0 across threads 0..3:")
    print(f"  thr 0 -> coord {sA_layout.get_hier_coord(0)}")
    print(f"  thr 1 -> coord {sA_layout.get_hier_coord(1)}")
    print(f"  thr 2 -> coord {sA_layout.get_hier_coord(2)}")
    print(f"  thr 3 -> coord {sA_layout.get_hier_coord(3)}")
    #
    # That is the usual gmem->smem copy idea:
    #   1. choose a block tile in GMEM
    #   2. allocate a same-shaped tile in SMEM
    #   3. give each thread a handful of tile coordinates to copy
    #   4. copy GMEM tile element -> SMEM tile element
    #   5. synchronize before using the SMEM tile


def show_gmem_to_smem_theory():
    print("== Part 1: GMEM To SMEM Theory ==")
    show_gmem_to_smem_theory_jit()


@cute.kernel
def gmem_to_smem_copy_kernel(
    gA: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # Select the 32x32 global-memory tile owned by this thread block.
    blk_coord = ((None, None), (bidy, bidx))
    blkA = gA[blk_coord]
    blkC = gC[blk_coord]

    # Allocate a 32x32 tile in shared memory. We use a simple row-major shared
    # memory layout so the staging step is easy to read.
    #
    # `sA_layout` answers the question:
    #   "if I know a coordinate (row, col) inside the tile, where should that
    #    element live inside SMEM?"
    #
    # We keep the same logical tile shape `(32, 32)`, but now the layout is for
    # the staged shared-memory copy rather than the original GMEM view.
    smem = cutlass.utils.SmemAllocator()
    sA_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))
    sA = smem.allocate_tensor(
        element_type=gA.element_type,
        layout=sA_layout,
        byte_alignment=16,
    )

    # Build the same "128 threads x 8 iterations" copy assignment from part 1.
    linear_copy = cute.zipped_divide(cute.make_layout(BLK_M * BLK_N, stride=1), (N_THR,))

    # Each thread copies 8 elements from the GMEM tile into the SMEM tile.
    #
    # `linear_copy((tidx, i))` gives a linear slot in the flattened 1024-element
    # tile, e.g. 5, 133, 261, ...
    #
    # `get_hier_coord(linear_idx)` turns that flat slot back into a 2D tile
    # coordinate like `(row, col)`. That lets us use the same coordinate to
    # index both:
    #   - `blkA[coord]` in the GMEM tile
    #   - `sA[coord]`  in the SMEM tile
    for i in cutlass.range_constexpr(COPIES_PER_THR):
        linear_idx = linear_copy((tidx, i))
        coord = sA_layout.get_hier_coord(linear_idx)
        sA[coord] = blkA[coord]

    # All threads must finish filling SMEM before anyone reads from it.
    #
    # Without this barrier, one thread could start reading `sA[...]` while some
    # other thread has not written its portion yet. The barrier makes the staged
    # tile visible to the whole CTA before the SMEM->GMEM phase below starts.
    cute.arch.barrier()

    # For this lesson, write the staged tile back out so we can verify the copy.
    # A real kernel would usually consume `sA` for some compute step here before
    # writing results onward.
    for i in cutlass.range_constexpr(COPIES_PER_THR):
        linear_idx = linear_copy((tidx, i))
        coord = sA_layout.get_hier_coord(linear_idx)
        blkC[coord] = sA[coord]


@cute.jit
def gmem_to_smem_copy(
    mA: cute.Tensor,
    mC: cute.Tensor,
):
    # Tile the full matrix into 32x32 block tiles.
    gA = cute.zipped_divide(mA, (BLK_M, BLK_N))
    gC = cute.zipped_divide(mC, (BLK_M, BLK_N))

    print(f"Block tiler: {(BLK_M, BLK_N)}")
    print(f"Tiled gA: {gA.type}")

    gmem_to_smem_copy_kernel(gA, gC).launch(
        grid=[N // BLK_N, M // BLK_M, 1],
        block=[N_THR, 1, 1],
    )


def run_gmem_to_smem_kernel_demo():
    print("\n== Part 2: Real GMEM -> SMEM -> GMEM Copy ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    # Make the values easy to read after a tiled copy.
    a = torch.arange(M * N, device="cuda", dtype=torch.float16).reshape(M, N)
    c = torch.zeros_like(a)

    mA = from_dlpack(a, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print("---- Starting shapes ----")
    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}\n")

    expected = a.clone()

    kernel = cute.compile(gmem_to_smem_copy, mA, mC)
    kernel(mA, mC)
    torch.cuda.synchronize()

    print("---- Results ----")
    print("input A, top-left 4x8:")
    print(a[:4, :8])
    print("\noutput C, top-left 4x8:")
    print(c[:4, :8])
    print(f"\nmatches torch: {torch.equal(c, expected)}")


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_gmem_to_smem_theory()
            return
        if mode == "--kernel":
            run_gmem_to_smem_kernel_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)