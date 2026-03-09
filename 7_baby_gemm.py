import os
import subprocess
import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# Tiny GEMM sizes so the printed layouts stay readable.
M = 16
N = 16
K = 8

# A simple CTA tile shape for GEMM:
#   - one block computes an 8x8 tile of C
#   - each K step consumes a depth-4 slice from A and B
CTA_M = 8
CTA_N = 8
CTA_K = 4

K_TILES = K // CTA_K
THREADS_PER_CTA = CTA_M * CTA_N  # one thread per output element in the tile


@cute.jit
def show_baby_gemm_theory_jit():
    block_row_idx = 1
    block_col_idx = 0
    thread_row_in_tile = 2
    thread_col_in_tile = 3
    tidx = thread_row_in_tile * CTA_N + thread_col_in_tile

    # We use the same storage convention as the Blackwell tutorial GEMM:
    #
    #   A is laid out as (M, K)
    #   B is laid out as (N, K)
    #   C is laid out as (M, N)
    #
    # and the math is:
    #
    #   C[m, n] = sum_k A[m, k] * B[n, k]
    #
    # So B is not stored as (K, N) here. Instead each row of B corresponds to a
    # fixed output column n, and K is still the reduction dimension.
    A = cute.make_layout((M, K), stride=(K, 1))
    B = cute.make_layout((N, K), stride=(K, 1))
    C = cute.make_layout((M, N), stride=(N, 1))

    print(f"A layout (M,K): {A}")
    print(f"B layout (N,K): {B}")
    print(f"C layout (M,N): {C}")
    print(f"CTA tiler (bM, bN, bK): {(CTA_M, CTA_N, CTA_K)}")
    print("")

    # The real Blackwell tutorial uses:
    #
    #   gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    #   gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    #   gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    #
    # which means:
    #   - tile A by (M,K)
    #   - tile B by (N,K)
    #   - tile C by (M,N)
    #   - then pick the CTA's block coordinate
    #
    # Here we show the same idea more explicitly with `zipped_divide(...)`,
    # because the resulting hierarchical layouts are easier to read in a lesson.
    tiled_A = cute.zipped_divide(A, (CTA_M, CTA_K))
    tiled_B = cute.zipped_divide(B, (CTA_N, CTA_K))
    tiled_C = cute.zipped_divide(C, (CTA_M, CTA_N))

    print(f"Tiled A by (bM,bK): {tiled_A}")
    # -> ((8,4),(2,2)):((8,1),(64,4))
    #
    # Read this as:
    #   - one A tile has shape (8 rows, 4 k-values)
    #   - the whole A matrix contains (2, 2) such tiles
    #   - inside one A tile, row stride is 8 and k stride is 1
    #   - moving to the next A tile row jumps by 64 elements
    #   - moving to the next A k-tile jumps by 4 elements

    print(f"Tiled B by (bN,bK): {tiled_B}")
    # -> ((8,4),(2,2)):((8,1),(64,4))
    #
    # This has the same shape pattern as A, except its leading logical mode is N
    # rather than M. A B tile is therefore:
    #   - 8 output columns wide
    #   - 4 k-values deep

    print(f"Tiled C by (bM,bN): {tiled_C}")
    # -> ((8,8),(2,2)):((16,1),(128,8))
    #
    # Read this as:
    #   - one CTA computes an 8x8 output tile of C
    #   - the full output matrix has (2, 2) such CTA tiles
    #   - moving to the next CTA tile row jumps by 8 full rows = 128 elements
    #   - moving to the next CTA tile col jumps by 8 elements
    print("")

    # Pick one CTA: block row 1, block col 0.
    blkC, blkC_offset = cute.slice_and_offset(
        ((None, None), (block_row_idx, block_col_idx)),
        tiled_C,
    )
    print(f"C tile for CTA ({block_row_idx}, {block_col_idx}): {blkC}")
    print(f"C tile base offset: {blkC_offset}")
    # This CTA covers:
    #   rows [8:16)
    #   cols [0:8)
    print("")

    # A GEMM CTA does not use just one A tile and one B tile.
    # It walks across K in chunks of CTA_K.
    #
    # For this example K=8 and CTA_K=4, so there are two K tiles:
    #   k_tile 0 -> k in [0:4)
    #   k_tile 1 -> k in [4:8)
    blkA0, blkA0_offset = cute.slice_and_offset(
        ((None, None), (block_row_idx, 0)),
        tiled_A,
    )
    blkB0, blkB0_offset = cute.slice_and_offset(
        ((None, None), (block_col_idx, 0)),
        tiled_B,
    )
    blkA1, blkA1_offset = cute.slice_and_offset(
        ((None, None), (block_row_idx, 1)),
        tiled_A,
    )
    blkB1, blkB1_offset = cute.slice_and_offset(
        ((None, None), (block_col_idx, 1)),
        tiled_B,
    )

    print(f"A tile for CTA row {block_row_idx}, k_tile 0: {blkA0}")
    print(f"A tile 0 base offset: {blkA0_offset}")
    print(f"B tile for CTA col {block_col_idx}, k_tile 0: {blkB0}")
    print(f"B tile 0 base offset: {blkB0_offset}")
    print("")
    print(f"A tile for CTA row {block_row_idx}, k_tile 1: {blkA1}")
    print(f"A tile 1 base offset: {blkA1_offset}")
    print(f"B tile for CTA col {block_col_idx}, k_tile 1: {blkB1}")
    print(f"B tile 1 base offset: {blkB1_offset}")
    print("")

    # To keep the toy kernel simple, each thread computes exactly one output
    # element inside the CTA's C tile.
    #
    # We therefore map the 64 threads to the 8x8 C tile directly:
    #   thread_idx -> (row_in_tile, col_in_tile)
    thr_output_layout = cute.make_layout((CTA_M, CTA_N), stride=(CTA_N, 1))
    print(f"Thread -> C-tile coordinate layout: {thr_output_layout}")
    print(f"Thread {tidx} computes C-tile coord {(thread_row_in_tile, thread_col_in_tile)}")
    # Thread 19 means:
    #   - row_in_tile = 2
    #   - col_in_tile = 3
    #
    # Because this CTA is block (1, 0), that local coordinate maps to:
    global_row = block_row_idx * CTA_M + thread_row_in_tile
    global_col = block_col_idx * CTA_N + thread_col_in_tile
    print(f"Global output coordinate for that thread: {(global_row, global_col)}")
    print("")

    # So this one thread computes:
    #
    #   C[10, 3] =
    #       dot( A[10, 0:4], B[3, 0:4] ) +
    #       dot( A[10, 4:8], B[3, 4:8] )
    #
    # In other words:
    #   - the CTA chooses the output tile in MxN
    #   - each K tile brings in one A block and one B block
    #   - each thread owns one output point inside the C tile
    print(f"k_tile 0 contributes A[{global_row}, 0:4] with B[{global_col}, 0:4]")
    print(f"k_tile 1 contributes A[{global_row}, 4:8] with B[{global_col}, 4:8]")


def show_baby_gemm_theory():
    print("== Part 1: Baby GEMM Theory ==")
    show_baby_gemm_theory_jit()


@cute.kernel
def baby_gemm_kernel(
    gA_mk: cute.Tensor,
    gB_nk: cute.Tensor,
    gC_mn: cute.Tensor,
    thr_output_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # Map the linear thread id to one output coordinate inside this CTA's C tile.
    row_in_tile, col_in_tile = thr_output_layout.get_hier_coord(tidx)
    c_coord = (row_in_tile, col_in_tile)

    # Slice out the output tile owned by this CTA.
    blkC = gC_mn[((None, None), (bidy, bidx))]

    # This thread accumulates one output value across all K tiles.
    #
    # We keep this kernel intentionally simple:
    #   - no shared memory staging
    #   - no tensor-core MMA fragments
    #   - no pipeline
    #
    # That way the indexing matches the theory exactly.
    acc = 0.0

    for k_tile_idx in cutlass.range_constexpr(K_TILES):
        # For a fixed CTA:
        #   - A uses this CTA's block row and the current k tile
        #   - B uses this CTA's block col and the current k tile
        blkA = gA_mk[((None, None), (bidy, k_tile_idx))]
        blkB = gB_nk[((None, None), (bidx, k_tile_idx))]

        # One C element is a dot product between:
        #   - one row of the A tile
        #   - one row of the B tile
        #
        # We use a row from B because B is stored as (N, K) in this tutorial
        # family, not as (K, N).
        for kk in cutlass.range_constexpr(CTA_K):
            acc += blkA[(row_in_tile, kk)] * blkB[(col_in_tile, kk)]

    # The reduction naturally promotes the accumulator to a wider type here.
    # For this lesson we store back to the original C element type explicitly.
    blkC[c_coord] = acc.to(gC_mn.element_type)


@cute.jit
def baby_gemm(
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
):
    # Tile the full matrices the same way the real GEMM thinks about them:
    #   A -> (bM, bK) tiles
    #   B -> (bN, bK) tiles
    #   C -> (bM, bN) tiles
    gA_mk = cute.zipped_divide(mA_mk, (CTA_M, CTA_K))
    gB_nk = cute.zipped_divide(mB_nk, (CTA_N, CTA_K))
    gC_mn = cute.zipped_divide(mC_mn, (CTA_M, CTA_N))

    # The thread layout is deliberately simple here:
    # each of the 64 threads owns one output coordinate in the 8x8 C tile.
    thr_output_layout = cute.make_layout((CTA_M, CTA_N), stride=(CTA_N, 1))

    print(f"CTA tiler (bM, bN, bK): {(CTA_M, CTA_N, CTA_K)}")
    print(f"Tiled gA: {gA_mk.type}")
    print(f"Tiled gB: {gB_nk.type}")
    print(f"Tiled gC: {gC_mn.type}")
    print(f"Thread -> C tile layout: {thr_output_layout}")

    baby_gemm_kernel(gA_mk, gB_nk, gC_mn, thr_output_layout).launch(
        grid=[N // CTA_N, M // CTA_M, 1],
        block=[THREADS_PER_CTA, 1, 1],
    )


def run_baby_gemm_demo():
    print("\n== Part 2: Tiny runnable GEMM ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    # Keep values small so the GEMM output is easy to inspect by eye.
    a = (torch.arange(M * K, device="cuda", dtype=torch.int32) % 4).to(torch.float16)
    b = (1 + (torch.arange(N * K, device="cuda", dtype=torch.int32) % 4)).to(torch.float16)
    a = a.reshape(M, K)
    b = b.reshape(N, K)
    c = torch.zeros((M, N), device="cuda", dtype=torch.float16)

    # 16-byte alignment matches common vector-friendly global-memory assumptions.
    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print("---- Starting shapes ----")
    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mB shape: {mB.shape}, stride: {mB.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}\n")

    expected = torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

    kernel = cute.compile(baby_gemm, mA, mB, mC)
    kernel(mA, mB, mC)
    torch.cuda.synchronize()

    print("---- Results ----")
    print("input A, top-left 4x4:")
    print(a[:4, :4])
    print("\ninput B, top-left 4x4:")
    print(b[:4, :4])
    print("\nexpected C, top-left 4x4:")
    print(expected[:4, :4])
    print("\noutput C, top-left 4x4:")
    print(c[:4, :4])
    print(f"\nmatches torch: {torch.equal(c, expected)}")


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_baby_gemm_theory()
            return
        if mode == "--kernel":
            run_baby_gemm_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
