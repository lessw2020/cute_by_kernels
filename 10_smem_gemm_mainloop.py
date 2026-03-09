import os
import subprocess
import sys

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack


# Small GEMM so the printed CuTe layouts stay readable.
M = 16
N = 16
K = 8

# One CTA computes an 8x8 tile of C.
# Each mainloop step consumes a k-depth of 4.
CTA_M = 8
CTA_N = 8
CTA_K = 4

K_TILES = K // CTA_K
THREADS_PER_CTA = CTA_M * CTA_N  # one thread per output element of the C tile
A_STAGE_ELEMS = CTA_M * CTA_K
B_STAGE_ELEMS = CTA_N * CTA_K


@cute.jit
def show_smem_gemm_mainloop_theory_jit():
    block_row_idx = 1
    block_col_idx = 0
    thread_row_in_tile = 2
    thread_col_in_tile = 3
    tidx = thread_row_in_tile * CTA_N + thread_col_in_tile

    # Match the tensor conventions used in the Blackwell tutorial GEMM:
    #
    #   A : (M, K)
    #   B : (N, K)
    #   C : (M, N)
    #
    # and:
    #
    #   C[m, n] = sum_k A[m, k] * B[n, k]
    #
    # So B is stored as (N, K), not (K, N).
    A = cute.make_layout((M, K), stride=(K, 1))
    B = cute.make_layout((N, K), stride=(K, 1))
    C = cute.make_layout((M, N), stride=(N, 1))

    print(f"A layout (M,K): {A}")
    print(f"B layout (N,K): {B}")
    print(f"C layout (M,N): {C}")
    print(f"CTA tiler (bM, bN, bK): {(CTA_M, CTA_N, CTA_K)}")
    print("")

    # The real tutorial writes this step with `local_tile(...)`.
    # Here we keep the same idea but show the hierarchical layouts explicitly.
    tiled_A = cute.zipped_divide(A, (CTA_M, CTA_K))
    tiled_B = cute.zipped_divide(B, (CTA_N, CTA_K))
    tiled_C = cute.zipped_divide(C, (CTA_M, CTA_N))

    print(f"Tiled A by (bM,bK): {tiled_A}")
    print(f"Tiled B by (bN,bK): {tiled_B}")
    print(f"Tiled C by (bM,bN): {tiled_C}")
    print("")

    # Pick CTA (1, 0), i.e. bottom-left output tile.
    blkC, blkC_offset = cute.slice_and_offset(
        ((None, None), (block_row_idx, block_col_idx)),
        tiled_C,
    )
    print(f"C tile for CTA ({block_row_idx}, {block_col_idx}): {blkC}")
    print(f"C tile base offset: {blkC_offset}")
    print("")

    # This CTA will march across K in 2 steps:
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
    print(f"A tile for CTA row {block_row_idx}, k_tile 0: {blkA0}")
    print(f"A tile 0 base offset: {blkA0_offset}")
    print(f"B tile for CTA col {block_col_idx}, k_tile 0: {blkB0}")
    print(f"B tile 0 base offset: {blkB0_offset}")
    print("")

    # In a real GEMM mainloop, those GMEM tiles are usually staged into SMEM
    # before the threads consume them.
    sA_layout = cute.make_layout((CTA_M, CTA_K), stride=(CTA_K, 1))
    sB_layout = cute.make_layout((CTA_N, CTA_K), stride=(CTA_K, 1))
    print(f"SMEM A tile layout: {sA_layout}")
    print(f"SMEM B tile layout: {sB_layout}")
    print("")

    # That staged tile is then consumed in thread-centric terms.
    # For this first mainloop lesson, we keep the thread mapping very simple:
    #
    #   thread_idx -> one output coordinate (row_in_tile, col_in_tile)
    thrC_layout = cute.make_layout((CTA_M, CTA_N), stride=(CTA_N, 1))
    print(f"Thread -> C-tile coordinate layout: {thrC_layout}")
    print(f"Thread {tidx} computes local C coord {(thread_row_in_tile, thread_col_in_tile)}")
    global_row = block_row_idx * CTA_M + thread_row_in_tile
    global_col = block_col_idx * CTA_N + thread_col_in_tile
    print(f"Global output coord for that thread: {(global_row, global_col)}")
    print("")

    # For k_tile 0, this one thread reads:
    #
    #   sA[row_in_tile, kk] for kk = 0..3
    #   sB[col_in_tile, kk] for kk = 0..3
    #
    # and accumulates:
    #
    #   partial += sA[row_in_tile, kk] * sB[col_in_tile, kk]
    #
    # Since B is stored as (N, K), its first mode is output-column n.
    print(f"For k_tile 0, thread {tidx} multiplies:")
    print(f"  A row fragment: A[{global_row}, 0:4]")
    print(f"  B row fragment: B[{global_col}, 0:4]")
    print("")

    print("Mainloop story:")
    print("  1. choose CTA tile of C")
    print("  2. choose matching A and B tiles for one k step")
    print("  3. stage A and B into SMEM")
    print("  4. each thread reads its row/col fragment from SMEM")
    print("  5. accumulate into one C value")
    print("  6. repeat for the next k tile")
    print("")

    # This is also the bridge to the real Blackwell tutorial:
    #
    #   gA/gB/gC        : CTA-local tiles in GMEM
    #   sA/sB           : staged tiles in SMEM
    #   partition_A/B/C : finer MMA-specific repartitioning
    #   make_fragment_* : register fragments
    #
    # Lesson 10 stops at the first stable mainloop shape:
    # CTA tiles -> SMEM tiles -> per-thread accumulation.
    print("Real tutorial bridge:")
    print("  this lesson stops before `partition_A/B/C` and tensor-core fragments")
    print("  but it matches the same CTA tile -> staged tile -> consume pattern")


def show_smem_gemm_mainloop_theory():
    print("== Part 1: SMEM GEMM Mainloop Theory ==")
    show_smem_gemm_mainloop_theory_jit()


@cute.kernel
def smem_gemm_mainloop_kernel(
    gA_mk: cute.Tensor,
    gB_nk: cute.Tensor,
    gC_mn: cute.Tensor,
    thrC_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # Each thread owns one output coordinate inside the CTA's 8x8 C tile.
    row_in_tile, col_in_tile = thrC_layout.get_hier_coord(tidx)
    c_coord = (row_in_tile, col_in_tile)

    # Output tile for this CTA.
    blkC = gC_mn[((None, None), (bidy, bidx))]

    # Allocate staged A and B tiles in SMEM.
    smem = utils.SmemAllocator()
    sA_layout = cute.make_layout((CTA_M, CTA_K), stride=(CTA_K, 1))
    sB_layout = cute.make_layout((CTA_N, CTA_K), stride=(CTA_K, 1))
    sA = smem.allocate_tensor(
        element_type=gA_mk.element_type,
        layout=sA_layout,
        byte_alignment=16,
    )
    sB = smem.allocate_tensor(
        element_type=gB_nk.element_type,
        layout=sB_layout,
        byte_alignment=16,
    )

    acc = 0.0

    for k_tile_idx in cutlass.range_constexpr(K_TILES):
        # Pick the A and B tiles for this CTA and this k step.
        blkA = gA_mk[((None, None), (bidy, k_tile_idx))]
        blkB = gB_nk[((None, None), (bidx, k_tile_idx))]

        # Stage the whole A tile and B tile into SMEM.
        #
        # A has 8*4 = 32 elements.
        # B has 8*4 = 32 elements.
        #
        # With 64 threads total:
        #   - threads 0..31 load A
        #   - threads 32..63 load B
        if tidx < A_STAGE_ELEMS:
            a_coord = sA_layout.get_hier_coord(tidx)
            sA[a_coord] = blkA[a_coord]
        else:
            b_slot = tidx - A_STAGE_ELEMS
            b_coord = sB_layout.get_hier_coord(b_slot)
            sB[b_coord] = blkB[b_coord]

        # Everyone must see the staged tiles before starting the multiply-accumulate.
        cute.arch.sync_threads()

        # Each thread computes one output value:
        #
        #   C[row_in_tile, col_in_tile] +=
        #       sum_kk sA[row_in_tile, kk] * sB[col_in_tile, kk]
        #
        # Notice the access pattern:
        #   - A contributes one row fragment from the staged A tile
        #   - B contributes one row fragment from the staged B tile
        #
        # That matches the (N,K) storage of B used in the tutorial family.
        for kk in cutlass.range_constexpr(CTA_K):
            acc += sA[(row_in_tile, kk)] * sB[(col_in_tile, kk)]

        # Do not start overwriting sA/sB for the next k tile until all threads
        # have finished reading the current staged tiles.
        cute.arch.sync_threads()

    blkC[c_coord] = acc.to(gC_mn.element_type)


@cute.jit
def smem_gemm_mainloop(
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
):
    # Tile the full tensors exactly the same way as the theory section:
    #   A -> (bM, bK)
    #   B -> (bN, bK)
    #   C -> (bM, bN)
    gA_mk = cute.zipped_divide(mA_mk, (CTA_M, CTA_K))
    gB_nk = cute.zipped_divide(mB_nk, (CTA_N, CTA_K))
    gC_mn = cute.zipped_divide(mC_mn, (CTA_M, CTA_N))

    # Thread mapping for the output tile:
    # one thread computes one (row_in_tile, col_in_tile) point of C.
    thrC_layout = cute.make_layout((CTA_M, CTA_N), stride=(CTA_N, 1))

    print(f"CTA tiler (bM, bN, bK): {(CTA_M, CTA_N, CTA_K)}")
    print(f"Tiled gA: {gA_mk.type}")
    print(f"Tiled gB: {gB_nk.type}")
    print(f"Tiled gC: {gC_mn.type}")
    print(f"Thread -> C tile layout: {thrC_layout}")

    smem_gemm_mainloop_kernel(gA_mk, gB_nk, gC_mn, thrC_layout).launch(
        grid=[N // CTA_N, M // CTA_M, 1],
        block=[THREADS_PER_CTA, 1, 1],
    )


def run_smem_gemm_mainloop_demo():
    print("\n== Part 2: Tiny SMEM GEMM Mainloop ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    # Keep values small and repetitive so the output is easy to inspect.
    a = (torch.arange(M * K, device="cuda", dtype=torch.int32) % 4).to(torch.float16)
    b = (1 + (torch.arange(N * K, device="cuda", dtype=torch.int32) % 4)).to(torch.float16)
    a = a.reshape(M, K)
    b = b.reshape(N, K)
    c = torch.zeros((M, N), device="cuda", dtype=torch.float16)

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print("---- Starting shapes ----")
    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mB shape: {mB.shape}, stride: {mB.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}\n")

    expected = torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

    kernel = cute.compile(smem_gemm_mainloop, mA, mB, mC)
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
            show_smem_gemm_mainloop_theory()
            return
        if mode == "--kernel":
            run_smem_gemm_mainloop_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
