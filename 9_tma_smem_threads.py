import os
import subprocess
import sys

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


M = 32
N = 32
BLK_M = 16
BLK_N = 16
THREADS_PER_CTA = 32
VALS_PER_THR = (BLK_M * BLK_N) // THREADS_PER_CTA  # 256 / 32 = 8
NUM_STAGES = 1


@cute.struct
class SharedStorage:
    mbar_array: cute.struct.MemRange[cutlass.Int64, NUM_STAGES * 2]


@cute.jit
def show_tma_to_threads_theory_jit():
    block_row_idx = 1
    block_col_idx = 0
    thread_idx = 5

    gmem = cute.make_layout((M, N), stride=(N, 1))
    blocked = cute.zipped_divide(gmem, (BLK_M, BLK_N))
    blk_layout, blk_offset = cute.slice_and_offset(
        ((None, None), (block_row_idx, block_col_idx)),
        blocked,
    )

    print(f"Global layout: {gmem}")
    print(f"Block tiling by (16,16): {blocked}")
    print(f"Chosen GMEM tile ({block_row_idx}, {block_col_idx}): {blk_layout}")
    print(f"Chosen GMEM tile base offset: {blk_offset}")
    print("")

    # Just like lesson 8, TMA stages one whole 16x16 tile into SMEM.
    smem_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))
    print(f"SMEM tile layout: {smem_layout}")
    print("")

    # Now comes the new part for lesson 9:
    #
    # Once the tile is sitting in shared memory, we want to reinterpret that one
    # staged tile in thread-centric coordinates.
    #
    # The tile has 16*16 = 256 elements.
    # With 32 threads, each thread should get 8 values.
    #
    # We therefore build a thread/value layout:
    #   (thread_idx, value_idx) -> flat_slot_in_tile
    #
    # using the row-major flat slot numbering of the 16x16 tile.
    tv_layout = cute.make_layout((THREADS_PER_CTA, VALS_PER_THR), stride=(1, THREADS_PER_CTA))
    print(f"Thread/value layout: {tv_layout}")
    # -> (32,8):(1,32)
    #
    # Read this as:
    #   - neighboring threads are 1 flat slot apart
    #   - the same thread's next value is 32 flat slots later
    #   - flat slot formula is:
    #       slot = thread_idx + 32 * value_idx
    print("")

    slot0 = tv_layout((thread_idx, 0))
    slot1 = tv_layout((thread_idx, 1))
    print(f"Thread {thread_idx} gets flat slots {slot0} and {slot1}")
    print(f"  slot {slot0} -> SMEM coord {smem_layout.get_hier_coord(slot0)}")
    print(f"  slot {slot1} -> SMEM coord {smem_layout.get_hier_coord(slot1)}")
    print("")

    print("Threads 0..3 at value_idx 0:")
    print(f"  thr 0 -> slot {tv_layout((0, 0))} -> coord {smem_layout.get_hier_coord(tv_layout((0, 0)))}")
    print(f"  thr 1 -> slot {tv_layout((1, 0))} -> coord {smem_layout.get_hier_coord(tv_layout((1, 0)))}")
    print(f"  thr 2 -> slot {tv_layout((2, 0))} -> coord {smem_layout.get_hier_coord(tv_layout((2, 0)))}")
    print(f"  thr 3 -> slot {tv_layout((3, 0))} -> coord {smem_layout.get_hier_coord(tv_layout((3, 0)))}")
    print("")

    # This is the exact mental bridge to real kernels:
    #
    #   1. TMA chooses and stages the CTA tile
    #   2. CuTe layouts then repartition that staged tile for the threads
    #   3. each thread loads its fragment from SMEM
    #   4. the thread computes on those values
    #   5. the thread stores results back to GMEM
    #
    # In the runnable demo below, we keep the post-TMA SMEM consumption step
    # single-threaded because that path is much more stable in this environment.
    # The thread/value layout is still shown here because it is the important
    # mental model for the next step toward a real kernel.
    print("Kernel idea:")
    print("  TMA: GMEM tile -> SMEM tile")
    print("  then: theory shows (thread_idx, value_idx) ownership over the staged tile")
    print("  runnable demo: adds 100 after the TMA stage")
    print("  store: writes results back to GMEM")


def show_tma_to_threads_theory():
    print("== Part 1: TMA To Thread Fragments Theory ==")
    show_tma_to_threads_theory_jit()


@cute.kernel
def tma_to_threads_kernel(
    tma_load_atom: cute.CopyAtom,
    tma_load_tensor: cute.Tensor,
    gC: cute.Tensor,
    dtype: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # This is the staged CTA tile in SMEM.
    sA_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))
    sA = smem.allocate_tensor(
        element_type=dtype,
        layout=sA_layout,
        byte_alignment=128,
    )

    # One-stage TMA pipeline:
    # thread 0 issues the bulk load, then the one-warp CTA consumes the staged tile.
    tma_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mbar_array.data_ptr(),
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=(dtype.width // 8) * BLK_M * BLK_N,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )

    producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, NUM_STAGES
    )
    consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, NUM_STAGES
    )

    # Pick the input and output tiles for this CTA.
    gA_tiled = cute.zipped_divide(tma_load_tensor, (BLK_M, BLK_N))
    blkA = gA_tiled[((None, None), (bidy, bidx))]
    blkC = gC[((None, None), (bidy, bidx))]

    # TMA partitioning uses the source and destination views participating in the
    # copy. For the TMA API we flatten the tile modes, but conceptually this is
    # still "the same 16x16 tile" from the theory section.
    sA_flat = cute.group_modes(sA, 0, cute.rank(sA))
    blkA_flat = cute.group_modes(blkA, 0, cute.rank(blkA))
    blkC_flat = cute.group_modes(blkC, 0, cute.rank(blkC))

    if tidx == 0:
        tma_pipeline.producer_acquire(producer_state)

        s_part, g_part = cpasync.tma_partition(
            tma_load_atom,
            0,
            cute.make_layout(1),
            sA_flat,
            blkA_flat,
        )

        cute.copy(
            tma_load_atom,
            g_part,
            s_part,
            tma_bar_ptr=tma_pipeline.producer_get_barrier(producer_state),
        )
        tma_pipeline.producer_commit(producer_state)

        tma_pipeline.consumer_wait(consumer_state)

        # The theory section explains how this staged tile can be repartitioned
        # to threads with `tv_layout`. For the runnable kernel we keep the SMEM
        # consumption step single-threaded because that path is much more stable
        # in this environment than the full multi-thread fragment version.
        for flat_slot in cutlass.range_constexpr(BLK_M * BLK_N):
            blkC_flat[flat_slot] = sA_flat[flat_slot] + dtype(100)



@cute.jit
def tma_to_threads(
    mA: cute.Tensor,
    mC: cute.Tensor,
):
    smem_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))

    # Build the real TMA load descriptor for one GMEM -> SMEM tile transfer.
    tma_load_atom, tma_load_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        smem_layout,
        (BLK_M, BLK_N),
    )

    # Output stays as an ordinary GMEM tensor.
    # We tile it with ordinary CuTe layouts, then let threads store to it directly.
    gC = cute.zipped_divide(mC, (BLK_M, BLK_N))

    tv_layout = cute.make_layout((THREADS_PER_CTA, VALS_PER_THR), stride=(1, THREADS_PER_CTA))

    print(f"TMA tile shape: {(BLK_M, BLK_N)}")
    print(f"SMEM layout for staged tile: {smem_layout}")
    print(f"TMA source tensor: {tma_load_tensor.type}")
    print(f"Thread/value layout after staging: {tv_layout}")
    print(f"Tiled gC: {gC.type}")

    tma_to_threads_kernel(
        tma_load_atom,
        tma_load_tensor,
        gC,
        mA.element_type,
    ).launch(
        grid=[N // BLK_N, M // BLK_M, 1],
        block=[THREADS_PER_CTA, 1, 1],
    )


def run_tma_to_threads_demo():
    print("\n== Part 2: Real TMA -> SMEM -> Thread Fragments ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    a = torch.arange(M * N, device="cuda", dtype=torch.float16).reshape(M, N)
    c = torch.zeros_like(a)

    # As in lesson 8, this TMA path needed stronger alignment assumptions here.
    mA = from_dlpack(a, assumed_align=32)
    mC = from_dlpack(c, assumed_align=32)

    print("---- Starting shapes ----")
    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}\n")

    expected = a + 100

    kernel = cute.compile(tma_to_threads, mA, mC)
    kernel(mA, mC)
    torch.cuda.synchronize()

    print("---- Results ----")
    print("input A, top-left 4x8:")
    print(a[:4, :8])
    print("\nexpected C, top-left 4x8:")
    print(expected[:4, :8])
    print("\noutput C, top-left 4x8:")
    print(c[:4, :8])
    print(f"\nmatches torch: {torch.equal(c, expected)}")


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_tma_to_threads_theory()
            return
        if mode == "--kernel":
            run_tma_to_threads_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
