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
THREADS_PER_CTA = 32  # one warp is enough because one thread issues the TMA ops
NUM_STAGES = 1


@cute.struct
class SharedStorage:
    # This is the little block of CTA-shared bookkeeping that lives in SMEM.
    #
    # TMA copies are asynchronous: one thread can issue the GMEM -> SMEM copy,
    # and other threads need a safe way to know when that staged tile is ready.
    #
    # `mbar_array` holds the memory barriers used by the pipeline. For each
    # stage, CuTe keeps barrier state in shared memory so the producer side
    # (issuing TMA) and the consumer side (reading the staged tile) can
    # synchronize correctly.
    #
    # The `* 2` here is about storage size, not "one for producer and one for
    # consumer": one mbarrier object takes 16 bytes, which is 2 x `Int64`
    # entries in shared memory.
    #
    # With `NUM_STAGES = 1`, this lesson only needs one stage's worth of
    # barriers, but the array is still written in the general staged form.
    mbar_array: cute.struct.MemRange[cutlass.Int64, NUM_STAGES * 2]


@cute.jit
def show_tma_theory_jit():
    block_row_idx = 1
    block_col_idx = 0

    gmem = cute.make_layout((M, N), stride=(N, 1))
    print(f"Global layout: {gmem}")
    # -> (32,32):(32,1)
    #
    # This is the same row-major matrix story as the earlier lessons:
    #   - moving down one row jumps by 32 elements
    #   - moving across one column jumps by 1 element

    blocked = cute.zipped_divide(gmem, (BLK_M, BLK_N))
    print(f"Tiled GMEM by (16,16): {blocked}")
    # -> ((16,16),(2,2)):((32,1),(512,16))
    #
    # Read this as:
    #   - one CTA tile has shape (16, 16)
    #   - the full matrix contains (2, 2) such tiles
    #   - inside one tile, addressing is still row-major with stride (32, 1)
    #   - moving to the next tile row jumps by 16 full rows
    #   - moving to the next tile col jumps by 16 elements

    blk_layout, blk_offset = cute.slice_and_offset(
        ((None, None), (block_row_idx, block_col_idx)),
        blocked,
    )
    print(f"Chosen GMEM tile ({block_row_idx}, {block_col_idx}): {blk_layout}")
    print(f"Chosen GMEM tile base offset: {blk_offset}")
    print("")

    # The SMEM tile has the same logical shape, but now its addresses are local
    # to shared memory rather than global memory.
    smem_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))
    print(f"SMEM tile layout: {smem_layout}")
    # -> (16,16):(16,1)
    #
    # Read this as:
    #   - the staged tile is also 16x16
    #   - inside SMEM, moving down one row jumps by 16 elements
    #   - moving across one column jumps by 1 element
    print("")

    # The key TMA idea is different from lesson 3:
    #
    #   lesson 3:
    #     threads explicitly owned copy coordinates and each thread copied its
    #     own elements
    #
    #   lesson 8:
    #     we still need the source and destination layouts, but we hand the whole
    #     tile description to TMA as one bulk transfer
    #
    # So the important CuTe objects become:
    #   1. the source tensor layout in GMEM
    #   2. the destination layout in SMEM
    #   3. the tile shape `(16,16)`
    #   4. the CTA coordinate `(block_row_idx, block_col_idx)`
    #
    # TMA then moves that tile for us, instead of us writing an explicit
    # per-thread copy loop.
    print("TMA mental model:")
    print("  source tile : GMEM tile (16x16) at CTA coordinate (1,0)")
    print("  dest tile   : SMEM tile (16x16)")
    print("  copy style  : bulk tile transfer, not per-thread scalar copies")
    print("")

    # The round trip in the kernel will therefore be:
    #   1. pick one GMEM tile from A
    #   2. use TMA to load that whole tile into SMEM
    #   3. use TMA again to store the same SMEM tile back to the matching tile in C
    #
    # The layouts are still essential:
    #   - GMEM layout says where the source tile lives
    #   - SMEM layout says how the staged tile is arranged
    #   - tiler says how big a bulk transfer should be
    print(f"Round-trip copy target in C: tile ({block_row_idx}, {block_col_idx})")


def show_tma_theory():
    print("== Part 1: TMA Theory ==")
    show_tma_theory_jit()


@cute.kernel
def tma_roundtrip_kernel(
    # A `CopyAtom` is the smallest "copy recipe" object in CuTe:
    # it packages up which hardware copy instruction to use plus the layout/
    # tiling information needed to apply that instruction to this tensor tile.
    #
    # Here:
    #   - `tma_load_atom` means "copy one tile from GMEM to SMEM with TMA"
    #   - `tma_store_atom` means "copy one tile from SMEM back to GMEM with TMA"
    tma_load_atom: cute.CopyAtom,
    tma_load_tensor: cute.Tensor,
    tma_store_atom: cute.CopyAtom,
    tma_store_tensor: cute.Tensor,
    dtype: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # This is the tile that TMA loads into and stores out of.
    sA_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))
    sA = smem.allocate_tensor(
        element_type=dtype,
        layout=sA_layout,
        byte_alignment=128,
    )

    # We use the smallest possible TMA pipeline:
    #   - 1 producer thread issues the TMA transaction
    #   - 1 consumer thread waits for completion
    #   - only 1 stage because this lesson performs just one tile transfer
    tma_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mbar_array.data_ptr(),
        num_stages=NUM_STAGES,
        # One thread plays the "producer" role: it issues the TMA transaction.
        # In larger kernels this could be a bigger cooperative group, but for
        # this lesson a single thread is enough.
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        # One thread also plays the "consumer" role: it waits until the staged
        # tile is ready before allowing the lesson to continue.
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        # `tx_count` is the number of bytes that one TMA transfer will move.
        # Here that is:
        #   bytes per element * elements in one 16x16 tile
        # = (dtype.width // 8) * BLK_M * BLK_N
        tx_count=(dtype.width // 8) * BLK_M * BLK_N,
        # `cta_layout_vmnk` describes the CTA cluster layout seen by the
        # pipeline. `(1,1,1,1)` means this lesson uses the simplest case:
        # one CTA, no multicast, no extra cluster structure to reason about.
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )

    producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, NUM_STAGES
    )
    consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, NUM_STAGES
    )

    # Partition the global tensors into 16x16 CTA tiles and pick the tile for
    # this CTA. This is still ordinary CuTe tiling; TMA acts on the chosen tile.
    gA_tiled = cute.zipped_divide(tma_load_tensor, (BLK_M, BLK_N))
    gC_tiled = cute.zipped_divide(tma_store_tensor, (BLK_M, BLK_N))
    gA_tile = gA_tiled[((None, None), (bidy, bidx))]
    gC_tile = gC_tiled[((None, None), (bidy, bidx))]

    # TMA partitioning wants the actual source and destination pieces that
    # participate in the transfer. For this small example we flatten the tile
    # modes just for the copy API, while keeping the conceptual explanation in 2D.
    sA_flat = cute.group_modes(sA, 0, cute.rank(sA))
    gA_tile_flat = cute.group_modes(gA_tile, 0, cute.rank(gA_tile))
    gC_tile_flat = cute.group_modes(gC_tile, 0, cute.rank(gC_tile))

    if tidx == 0:
        # ------------------------------------------------------------------
        # 1. GMEM -> SMEM via TMA load
        # ------------------------------------------------------------------
        tma_pipeline.producer_acquire(producer_state)

        # `tma_partition(...)` specializes the generic copy atom to this
        # particular tile instance.
        #
        # It returns matching source/destination pieces:
        #   - `g_load_part`: the GMEM tile view this TMA load should read
        #   - `s_load_part`: the SMEM tile view this TMA load should fill
        #
        # So after this step, `cute.copy(tma_load_atom, ...)` is no longer
        # "copy some generic tile", but "copy this exact GMEM tile into this
        # exact SMEM tile".
        s_load_part, g_load_part = cpasync.tma_partition(
            tma_load_atom,
            0,
            cute.make_layout(1),
            sA_flat,
            gA_tile_flat,
        )

        cute.copy(
            tma_load_atom,
            g_load_part,
            s_load_part,
            tma_bar_ptr=tma_pipeline.producer_get_barrier(producer_state),
        )
        tma_pipeline.producer_commit(producer_state)

        # Wait until the TMA engine has finished populating SMEM.
        tma_pipeline.consumer_wait(consumer_state)

        # ------------------------------------------------------------------
        # 2. SMEM -> GMEM via TMA store
        # ------------------------------------------------------------------
        # This fence makes the shared-memory view visible to the async proxy
        # used by the outgoing bulk store.
        cute.arch.fence_proxy("async.shared", space="cta")

        # Same idea for the store atom: partition the generic SMEM -> GMEM
        # recipe down to the specific staged tile in SMEM and the specific
        # destination tile in GMEM for this CTA.
        s_store_part, g_store_part = cpasync.tma_partition(
            tma_store_atom,
            0,
            cute.make_layout(1),
            sA_flat,
            gC_tile_flat,
        )

        cute.copy(tma_store_atom, s_store_part, g_store_part)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)

        tma_pipeline.consumer_release(consumer_state)


@cute.jit
def tma_roundtrip(
    mA: cute.Tensor,
    mC: cute.Tensor,
):
    smem_layout = cute.make_layout((BLK_M, BLK_N), stride=(BLK_N, 1))

    # Build the real TMA descriptors:
    #   - one for loading a GMEM tile into a same-shaped SMEM tile
    #   - one for storing that SMEM tile back out to GMEM
    #
    # `make_tiled_tma_atom(...)` returns:
    #   - a `CopyAtom`, which is the copy instruction/configuration object
    #   - a tensor view whose coordinates match what that TMA copy expects
    tma_load_atom, tma_load_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        smem_layout,
        (BLK_M, BLK_N),
    )
    tma_store_atom, tma_store_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        mC,
        smem_layout,
        (BLK_M, BLK_N),
    )

    print(f"TMA tile shape: {(BLK_M, BLK_N)}")
    print(f"SMEM layout for TMA: {smem_layout}")
    print(f"TMA source tensor: {tma_load_tensor.type}")
    print(f"TMA dest tensor: {tma_store_tensor.type}")
    # The printed `coord_tensor` objects are the TMA descriptors' tensor views:
    # they describe the global tensor shape/stride that the TMA engine will use
    # when we later select one CTA tile and issue the bulk transfer.

    tma_roundtrip_kernel(
        tma_load_atom,
        tma_load_tensor,
        tma_store_atom,
        tma_store_tensor,
        mA.element_type,
    ).launch(
        grid=[N // BLK_N, M // BLK_M, 1],
        block=[THREADS_PER_CTA, 1, 1],
    )


def run_tma_roundtrip_demo():
    print("\n== Part 2: Real TMA GMEM -> SMEM -> GMEM ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    a = torch.arange(M * N, device="cuda", dtype=torch.float16).reshape(M, N)
    c = torch.zeros_like(a)

    # TMA ended up needing stricter alignment than the earlier scalar-copy lessons
    # in this environment. `assumed_align=32` and the 128-byte SMEM alignment used
    # above were enough to make the round-trip copy reliable here.
    mA = from_dlpack(a, assumed_align=32)
    mC = from_dlpack(c, assumed_align=32)

    print("---- Starting shapes ----")
    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}\n")

    expected = a.clone()

    kernel = cute.compile(tma_roundtrip, mA, mC)
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
            show_tma_theory()
            return
        if mode == "--kernel":
            run_tma_roundtrip_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
