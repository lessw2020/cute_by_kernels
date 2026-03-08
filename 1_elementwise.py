import os
import subprocess
import sys

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.jit
def show_layout_theory_jit():
    n = 1024
    block_size = 256
    block_idx_x = 2

    gmem = cute.make_layout(n, stride=1)
    print(f"Global layout: {gmem}")
    # -> 1024:1

    blocked = cute.zipped_divide(gmem, (block_size,))
    print(f"After block divide: {blocked}")
    # -> ((256),(4)):((1),(256))

    blk_layout, blk_offset = cute.slice_and_offset((None, block_idx_x), blocked)
    print(f"Block {block_idx_x} layout: {blk_layout}")
    print(f"Block {block_idx_x} base offset: {blk_offset}")
    # -> ((256)):((1))
    # -> 512

    threaded = cute.zipped_divide(blk_layout, (1,))
    print(f"After thread divide: {threaded}")
    # -> ((1),(256)):((0),(1))

    local_idx = threaded((0, 0))
    print(f"  thread 0 -> global offset {blk_offset + local_idx}")
    local_idx = threaded((0, 1))
    print(f"  thread 1 -> global offset {blk_offset + local_idx}")
    local_idx = threaded((0, 2))
    print(f"  thread 2 -> global offset {blk_offset + local_idx}")
    local_idx = threaded((0, 3))
    print(f"  thread 3 -> global offset {blk_offset + local_idx}")


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

    # Match the notebook's TV-layout construction.
    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
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

    # This TV layout covers one exact (64, 512) tile, so we pick that shape
    # to avoid boundary predicates in the minimal example.
    m = 64
    n = 512

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
