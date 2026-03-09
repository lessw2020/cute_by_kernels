import os
import subprocess
import sys

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


N_BLK = 256
N_THR = 64
ELEMS_PER_THR = N_BLK // N_THR  # = 4


@cute.jit
def show_interleaving_theory_jit():
    # One block covers 256 elements. We want 64 threads, 4 elements each.
    blk = cute.make_layout(N_BLK, stride=1)
    print(f"Block layout: {blk}")
    # -> 256:1
    #
    # Read this as:
    #   - the logical block contains 256 1D elements
    #   - consecutive coordinates are 1 element apart in memory

    print("\nOption A: contiguous ownership")
    contiguous = cute.zipped_divide(blk, (ELEMS_PER_THR,))
    print(f"Contiguous: {contiguous}")
    # -> ((4),(64)):((1),(4))
    #
    # Read this as:
    #   - each thread owns a contiguous tile of 4 elements
    #   - there are 64 such tiles
    #   - inside one thread's tile, stride is 1
    #   - moving to the next thread's tile jumps by 4
    #
    # So thread 0 owns [0,1,2,3], thread 1 owns [4,5,6,7], etc.
    print(f"  thr 0, elem 0 -> offset {contiguous((0, 0))}")
    print(f"  thr 0, elem 1 -> offset {contiguous((1, 0))}")
    print(f"  thr 0, elem 2 -> offset {contiguous((2, 0))}")
    print(f"  thr 0, elem 3 -> offset {contiguous((3, 0))}")
    print(f"  thr 1, elem 0 -> offset {contiguous((0, 1))}")
    print(f"  thr 1, elem 1 -> offset {contiguous((1, 1))}")
    print(f"  thr 1, elem 2 -> offset {contiguous((2, 1))}")
    print(f"  thr 1, elem 3 -> offset {contiguous((3, 1))}")

    print("\nOption B: interleaved ownership")
    interleaved = cute.zipped_divide(blk, (N_THR,))
    print(f"Interleaved: {interleaved}")
    # -> ((64),(4)):((1),(64))
    #
    # Read this as:
    #   - the first mode is which thread inside one iteration
    #   - the second mode is which iteration that thread is handling
    #   - inside one iteration, neighboring threads are 1 apart
    #   - moving to the next iteration for the same thread jumps by 64
    #
    # So thread 0 owns [0,64,128,192], thread 1 owns [1,65,129,193], etc.
    print(f"  thr 0, iter 0 -> offset {interleaved((0, 0))}")
    print(f"  thr 0, iter 1 -> offset {interleaved((0, 1))}")
    print(f"  thr 0, iter 2 -> offset {interleaved((0, 2))}")
    print(f"  thr 0, iter 3 -> offset {interleaved((0, 3))}")
    print(f"  thr 1, iter 0 -> offset {interleaved((1, 0))}")
    print(f"  thr 1, iter 1 -> offset {interleaved((1, 1))}")
    print(f"  thr 1, iter 2 -> offset {interleaved((1, 2))}")
    print(f"  thr 1, iter 3 -> offset {interleaved((1, 3))}")

    print("\nWhy interleaving helps")
    print("  contiguous, iter 0 across threads: 0, 4, 8, 12, ...")
    print("  interleaved, iter 0 across threads: 0, 1, 2, 3, ...")
    print("  interleaved keeps threads in the same iteration on consecutive addresses.")


def show_interleaving_theory():
    print("== Part 1: Thread Interleaving Theory ==")
    show_interleaving_theory_jit()


@cute.kernel
def interleaved_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # `gA`, `gB`, and `gC` are block-tiled views of the original vectors:
    #   ((coord_inside_block), which_block)
    # Slicing with `(None, bidx)` picks the block handled by this CTA.
    blkA = gA[(None, bidx)]
    blkB = gB[(None, bidx)]
    blkC = gC[(None, bidx)]

    # `tv_layout` maps:
    #   (thread_idx, iter_idx) -> coord_inside_block
    #
    # Composing the block-local tensor with that layout rewrites the tensor into
    # thread-centric coordinates:
    #   (thread_idx, iter_idx) -> address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    # Select one thread's whole fragment. `None` means "keep all iterations for
    # this thread", so each thread gets 4 interleaved values.
    thrA = tidfrgA[(tidx, None)]
    thrB = tidfrgB[(tidx, None)]
    thrC = tidfrgC[(tidx, None)]

    thrC[None] = thrA.load() + thrB.load()


@cute.jit
def interleaved_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # First tile the 1D vector into 256-element blocks.
    gA = cute.zipped_divide(mA, (N_BLK,))
    gB = cute.zipped_divide(mB, (N_BLK,))
    gC = cute.zipped_divide(mC, (N_BLK,))
    #
    # For a 1024-element input, this produces 4 blocks:
    #
    #   gA : ((256), (4)) : ((1), (256))
    #
    # Read this as:
    #   - each block contains 256 contiguous elements
    #   - there are 4 such blocks
    #   - inside one block, neighboring elements are stride-1
    #   - moving to the next block jumps by 256 elements

    # Build the interleaved thread/value layout:
    #   (thread_idx, iter_idx) -> thread_idx + iter_idx * 64
    tv_layout = cute.make_layout((N_THR, ELEMS_PER_THR), stride=(1, N_THR))
    #
    # So:
    #   - thread dimension stride = 1   -> neighboring threads touch neighboring addresses
    #   - iteration dimension stride = 64 -> the same thread revisits memory every 64 elements
    #
    # This is the key interleaving pattern from part 1.

    print(f"Block tiler: {(N_BLK,)}")
    print(f"Thread/value layout: {tv_layout}")
    print(f"Tiled gA: {gA.type}")
    # The printed `gA.type` is the low-level memref form of the same idea:
    # a 256-element block view repeated across 4 blocks.

    interleaved_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[N_THR, 1, 1],
    )


def run_interleaved_kernel_demo():
    print("\n== Part 2: Real gmem tensors with interleaved threads ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    # Four 256-element blocks so we can see the same structure as the theory.
    n = 4 * N_BLK

    # Make the data easy to read: A counts up, B is constant 100.
    a = torch.arange(n, device="cuda", dtype=torch.float16)
    b = torch.full((n,), 100, device="cuda", dtype=torch.float16)
    c = torch.zeros_like(a)

    # Assume 16-byte alignment to match common vectorized global-memory access.
    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    print("---- Starting shapes ----")
    print(f"mA shape: {mA.shape}, stride: {mA.stride}")
    print(f"mB shape: {mB.shape}, stride: {mB.stride}")
    print(f"mC shape: {mC.shape}, stride: {mC.stride}\n")

    expected = a + b

    interleaved = cute.compile(interleaved_add, mA, mB, mC)
    interleaved(mA, mB, mC)
    torch.cuda.synchronize()

    print("---- Results ----")
    print("input A, first 16:", a[:16].tolist())
    print("input B, first 16:", b[:16].tolist())
    print("")
    print("expected, first 16:", expected[:16].tolist())
    print("output C, first 16:", c[:16].tolist())
    print(f"\nmatches torch: {torch.equal(c, expected)}")


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_interleaving_theory()
            return
        if mode == "--kernel":
            run_interleaved_kernel_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)