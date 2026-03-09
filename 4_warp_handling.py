import os
import subprocess
import sys
import time

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# One CTA handles 512 logical elements with 128 threads.
# That means:
#   - 4 warps per CTA
#   - 32 lanes per warp
#   - 4 elements per thread
N_BLK = 512
N_WARPS = 4
WARP_SIZE = 32
N_THR = N_WARPS * WARP_SIZE
ELEMS_PER_THR = N_BLK // N_THR


@cute.jit
def show_warp_handling_theory_jit():
    blk = cute.make_layout(N_BLK, stride=1)

    # A flat thread decomposition only remembers:
    #   - which thread
    #   - which iteration of that thread
    #
    # It does not preserve the "warp, then lane" hierarchy inside mode 0.
    flat = cute.zipped_divide(blk, (N_THR,))
    print(f"Flat tile: {flat}")
    # -> ((128),(4)):((1),(128))
    #
    # Read this as:
    #   - 128 logical threads
    #   - each thread performs 4 iterations
    #   - mode 0 is just a flat thread id, not (lane, warp)
    print("")

    # A hierarchical decomposition says:
    #   - first split threads into (lane, warp)
    #   - then keep the iteration dimension separate
    #
    # So instead of a flat `(thread, iter)` layout, we explicitly build:
    #   `((lane, warp), iter) -> linear offset`
    #
    # with:
    #   offset = lane + warp*32 + iter*128
    hier = cute.make_layout(
        ((WARP_SIZE, N_WARPS), ELEMS_PER_THR),
        stride=((1, WARP_SIZE), N_THR),
    )
    print(f"Hierarchical tile: {hier}")
    # -> (((32,4)),(4)):(((1,32)),(128))
    #
    # Read this as:
    #   - mode 0 is now (lane, warp)
    #   - mode 1 is the per-thread iteration
    #   - warp structure is explicit in the layout, not reconstructed by hand
    #   - stride (1, 32) inside mode 0 means:
    #       * moving by +1 lane advances by 1 element
    #       * moving by +1 warp advances by 32 elements
    #   - stride 128 for mode 1 means:
    #       * moving to the next iteration jumps by 128 elements
    #         because one full CTA worth of threads has been skipped
    print("")

    lane_id = 5
    warp_id = 2
    print(f"Warp {warp_id}, lane {lane_id} accesses:")
    print(f"  iter 0 -> offset {lane_id + warp_id * WARP_SIZE + 0 * N_THR}")
    print(f"  iter 1 -> offset {lane_id + warp_id * WARP_SIZE + 1 * N_THR}")
    print(f"  iter 2 -> offset {lane_id + warp_id * WARP_SIZE + 2 * N_THR}")
    print(f"  iter 3 -> offset {lane_id + warp_id * WARP_SIZE + 3 * N_THR}")
    print("")

    print("How to read those offsets:")
    print("  lane 5 of warp 2 starts at 5 + 2*32 = 69")
    print("  each later iteration jumps by 128 because there are 128 threads total")
    print("")

    print("Why this is useful:")
    print("  warp 2, iter 0 owns offsets 64..95")
    print("  warp 2, iter 1 owns offsets 192..223")
    print("  warp 2, iter 2 owns offsets 320..351")
    print("  warp 2, iter 3 owns offsets 448..479")
    print("")

    print("So each warp owns contiguous 32-element chunks.")
    print("That matches warp-centric operations like shuffle and warp reductions much")
    print("better than a flat thread id that hides the warp/lane split.")


def show_warp_handling_theory():
    print("== Part 1: Warp Handling Theory ==")
    show_warp_handling_theory_jit()


@cute.kernel
def warp_handling_kernel(
    warp_out: cute.Tensor,
    lane_out: cute.Tensor,
    iter_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = tidx % WARP_SIZE

    # Build the same hierarchical layout used in the theory section.
    # It maps:
    #   ((lane, warp), iter) -> offset
    # with:
    #   offset = lane + warp*32 + iter*128
    hier = cute.make_layout(
        ((WARP_SIZE, N_WARPS), ELEMS_PER_THR),
        stride=((1, WARP_SIZE), N_THR),
    )

    # Each thread writes out:
    #   - which warp owns an offset
    #   - which lane owns an offset
    #   - which per-thread iteration produced that offset
    #
    # This makes the hierarchical ownership visible in ordinary tensors.
    for i in cutlass.range_constexpr(ELEMS_PER_THR):
        offset = hier(((lane_idx, warp_idx), i))
        warp_out[offset] = warp_idx
        lane_out[offset] = lane_idx
        iter_out[offset] = i


@cute.jit
def run_warp_handling_kernel(
    mWarp: cute.Tensor,
    mLane: cute.Tensor,
    mIter: cute.Tensor,
):
    # These outputs are simple 1D arrays of length 512.
    # The kernel fills them using the hierarchical (lane, warp, iter) mapping.
    print(f"Threads per CTA: {N_THR}")
    print(f"Elements per thread: {ELEMS_PER_THR}")
    print("Launching one CTA over a 512-element block.")

    warp_handling_kernel(mWarp, mLane, mIter).launch(
        grid=[1, 1, 1],
        block=[N_THR, 1, 1],
    )


def run_warp_handling_demo():
    print("\n== Part 2: Real Warp/Lane Ownership Demo ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    warp_out = torch.full((N_BLK,), -1, device="cuda", dtype=torch.int32)
    lane_out = torch.full((N_BLK,), -1, device="cuda", dtype=torch.int32)
    iter_out = torch.full((N_BLK,), -1, device="cuda", dtype=torch.int32)

    mWarp = from_dlpack(warp_out, assumed_align=16)
    mLane = from_dlpack(lane_out, assumed_align=16)
    mIter = from_dlpack(iter_out, assumed_align=16)

    print("Compiling kernel with cute.compile ...")
    compile_start = time.time()
    kernel = cute.compile(run_warp_handling_kernel, mWarp, mLane, mIter)
    print(f"Compile finished in {time.time() - compile_start:.2f}s")

    print("Launching kernel ...")
    launch_start = time.time()
    kernel(mWarp, mLane, mIter)
    torch.cuda.synchronize()
    print(f"Kernel run finished in {time.time() - launch_start:.2f}s")
    print("Preparing readable output and verification ...")

    # Build the same expected ownership on the CPU so we can verify it.
    expected_warp = torch.empty_like(warp_out)
    expected_lane = torch.empty_like(lane_out)
    expected_iter = torch.empty_like(iter_out)
    for warp in range(N_WARPS):
        for lane in range(WARP_SIZE):
            for i in range(ELEMS_PER_THR):
                offset = lane + warp * WARP_SIZE + i * N_THR
                expected_warp[offset] = warp
                expected_lane[offset] = lane
                expected_iter[offset] = i

    warp_cpu = warp_out.cpu()
    lane_cpu = lane_out.cpu()
    iter_cpu = iter_out.cpu()
    expected_warp_cpu = expected_warp.cpu()
    expected_lane_cpu = expected_lane.cpu()
    expected_iter_cpu = expected_iter.cpu()

    print("Warp ownership, iter 0 in warp-sized chunks:")
    print("  offsets [0:32]:", warp_cpu[0:32].tolist())
    print("  offsets [32:64]:", warp_cpu[32:64].tolist())
    print("  offsets [64:96]:", warp_cpu[64:96].tolist())
    print("  offsets [96:128]:", warp_cpu[96:128].tolist())
    print("")

    print("Lane ownership inside warp 2, iter 0 chunk [64:96]:")
    print(lane_cpu[64:96].tolist())
    print("")

    print("One lane across all iterations:")
    print(
        "  warp 2, lane 5 offsets:",
        [
            5 + 2 * WARP_SIZE + i * N_THR
            for i in range(ELEMS_PER_THR)
        ],
    )
    print(
        "  recorded iters:",
        [
            int(iter_cpu[5 + 2 * WARP_SIZE + i * N_THR].item())
            for i in range(ELEMS_PER_THR)
        ],
    )
    print("")

    print("matches expected warp ids:", torch.equal(warp_cpu, expected_warp_cpu))
    print("matches expected lane ids:", torch.equal(lane_cpu, expected_lane_cpu))
    print("matches expected iter ids:", torch.equal(iter_cpu, expected_iter_cpu))


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_warp_handling_theory()
            return
        if mode == "--kernel":
            run_warp_handling_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)