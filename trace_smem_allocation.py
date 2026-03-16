import os
import sys

if "CUTE_DSL_ARCH" not in os.environ:
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        if (major, minor) == (10, 3):
            # This machine is a Blackwell B300-class system, and the tcgen05 MMA
            # ops used below expect the accelerator-class suffix.
            os.environ["CUTE_DSL_ARCH"] = "sm_103a"
        else:
            os.environ["CUTE_DSL_ARCH"] = f"sm_{major}{minor}"
    except Exception:
        # Fall back to the Blackwell architecture available on this machine.
        os.environ["CUTE_DSL_ARCH"] = "sm_103a"

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

"""
Minimal CuTeDSL example that traces Blackwell GEMM shared-memory allocation.

The kernel:
1. Constructs the same one-stage A/B shared-memory layouts used by ``fp16_gemm_0.py``.
2. Allocates swizzled SMEM buffers with ``SmemAllocator``.
3. Mirrors one CTA tile of A and B into plain SMEM for tracing.
4. Prints the SMEM layouts from inside the kernel.
5. Copies just the first row through a plain SMEM mirror for host-side verification.
"""

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 1


@cute.kernel
def trace_smem_kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    trace_a_row: cute.Tensor,
    trace_b_row: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # Shared-memory allocation mirrors one stage of the A/B buffers in the GEMM tutorial.
    # CuTeDSL does not currently support simple direct tracing from the swizzled
    # SMEM buffers, so this example also allocates plain row-major mirror buffers
    # that are easier to inspect element-by-element.
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # These mirrors use ordinary row-major layouts so we can trace a small sample
    # of values without fighting the swizzled address mapping.
    a_trace_layout = cute.make_layout((mma_tiler_mnk[0], mma_tiler_mnk[2]), stride=(mma_tiler_mnk[2], 1))
    b_trace_layout = cute.make_layout((mma_tiler_mnk[1], mma_tiler_mnk[2]), stride=(mma_tiler_mnk[2], 1))
    sA_trace = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_trace_layout,
        byte_alignment=16,
    )
    sB_trace = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_trace_layout,
        byte_alignment=16,
    )

    # Select the CTA tile owned by this block and the first K tile.
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gA_tile = cute.slice_(gA, (None, None, 0))
    gB_tile = cute.slice_(gB, (None, None, 0))

    if tidx == 0:
        cute.printf("=== trace_smem kernel ===")
        cute.printf("A global tile layout      = {}", gA_tile.layout)
        cute.printf("B global tile layout      = {}", gB_tile.layout)
        cute.printf("A one-stage SMEM layout   = {}", sA.layout)
        cute.printf("B one-stage SMEM layout   = {}", sB.layout)
        cute.printf("A trace SMEM layout       = {}", sA_trace.layout)
        cute.printf("B trace SMEM layout       = {}", sB_trace.layout)
        cute.printf("Read one-stage SMEM as: the same logical tile, but with K factored as 16 x 4")
        cute.printf("For A: ((128,16),1,4):((64,1),0,16) is roughly offset = m*64 + k_inner + k_outer*16")

        # Copy just the first row through the plain SMEM mirrors. This still
        # demonstrates the allocation and staging story, but keeps the trace
        # small enough that the kernel compiles reliably.
        for col in cutlass.range_constexpr(mma_tiler_mnk[2]):
            sA_trace[(0, col)] = gA_tile[(0, col)]
            sB_trace[(0, col)] = gB_tile[(0, col)]
            trace_a_row[col] = sA_trace[(0, col)]
            trace_b_row[col] = sB_trace[(0, col)]


@cute.jit
def host_function(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    trace_a_row: cute.Tensor,
    trace_b_row: cute.Tensor,
):
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        mA_mkl.element_type,
        4,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        mB_nkl.element_type,
        4,
    )
    a_smem_layout = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout = cute.select(b_smem_layout, mode=[0, 1, 2])

    trace_smem_kernel(
        mA_mkl,
        mB_nkl,
        trace_a_row,
        trace_b_row,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=(1, 1, 1),
        block=(threads_per_cta, 1, 1),
    )


def run_trace_smem():
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch

    print("==============================================================")
    print("Tracing Blackwell GEMM shared-memory allocation in CuTeDSL")
    print(f"  A tile shape: ({mma_tiler_mnk[0]}, {mma_tiler_mnk[2]})")
    print(f"  B tile shape: ({mma_tiler_mnk[1]}, {mma_tiler_mnk[2]})")
    print("==============================================================")
    print()

    a = torch.arange(
        mma_tiler_mnk[0] * mma_tiler_mnk[2],
        dtype=cutlass_torch.dtype(io_dtype),
        device="cuda",
    ).reshape((mma_tiler_mnk[0], mma_tiler_mnk[2]))
    b = torch.arange(
        mma_tiler_mnk[1] * mma_tiler_mnk[2],
        dtype=cutlass_torch.dtype(io_dtype),
        device="cuda",
    ).reshape((mma_tiler_mnk[1], mma_tiler_mnk[2]))
    trace_a_row = torch.empty((mma_tiler_mnk[2],), dtype=cutlass_torch.dtype(io_dtype), device="cuda")
    trace_b_row = torch.empty((mma_tiler_mnk[2],), dtype=cutlass_torch.dtype(io_dtype), device="cuda")

    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    trace_a_row_tensor = from_dlpack(trace_a_row, assumed_align=16)
    trace_b_row_tensor = from_dlpack(trace_b_row, assumed_align=16)

    print("Host input A first row:")
    print(a[0].cpu())
    print()
    print("Host input B first row:")
    print(b[0].cpu())
    print()
    print("Launching kernel...")
    host_function(a_tensor, b_tensor, trace_a_row_tensor, trace_b_row_tensor, no_cache=True)

    torch.testing.assert_close(trace_a_row.cpu(), a[0].cpu(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(trace_b_row.cpu(), b[0].cpu(), atol=0.0, rtol=0.0)

    print()
    print("Host traced A first row via plain SMEM mirror:")
    print(trace_a_row.cpu())
    print()
    print("Host traced B first row via plain SMEM mirror:")
    print(trace_b_row.cpu())


if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()
    run_trace_smem()
    print("PASS")
    sys.stdout.flush()
    os._exit(0)