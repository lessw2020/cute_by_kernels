import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
ab_stages = 4


@cute.kernel
def kernel(done: cute.Tensor, a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()

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

    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(io_dtype, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
    sB = smem.allocate_tensor(io_dtype, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tCrA.fill(io_dtype(1.0))
    tCrB.fill(io_dtype(1.0))
    tCtAcc.fill(acc_dtype(0.0))

    for k_block_idx in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        coord = (None, None, k_block_idx)
        cute.gemm(
            tiled_mma,
            tCtAcc,
            tCrA[coord],
            tCrB[coord],
            tCtAcc,
        )
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    if tidx == 0:
        done[0] = 1


@cute.jit
def launch(done: cute.Tensor):
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
    a_layout = cute.select(sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages), mode=[0,1,2])
    b_layout = cute.select(sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages), mode=[0,1,2])
    kernel(done, a_layout, b_layout).launch(grid=[1,1,1], block=[128,1,1])
