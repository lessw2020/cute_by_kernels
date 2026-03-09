import os
import subprocess
import sys

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack


# Part 1 uses tiny conceptual shapes so the layout story is easy to read.
M = 16
N = 16
K = 8
CTA_M = 8
CTA_N = 8
CTA_K = 4


# Part 2 switches to one exact tile of the real Blackwell tutorial GEMM.
io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
ab_stages = 4


@cute.jit
def show_real_gemm_symbols_theory_jit():
    block_row_idx = 1
    block_col_idx = 0

    A = cute.make_layout((M, K), stride=(K, 1))
    B = cute.make_layout((N, K), stride=(K, 1))
    C = cute.make_layout((M, N), stride=(N, 1))

    tiled_A = cute.zipped_divide(A, (CTA_M, CTA_K))
    tiled_B = cute.zipped_divide(B, (CTA_N, CTA_K))
    tiled_C = cute.zipped_divide(C, (CTA_M, CTA_N))

    print(f"A layout (M,K): {A}")
    print(f"B layout (N,K): {B}")
    print(f"C layout (M,N): {C}")
    print(f"CTA tiler (bM, bN, bK): {(CTA_M, CTA_N, CTA_K)}")
    print("")

    # These are the conceptual versions of the tutorial's names:
    #
    #   mA / mB / mC : whole tensors
    #   gA / gB / gC : CTA-local tiles cut out of those tensors
    #
    # In the real tutorial this extraction is written with `local_tile(...)`.
    print(f"Conceptual gA tiling: {tiled_A}")
    print(f"Conceptual gB tiling: {tiled_B}")
    print(f"Conceptual gC tiling: {tiled_C}")
    print("")

    gA, _ = cute.slice_and_offset(((None, None), (block_row_idx, 0)), tiled_A)
    gB, _ = cute.slice_and_offset(((None, None), (block_col_idx, 0)), tiled_B)
    gC, _ = cute.slice_and_offset(((None, None), (block_row_idx, block_col_idx)), tiled_C)

    print(f"Conceptual gA tile: {gA}")
    print(f"Conceptual gB tile: {gB}")
    print(f"Conceptual gC tile: {gC}")
    print("")

    # The next symbol step in the real GEMM is:
    #
    #   tCgA = thr_mma.partition_A(gA)
    #   tCgB = thr_mma.partition_B(gB)
    #   tCgC = thr_mma.partition_C(gC)
    #
    # Read those names as:
    #   - gA/gB/gC   : CTA tiles
    #   - tCgA/B/C   : the same CTA tiles, but repartitioned to match the MMA
    #                  engine's view of the work
    #
    # Then:
    #
    #   tCrA = tiled_mma.make_fragment_A(sA)
    #   tCrB = tiled_mma.make_fragment_B(sB)
    #   tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    #
    # Read those as:
    #   - sA/sB      : staged SMEM tiles
    #   - tCrA/tCrB  : register fragments that will feed MMA
    #   - tCtAcc     : accumulator fragment that MMA updates
    print("Symbol progression in the real GEMM:")
    print("  mA/mB/mC   : whole tensors")
    print("  gA/gB/gC   : CTA tiles")
    print("  sA/sB      : staged SMEM tiles")
    print("  tCgA/B/C   : MMA-partitioned CTA views")
    print("  tCrA/tCrB  : register fragments for A/B")
    print("  tCtAcc     : accumulator fragment for C")
    print("")

    print("What `partition_A/B/C` means:")
    print("  it does not change the math")
    print("  it changes the coordinate system so the tile matches the MMA engine's expected view")
    print("")

    print("What `make_fragment_A/B/C` means:")
    print("  it creates the register-side fragment shapes that MMA consumes or updates")
    print("  it is the bridge from staged SMEM layout to register operands")


def show_real_gemm_symbols_theory():
    print("== Part 1: Real GEMM Symbol Theory ==")
    show_real_gemm_symbols_theory_jit()


@cute.kernel
def inspect_real_gemm_symbols_kernel(
    tiled_mma: cute.TiledMma,
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    a_smem_layout_one_stage: cute.ComposedLayout,
    b_smem_layout_one_stage: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(
            element_type=gA.element_type,
            layout=a_smem_layout_one_stage.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_one_stage.inner,
        )
        sB = smem.allocate_tensor(
            element_type=gB.element_type,
            layout=b_smem_layout_one_stage.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_one_stage.inner,
        )

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        print("MMA-partitioned CTA views:")
        print(f"  tCgA shape: {tCgA.shape}")
        print(f"  tCgB shape: {tCgB.shape}")
        print(f"  tCgC shape: {tCgC.shape}")
        print("")

        print("Register fragment shapes:")
        print(f"  tCrA shape: {tCrA.shape}")
        print(f"  tCrB shape: {tCrB.shape}")
        print(f"  tCtAcc shape: {tCtAcc.shape}")
        print("")

        print("How to read those symbols:")
        print("  gA/gB/gC   : CTA tiles cut from the whole tensors")
        print("  tCgA/B/C   : the same tiles, repartitioned into MMA-friendly coordinates")
        print("  tCrA/tCrB  : the register fragments that hold A/B operands for MMA")
        print("  tCtAcc     : the accumulator fragment updated by MMA")


@cute.jit
def inspect_real_gemm_symbols(
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
):
    # This is the exact MMA object construction from the Blackwell tutorial.
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

    # Build the SMEM layouts exactly the way the tutorial does.
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        mA_mk.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        mB_nk.element_type,
        ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    # The tutorial uses `local_tile(...)` rather than `zipped_divide(...)`.
    # Because these tensors are exactly one CTA tile large, the CTA coordinate is
    # just `(0, 0, None)`.
    mma_coord_mnk = (0, 0, None)
    gA = cute.local_tile(mA_mk, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nk, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    print("Exact Blackwell tutorial objects:")
    print(f"tiled_mma: {tiled_mma}")
    print("")

    print("Whole tensors:")
    print(f"  mA shape/stride: {mA_mk.shape} / {mA_mk.stride}")
    print(f"  mB shape/stride: {mB_nk.shape} / {mB_nk.stride}")
    print(f"  mC shape/stride: {mC_mn.shape} / {mC_mn.stride}")
    print("")

    print("CTA-local views from `local_tile(...)`:")
    print(f"  gA: {gA.type}")
    print(f"  gB: {gB.type}")
    print(f"  gC: {gC.type}")
    print("")

    print("One-stage SMEM layouts:")
    print(f"  sA layout: {a_smem_layout_one_stage}")
    print(f"  sB layout: {b_smem_layout_one_stage}")
    print("")

    inspect_real_gemm_symbols_kernel(
        tiled_mma,
        gA,
        gB,
        gC,
        a_smem_layout_one_stage,
        b_smem_layout_one_stage,
    ).launch(
        grid=[1, 1, 1],
        block=[1, 1, 1],
    )


def run_real_gemm_symbol_demo():
    print("\n== Part 2: Real Blackwell GEMM Symbol Walk ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the real tensor demo is skipped.")
        return

    # These shapes are exactly one CTA tile from the tutorial:
    #   A : (128, 64)
    #   B : (256, 64)
    #   C : (128, 256)
    #
    # We only inspect the layout objects, so the actual values are unimportant.
    a = torch.zeros(mma_tiler_mnk[0], mma_tiler_mnk[2], device="cuda", dtype=torch.float16)
    b = torch.zeros(mma_tiler_mnk[1], mma_tiler_mnk[2], device="cuda", dtype=torch.float16)
    c = torch.zeros(mma_tiler_mnk[0], mma_tiler_mnk[1], device="cuda", dtype=torch.float16)

    # The tutorial uses stronger 32-byte alignment assumptions for these operands.
    mA = from_dlpack(a, assumed_align=32)
    mB = from_dlpack(b, assumed_align=32)
    mC = from_dlpack(c, assumed_align=32)

    inspect = cute.compile(inspect_real_gemm_symbols, mA, mB, mC)
    inspect(mA, mB, mC)
    torch.cuda.synchronize()


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_real_gemm_symbols_theory()
            return
        if mode == "--inspect":
            run_real_gemm_symbol_demo()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--inspect"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
