import importlib.util
import os
import subprocess
import sys

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack


REAL_GEMM_PATH = (
    "/home/less/cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py"
)


def _load_real_gemm_module():
    spec = importlib.util.spec_from_file_location("real_blackwell_gemm", REAL_GEMM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


real_gemm = _load_real_gemm_module()

# Reuse the exact Blackwell tutorial constants so this lesson stays anchored to
# the real kernel rather than a simplified stand-in.
io_dtype = real_gemm.io_dtype
acc_dtype = real_gemm.acc_dtype
mma_inst_shape_mnk = real_gemm.mma_inst_shape_mnk
mma_tiler_mnk = real_gemm.mma_tiler_mnk
threads_per_cta = real_gemm.threads_per_cta
ab_stages = real_gemm.ab_stages

TILE_M, TILE_N, TILE_K = mma_tiler_mnk


@cute.jit
def show_blackwell_first_gemm_theory_jit():
    print("This lesson stays on the real Blackwell tutorial kernel.")
    print("")
    print("The exact host-side setup in `fp16_gemm_0.py` is:")
    print("  1. build `tiled_mma`")
    print("  2. build staged SMEM layouts for A and B")
    print("  3. build TMA atoms that know how to move GMEM tiles into those SMEM layouts")
    print("  4. launch the kernel")
    print("")
    print("Inside the real kernel, the symbol flow is:")
    print("  mA/mB/mC      : whole tensors")
    print("  gA/gB/gC      : CTA tiles cut out with `local_tile(...)`")
    print("  tCgA/tCgB/tCgC: those same CTA tiles, repartitioned to match MMA coordinates")
    print("  sA/sB         : staged shared-memory tiles")
    print("  tAsA/tBsB     : TMA destinations in staged SMEM")
    print("  tAgA/tBgB     : matching TMA sources in GMEM")
    print("  tCrA/tCrB     : fragment views that MMA reads from staged SMEM")
    print("  tCtAcc        : accumulator fragment that MMA updates")
    print("")
    print("The key Blackwell point is:")
    print("  `tCrA` and `tCrB` are not stand-alone arrays you fill by hand")
    print("  they are fragment views over the staged SMEM tiles")
    print("  TMA fills `sA` and `sB`, then `cute.gemm(...)` consumes slices of `tCrA/tCrB`")
    print("")
    print("The real GEMM loop in the tutorial is:")
    print("  ab_full = ab_consumer.wait_and_advance()")
    print("  for k_block_idx in range_constexpr(num_k_blocks):")
    print("      k_block_coord = (None, None, k_block_idx, ab_full.index)")
    print("      cute.gemm(tiled_mma, tCtAcc, tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)")
    print("")
    print("Read `k_block_coord` as:")
    print("  - keep the MMA and MMA-subtile modes intact with `None`")
    print("  - pick one K-block inside the fragment")
    print("  - pick which SMEM pipeline stage is currently ready")
    print("")
    print("So the compute flow is not:")
    print("  GMEM -> registers -> MMA")
    print("")
    print("It is:")
    print("  GMEM --TMA--> staged SMEM --fragment view--> `cute.gemm(...)` -> TMEM accumulator")


def show_blackwell_first_gemm_theory():
    print("== Part 1: Blackwell First GEMM Theory ==")
    show_blackwell_first_gemm_theory_jit()


@cute.kernel
def inspect_blackwell_first_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    a_tma_tensor: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    b_tma_tensor: cute.Tensor,
    mC_mn: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    if tidx == 0:
        mma_coord_mnk = (bidx, bidy, None)

        # These lines mirror the real tutorial kernel exactly: first cut out the
        # CTA tile each block owns, then repartition that tile for MMA.
        #
        # `proj` tells `local_tile` which GEMM axes this tensor actually uses:
        #   A uses (M, K), so ignore N      -> proj=(1, None, 1)
        #   B uses (N, K), so ignore M      -> proj=(None, 1, 1)
        #   C uses (M, N), so ignore K      -> proj=(1, 1, None)
        gA = cute.local_tile(a_tma_tensor, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(b_tma_tensor, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

        smem = utils.SmemAllocator()
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

        # `thr_mma` is one thread's view of the tiled MMA object.
        # We use slice 0 here just to inspect the symbol shapes in a stable way.
        thr_mma = tiled_mma.get_slice(0)

        # `partition_A/B/C` does not move data. It re-expresses the CTA tiles
        # `gA/gB/gC` in the coordinate system that the MMA engine expects.
        # So `tCg*` still refer to the CTA's GMEM tiles, but now with
        # MMA-friendly indexing.
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        # `make_fragment_A/B` builds the operand fragment views that MMA will
        # read from staged SMEM. These are the "register-side" shapes that the
        # later `cute.gemm(...)` loop indexes as `(None, None, k_block, stage)`.
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # The accumulator fragment is shaped like one CTA tile of C, but in the
        # MMA engine's preferred fragment coordinates rather than plain `(m, n)`.
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        # This is the exact TMA retile step from the tutorial:
        # it couples the staged SMEM layout with the MMA-partitioned GMEM view.
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        print("Exact real-kernel objects for one CTA:")
        print(f"  gA type: {gA.type}")
        print(f"  gB type: {gB.type}")
        print(f"  gC type: {gC.type}")
        print("")

        print("MMA-partitioned GMEM views:")
        print(f"  tCgA shape: {tCgA.shape}")
        print(f"  tCgB shape: {tCgB.shape}")
        print(f"  tCgC shape: {tCgC.shape}")
        print("")

        print("Staged SMEM layouts:")
        print(f"  sA type: {sA.type}")
        print(f"  sB type: {sB.type}")
        print("")

        print("TMA partition views:")
        print(f"  tAsA shape: {tAsA.shape}")
        print(f"  tAgA shape: {tAgA.shape}")
        print(f"  tBsB shape: {tBsB.shape}")
        print(f"  tBgB shape: {tBgB.shape}")
        print("")

        print("MMA fragment views:")
        print(f"  tCrA shape: {tCrA.shape}")
        print(f"  tCrB shape: {tCrB.shape}")
        print(f"  tCtAcc shape: {tCtAcc.shape}")
        print("")

        print("How the real mainloop reads these:")
        print("  num_k_tiles in gA along global K: 1 in this demo (K is exactly one CTA tile)")
        print(f"  num_k_blocks inside one staged A fragment: {cute.size(tCrA, mode=[2])}")
        print("  real coord form: (None, None, k_block_idx, stage_idx)")
        print("")

        print("Important distinction:")
        print("  `tCgA` is a GMEM tile with MMA-friendly coordinates")
        print("  `tCrA` is an SMEM-backed fragment view that `cute.gemm(...)` consumes")
        print("  the tutorial later swaps `tCtAcc` onto TMEM before the mainloop")


@cute.jit
def inspect_blackwell_first_gemm(
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
):
    # This host setup is copied from the real tutorial host function.
    op = real_gemm.tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        real_gemm.tcgen05.CtaGroup.ONE,
        real_gemm.tcgen05.OperandSource.SMEM,
        real_gemm.tcgen05.OperandMajorMode.K,
        real_gemm.tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem_layout = real_gemm.sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        mA_mk.element_type,
        ab_stages,
    )
    b_smem_layout = real_gemm.sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        mB_nk.element_type,
        ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(real_gemm.tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        mA_mk,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        mB_nk,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )

    inspect_blackwell_first_gemm_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        mC_mn,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=[1, 1, 1],
        block=[1, 1, 1],
    )


def make_demo_tensors():
    # Use one exact Blackwell CTA tile so every printed object matches the real
    # tutorial tile sizes:
    #   A : (128, 64)
    #   B : (256, 64)
    #   C : (128, 256)
    #
    # Choose easy values:
    #   A is all ones
    #   B[n, k] is constant across K for each output-column n
    #
    # Then:
    #   C[m, n] = sum_k 1 * B[n, k] = 64 * B[n, 0]
    #
    # so the first row is simple to inspect.
    a = torch.ones((TILE_M, TILE_K), device="cuda", dtype=torch.float16)
    b_col_values = ((torch.arange(TILE_N, device="cuda", dtype=torch.int32) % 8) + 1).to(
        torch.float16
    )
    b = b_col_values[:, None].repeat(1, TILE_K).contiguous()
    c = torch.zeros((TILE_M, TILE_N), device="cuda", dtype=torch.float16)

    mA = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=TILE_K)
    )
    mB = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=TILE_K)
    )
    mC = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=TILE_N)
    )
    return a, b, c, mA, mB, mC


def run_blackwell_first_gemm_inspect():
    print("\n== Part 2: Inspect Real Blackwell Kernel Objects ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the inspection step is skipped.")
        return

    _, _, _, mA, mB, mC = make_demo_tensors()
    inspect = cute.compile(inspect_blackwell_first_gemm, mA, mB, mC)
    inspect(mA, mB, mC)
    torch.cuda.synchronize()


def run_blackwell_first_gemm_kernel():
    print("\n== Part 3: Run Real Blackwell GEMM Kernel ==")

    if not torch.cuda.is_available():
        print("CUDA is not available, so the kernel demo is skipped.")
        return

    a, b, c, mA, mB, mC = make_demo_tensors()

    # This is the exact tutorial host entry point.
    real_gemm.host_function(mA, mB, mC, no_cache=True)
    torch.cuda.synchronize()

    expected = torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32)).to(
        torch.float16
    )
    print("A first row, first 8:", a[0, :8].tolist())
    print("B first 8 rows, k=0:", b[:8, 0].tolist())
    print("")
    print("Expected C first row, first 8:", expected[0, :8].tolist())
    print("Output C first row, first 8:", c[0, :8].tolist())
    print("")
    print("matches torch:", torch.equal(c, expected))


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--theory":
            show_blackwell_first_gemm_theory()
            return
        if mode == "--inspect":
            run_blackwell_first_gemm_inspect()
            return
        if mode == "--kernel":
            run_blackwell_first_gemm_kernel()
            return
        raise ValueError(f"Unknown mode: {mode}")

    subprocess.run([sys.executable, __file__, "--theory"], check=True)
    subprocess.run([sys.executable, __file__, "--inspect"], check=True)
    subprocess.run([sys.executable, __file__, "--kernel"], check=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
