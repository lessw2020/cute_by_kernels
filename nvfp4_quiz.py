"""
CuTe DSL: NVFP4 Block-Scaled GEMM — Layouts, S2T Scale Copy, TMEM Placement
=============================================================================
Practice questions covering:
  - blockscaled_utils.tile_atom_to_shape_SF and make_smem_layout_sfa/sfb
  - NVFP4 (Float4E2M1FN) quantization granularity: sf_vec_size=16
  - filter_zeros on TMA-partitioned SFA/SFB (unlike blockwise GEMM which filters AFTER partition)
  - SMEM-to-TMEM (S2T) copy with tcgen05.Cp4x32x128bOp
  - tCtSFA / tCtSFB TMEM tensor placement after the accumulator
  - tcgen05.find_tmem_tensor_col_offset for sequential TMEM allocation
  - tiled_mma.set(Field.SFA / Field.SFB) per kblock
  - Simplified single-warp-0 design vs warp-specialized blockwise kernel
  - Ld32x32bOp(Repetition.x128) for the larger 128x256 tile epilogue
  - cute.assume() for alignment hints on tensor shapes/strides

Run modes:
  python cute_nvfp4_layout_qa.py          # full Q+A
  python cute_nvfp4_layout_qa.py --quiz   # questions + hints only
"""

import sys

QUIZ_MODE = "--quiz" in sys.argv

def section(title):
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def qa(n, question, hint, answer):
    print(f"\nQ{n:02d}.  {question}")
    print(f"  HINT: {hint}")
    if not QUIZ_MODE:
        print(f"  ANSWER:\n{answer}")
    print()

# ---------------------------------------------------------------------------
section("PART 1 — NVFP4 Quantization Format and Scale Layout Construction")
# ---------------------------------------------------------------------------

qa(1,
"""NVFP4 uses sf_vec_size=16, meaning one Float8E4M3FN scale covers 16
   consecutive Float4E2M1FN elements.  The scale layout is built with:
     sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
         a_tensor.shape, sf_vec_size
     )
   where a_tensor.shape = (M, K, L) with K-major storage.

   (a) Conceptually, what shape should sfa_layout have in terms of
       M, K, L, and sf_vec_size=16?
   (b) How does NVFP4's sf_vec_size=16 differ from MXFP8's granularity
       of 128 in terms of what the inner layout modes represent?
   (c) Given M=8192, K=8192, L=1, how many total scale values does
       sfa_layout represent?""",
hint="One scale per 16 K-elements.  The shape should look like ((Atom_M, M/Atom_M), (Atom_K, K/Atom_K), L).",
answer="""\
  (a) tile_atom_to_shape_SF fills the SF tensor to match the MMA atom's
      expected scale layout.  For NVFP4 with sf_vec_size=16:
        Along K: one scale per 16 FP4 elements -> K/16 scale groups
        Along M: the MMA atom groups M elements together; for a
                 128×256×64 MMA, the atom covers 128 M rows with one
                 scale per 16 K elements.

      Conceptually: sfa_layout.shape ≈ ((Atom_M, M/Atom_M), (16, K/16), L)
      where Atom_M is the M-extent of one MMA scale atom (typically 1 for
      NVFP4 since scale is per-K-group, not per-block), and the inner
      Atom_K=16 mode is the broadcast granularity (stride 0).

  (b) Both NVFP4 and MXFP8 are per-row scaling formats — each M-row
      has its own independent set of scale factors.  The only difference
      is the K-group size:

        NVFP4  (sf_vec_size=16): one FP8 scale per row per 16 K-elements.
          Layout: stride-0 on the K-inner mode (size 16) only.
          M-mode has normal non-zero strides — every row is independent.

        MXFP8  (sf_vec_size=32): one FP8 scale per row per 32 K-elements.
          Layout: stride-0 on the K-inner mode (size 32) only.
          Same per-row structure, just coarser along K.

      Visually (small example, M=4, K=64):

        NVFP4 scale shape (4, 4) — 4 groups of 16 K-elements per row:
          K:    0..15  16..31  32..47  48..63
          M=0 [  s0,0    s0,1    s0,2    s0,3 ]
          M=1 [  s1,0    s1,1    s1,2    s1,3 ]
          M=2 [  s2,0    s2,1    s2,2    s2,3 ]
          M=3 [  s3,0    s3,1    s3,2    s3,3 ]

        MXFP8 scale shape (4, 2) — 2 groups of 32 K-elements per row:
          K:    0..31  32..63
          M=0 [  s0,0    s0,1 ]
          M=1 [  s1,0    s1,1 ]
          ...

      The blockwise_gemm.py example (scale_granularity_m=128,
      scale_granularity_k=128) is DeepSeek-style coarse block scaling —
      NOT MXFP8.  It has stride-0 on BOTH the M-inner (size 128) and
      K-inner (size 128) modes, meaning one scalar covers a 128×128 block.

  (c) Total scale values = M * (K / sf_vec_size) * L
                         = 8192 * (8192 / 16) * 1
                         = 8192 * 512
                         = 4,194,304 Float8E4M3FN scale values for A.
      Each is a single FP8 scalar covering 16 consecutive K-elements
      for exactly one M-row (not shared across rows).
""")

qa(2,
"""The SMEM scale layout is built with:
     self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
         tiled_mma, self.mma_tiler, sf_vec_size, self.num_ab_stage
     )

   Unlike sA/sB which use ComposedLayout (outer + swizzle inner),
   sSFA is allocated as:
     sSFA = smem.allocate_tensor(
         element_type=sf_dtype,
         layout=sfa_smem_layout_staged,
         byte_alignment=128,
     )  # no swizzle= argument

   (a) Why does sSFA not need a swizzle?
   (b) The sSFA layout has 4 modes like sA/sB: (MMA, MMA_M, MMA_K, STAGE).
       For mma_tiler=(128,256,256) and sf_vec_size=16, describe roughly
       what MMA_K means for the scale vs for the data tensor sA.
   (c) After TMA loads sSFA, the kernel calls:
         tCsSFA_compact = cute.filter_zeros(sSFA)
       What is being filtered, and what is the shape of tCsSFA_compact
       relative to sSFA?""",
hint="Scales don't feed the MMA SMEM read path; they go to TMEM via S2T.  filter_zeros removes broadcast modes.",
answer="""\
  (a) SMEM swizzle is needed to avoid bank conflicts when the MMA engine
      reads operand data (A/B) directly from SMEM in the tcgen05.mma
      instruction.  The MMA instruction accesses sA and sB in a highly
      parallel pattern where multiple warps read the same column, causing
      bank conflicts without swizzling.

      Scale factors (sSFA) are NOT read by the MMA instruction directly
      from SMEM.  Instead, they are copied to TMEM via the S2T path
      (Cp4x32x128bOp) before MMA runs.  The S2T copy is a single warp-0
      operation that accesses SMEM sequentially; it does not suffer from
      the parallel-read bank conflicts that MMA causes.  Therefore no
      swizzle is needed.

  (b) For sA (data), MMA_K reflects how many K-slices the MMA atom
      processes per invocation — e.g., with mma_inst_shape_k=64 and
      mma_tiler_k=256, MMA_K=256/64=4 atoms along K.

      For sSFA (scale), MMA_K reflects how many scale groups there are
      along K within one CTA tile.  With sf_vec_size=16 and tile_k=256:
        MMA_K (scale) = 256 / 16 = 16 K-groups per tile.
      Each MMA K-block (covering 64 FP4 elements) contains 64/16=4
      scale groups.  The scale MMA_K is therefore coarser than the
      data MMA_K when viewed in terms of MMA invocations, but finer
      than MXFP8's single scale per tile.

  (c) sSFA was built from tile_atom_to_shape_SF which encodes the
      broadcast (stride-0) modes for the 16-element K-granularity.
      filter_zeros strips those broadcast modes, leaving only the
      physically-distinct scale elements.

      If sSFA.shape = (MMA, (16, MMA_K_blocks), MMA_M, STAGE):
        - The inner mode 16 has stride 0 (broadcast over 16 K-elements)
        - filter_zeros removes it
      tCsSFA_compact.shape ≈ (MMA, MMA_K_blocks, MMA_M, STAGE)
      which is smaller by a factor of sf_vec_size=16 along K.
""")

qa(3,
"""The TMA partition for SFA is followed immediately by filter_zeros:
     tAsSFA, tAgSFA = cpasync.tma_partition(
         tma_atom_sfa, 0, cute.make_layout(1),
         cute.group_modes(sSFA, 0, 3),
         cute.group_modes(tCgSFA, 0, 3),
     )
     tAsSFA = cute.filter_zeros(tAsSFA)
     tAgSFA = cute.filter_zeros(tAgSFA)

   Compare this to blockwise_gemm.py where filter_zeros is only called
   INSIDE the scale-load loop on the per-k-tile slice.

   (a) Why can this kernel apply filter_zeros to the full tAgSFA at
       construction time, whereas blockwise_gemm cannot?
   (b) The TMA atom for SFA uses internal_type=cutlass.Int16.  Why?
       The actual data type is Float8E4M3FN (8 bits); why Int16?
   (c) After filter_zeros, how does tAgSFA's shape relate to tAgA?
       (Both use tma_partition with the same layout(1) warpgroup.)""",
hint="blockwise uses runtime tile coords in the filter; this kernel has static shapes.  Int16 = packed pair of FP8.",
answer="""\
  (a) blockwise_gemm uses a *persistent* scheduler where tile coordinates
      (m, n, l) are only known at runtime inside the warp loop.  The
      filter_zeros slice must be applied after selecting the per-tile
      coords, so it happens inside the loop.

      This NVFP4 kernel uses a simple grid-based launch: each CTA handles
      exactly one output tile at a fixed (bidx, bidy, bidz) coordinate.
      The tile coord is known at kernel-launch time and baked into the
      TMA descriptor.  The tma_partition result tAgSFA already has the
      full (atom_v, RestK, RestL) shape with no runtime-dependent axes
      that would change the filter result.  Applying filter_zeros once
      at construction time is both correct and more efficient.

  (b) NVFP4 scale factors (Float8E4M3FN, 8-bit) are stored in pairs in
      global memory: two FP8 scales are packed into one 16-bit word
      because NVFP4 data itself is 4-bit, and the scale / data packing
      follows the hardware's Int16 granularity expectation for TMA bulk
      transfers.  TMA bulk transfers operate on 16-byte minimum quanta;
      using internal_type=Int16 tells TMA how to interpret the packed
      FP8 pairs during the GMEM→SMEM transfer.  Without this hint, TMA
      might use FP8 addressing that misaligns the packed scale words.

  (c) After filter_zeros, tAgSFA has the broadcast K-modes removed.
      tAgA shape (for data):   ((atom_v, rest_v), RestK, RestL)
        where RestK = mma_tiler_k / mma_inst_shape_k = 256/64 = 4 K-tiles.
      tAgSFA shape (for scale): ((atom_v, rest_v), RestK_scale, RestL)
        where RestK_scale = RestK * (mma_inst_shape_k / sf_vec_size)
                          = 4 * (64/16) = 4 * 4 = 16 scale K-groups.

      So tAgSFA has 4× more entries along K than tAgA — one scale
      entry per 16 FP4 elements rather than one TMA load per 64-element
      MMA K-block.  Both are indexed with the same .count / .index
      pattern in the k-tile loop, but the scale's RestK axis is larger.
""")

# ---------------------------------------------------------------------------
section("PART 2 — TMEM Layout for Scale Factors and Sequential Allocation")
# ---------------------------------------------------------------------------

qa(4,
"""After allocating the accumulator in TMEM, the kernel places SFA and
   SFB TMEM tensors immediately after it using:
     sfa_tmem_ptr = cute.recast_ptr(
         acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
         dtype=sf_dtype,
     )
     tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
         tiled_mma, self.mma_tiler, sf_vec_size,
         cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
     )
     tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

   (a) What does tcgen05.find_tmem_tensor_col_offset(tCtAcc) return,
       and in what units?
   (b) Why is cute.recast_ptr needed when going from Float32 (acc) to
       Float8E4M3FN (scale)?
   (c) What is the purpose of passing the one-stage sfa_smem_layout to
       make_tmem_layout_sfa — why does it need the SMEM layout?""",
hint="TMEM is addressed in 32b columns.  recast_ptr changes the element type without moving the pointer.  TMEM layout must mirror SMEM to make S2T copy correct.",
answer="""\
  (a) tcgen05.find_tmem_tensor_col_offset(tCtAcc) returns the number of
      32-bit TMEM columns occupied by tCtAcc — i.e., the column stride
      needed to place the next tensor immediately after the accumulator
      in TMEM without overlap.  Units are TMEM columns (each column is
      32 bits × number of threads = 32b × 32 = 128 bytes in a warp).

      For a 128×256 Float32 accumulator with the Blackwell TMEM layout,
      this is typically 128 columns (one per M-element group, covering
      the full N extent).

  (b) cute.recast_ptr changes the *element type* interpretation of the
      pointer without changing the underlying hardware address.  After
      adding the column offset (which is in Float32 units), the pointer
      points to a Float32-aligned location.  Recasting to Float8E4M3FN
      tells CuTe that subsequent tensor operations using sfa_tmem_ptr
      should treat each element as 8 bits rather than 32 bits.  This
      is necessary because Float32 and Float8 have different element
      sizes, so the layout arithmetic (offsets, cosize) would be wrong
      if the pointer type were left as Float32.

  (c) The TMEM layout for scale factors must be the "transpose" / mirror
      of the SMEM layout so that the S2T copy (Cp4x32x128bOp) can issue
      a single bulk transfer.  S2T copy reads from SMEM using a descriptor
      and writes to TMEM; the TMEM destination layout must match the
      access pattern of the Cp4x32x128b instruction.

      make_tmem_layout_sfa takes the SMEM layout as a template to derive
      the corresponding TMEM column assignments.  Without this coupling,
      the S2T copy would write scale values to wrong TMEM positions,
      and the MMA instruction would read incorrect scale factors.
""")

qa(5,
"""Similarly, SFB is placed after SFA in TMEM:
     sfb_tmem_ptr = cute.recast_ptr(
         acc_tmem_ptr
         + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
         + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
         dtype=sf_dtype,
     )

   (a) Why must tCtSFA be allocated before tCtSFB (i.e., why does SFB
       offset depend on SFA's size, not independently from the accumulator)?
   (b) The kernel allocates self.num_tmem_alloc_cols=512 total TMEM columns.
       If the accumulator uses 128 columns and SFA uses 64 columns,
       how many columns are left for SFB, and is 512 sufficient?
   (c) Why is num_tmem_alloc_cols fixed at 512 regardless of tile size?""",
hint="TMEM is a flat column array; tensors are packed sequentially.  Blackwell SM100 has 512 TMEM columns per CTA.",
answer="""\
  (a) TMEM is a flat linear array of columns.  There is no allocator
      that automatically finds free gaps — the kernel manages placement
      manually.  The sequence is:
        [0 .. acc_cols-1]          : accumulator
        [acc_cols .. acc+sfa-1]    : SFA
        [acc+sfa .. acc+sfa+sfb-1] : SFB
      SFB must start at acc_cols + sfa_cols, which requires knowing
      sfa_cols = find_tmem_tensor_col_offset(tCtSFA).  There is no
      other way to know where SFA ends without computing its size.

  (b) Columns available for SFB = 512 - 128 - 64 = 320 columns.
      For a 128×256 tile with 256-element K and sf_vec_size=16:
        SFB scale count per tile ≈ N * (K/16) = 256 * 16 = 4096 FP8 values.
        At 8b per FP8 and 32b per TMEM column:
          sfb_cols ≈ 4096 * 8 / 32 = 1024 bits / 32 = 32 columns
          (exact value depends on threading and atom geometry,
           but 32–64 columns is typical for this tile size)
      320 available > ~64 needed for SFB -> 512 is sufficient.

  (c) 512 is the total number of TMEM columns per CTA on SM100 (Blackwell).
      It is a hardware constant, not a per-kernel tunable.  The kernel
      must fit all TMEM tensors (acc + SFA + SFB) within this budget.
      num_tmem_alloc_cols=512 therefore means "allocate the entire TMEM
      column space" rather than a specific count chosen for this tile.
      The underlying sm100.tcgen05.alloc instruction takes a column count;
      passing 512 reserves the full hardware allocation for this CTA.
""")

# ---------------------------------------------------------------------------
section("PART 3 — SMEM-to-TMEM (S2T) Copy for Scale Factors")
# ---------------------------------------------------------------------------

qa(6,
"""The S2T copy path for SFA is:
     copy_atom_s2t = cute.make_copy_atom(
         tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype
     )
     tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
     thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
     tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
     tCsSFA_compact_s2t  = tcgen05.get_s2t_smem_desc_tensor(
                               tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
                           )
     tCtSFA_compact_s2t  = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

   (a) What is Cp4x32x128bOp physically?  What does "4×32×128b" describe?
   (b) Why is get_slice(0) used (thread 0 only) rather than all threads?
   (c) What does tcgen05.get_s2t_smem_desc_tensor do, and why is it
       needed for the source (SMEM) but not the destination (TMEM)?""",
hint="Cp4 = 4 SMEM banks; 32×128b = 32 rows × 128 bits per TMEM column.  S2T uses SMEM descriptors like TMA.",
answer="""\
  (a) tcgen05.Cp4x32x128bOp is the Blackwell SMEM-to-TMEM bulk copy
      instruction.  The dimensions mean:
        4  : number of SMEM bank groups accessed simultaneously
        32 : number of rows (TMEM rows = thread lanes) covered per issue
        128b : bits transferred per row per issue = 128 bits = 16 bytes

      In one Cp4x32x128b instruction, a warp moves 32 × 128b = 512 bytes
      (4KB?) of FP8 scale data from a 4-bank SMEM region into 32 TMEM
      rows.  The "4×" refers to issuing across 4 bank groups in parallel
      to achieve maximum SMEM read bandwidth.

  (b) The Cp4x32x128bOp is a *warp-cooperative* instruction that is
      issued by a single thread but acts on behalf of all 32 lanes.
      The hardware uses the issuing thread's SMEM descriptor pointer;
      all 32 lanes contribute their TMEM destination addresses implicitly.
      Using get_slice(0) sets up the copy for lane 0 (the issuing thread);
      only lane 0 actually executes the `cute.copy(...)` call in the
      k-tile loop — consistent with the entire mainloop being inside
      `if warp_idx == 0`.

  (c) tcgen05.get_s2t_smem_desc_tensor converts the regular SMEM tensor
      tCsSFA_compact_s2t_ (which has normal pointer-based addressing)
      into a *SMEM descriptor tensor* — a tensor whose elements are
      hardware SMEM descriptors rather than raw element values.

      The Cp4x32x128bOp instruction takes the SMEM source as a descriptor
      (analogous to how TMA uses a TMA descriptor) because it needs to
      encode the SMEM bank layout, byte offset, and swizzle in a compact
      hardware-readable format.

      The TMEM destination (tCtSFA_compact_s2t) uses normal TMEM column
      addressing, which the hardware derives from the thread's TMEM base
      pointer and the layout offsets — no descriptor needed.
""")

qa(7,
"""Inside the mainloop, the S2T copy is issued as:
     s2t_stage_coord = (None, None, None, None, ab_full.index)
     tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
     cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged,
               tCtSFA_compact_s2t)

   Note that tCtSFA_compact_s2t has no stage index — only the SMEM
   source is stage-indexed.

   (a) Why does tCtSFA_compact_s2t have no stage axis?
   (b) What ordering relationship must hold between the S2T copy and
       the subsequent cute.gemm calls in the kblock loop?
   (c) The kblock loop then does:
         tiled_mma.set(Field.SFA, tCtSFA[sf_kblock_coord].iterator)
         cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], ...)
       Why does the MMA instruction need Field.SFA to be SET before
       each kblock rather than reading from tCtSFA directly?""",
hint="TMEM is single-buffered for scales.  S2T must complete before MMA reads it.  Field.SFA is an MMA register operand pointer.",
answer="""\
  (a) TMEM scale buffers are single-buffered in this kernel (unlike the
      AB pipeline which has num_ab_stage=4 SMEM stages).  There is only
      one tCtSFA in TMEM because:
        - The S2T copy overwrites tCtSFA before each MMA batch
        - The accumulator (tCtAcc) does not need staging since ACCUMULATE
          mode handles in-place accumulation
        - Adding TMEM scale stages would require more of the 512-column
          TMEM budget, which is already split three ways
      So tCtSFA has shape (MMA, MMA_M, MMA_K) with no stage axis.

  (b) The S2T copy must COMPLETE before any cute.gemm call that uses
      tCtSFA.  The hardware ordering is:
        1. ab_consumer.wait_and_advance()  -> SMEM sSFA is valid
        2. cute.copy(s2t, sSFA, tCtSFA)   -> scale moves to TMEM
        3. [implicit fence]                -> TMEM write visible
        4. tiled_mma.set(Field.SFA, ...)  -> pointer loaded into MMA reg
        5. cute.gemm(...)                 -> MMA reads tCtSFA from TMEM

      Steps 2→4→5 are implicitly ordered because they are issued by
      the same warp (warp 0) in program order.  The tcgen05 architecture
      guarantees that TMEM writes from S2T are visible to subsequent MMA
      instructions issued by the same warp without explicit fences.

  (c) tcgen05.mma is a hardware instruction that takes scale factor
      POINTERS (TMEM column addresses) as operands, not the scale values
      themselves.  The MMA instruction reads scale factors from TMEM at
      the specified column addresses during execution.

      Field.SFA and Field.SFB are like setting an MMA "register" that
      holds the base address of the scale tensor in TMEM.  Each kblock
      uses a different slice of tCtSFA (indexed by kblock_idx), so the
      pointer must be updated per kblock:
        sf_kblock_coord = (None, None, kblock_idx)
        tiled_mma.set(Field.SFA, tCtSFA[sf_kblock_coord].iterator)
      This moves the TMEM read window to the correct K-group for this
      kblock.  Without updating the pointer, all kblocks would read
      the scale from kblock_idx=0, producing wrong results for k>0.
""")

# ---------------------------------------------------------------------------
section("PART 4 — Single-Warp Design vs Warp-Specialized Blockwise")
# ---------------------------------------------------------------------------

qa(8,
"""This NVFP4 kernel uses a dramatically simpler structure: only warp 0
   runs the mainloop (TMA + S2T + MMA), while ALL threads participate
   in the epilogue.  Compare to blockwise_gemm.py's 12 warps.

   (a) What is the key reason this kernel can get away with warp 0
       handling TMA, S2T, AND MMA sequentially in one warp?
   (b) The mainloop pipeline uses:
         ab_producer.acquire_and_advance()
         ab_consumer.wait_and_advance()
       with a single `for k_tile in cutlass.range(k_tile_cnt, prefetch_stages=...)`.
       What does prefetch_stages=num_ab_stage - 2 do?
   (c) blockwise_gemm has separate acc_update warps to scale and
       accumulate intermediate results across K-tiles.  This kernel has
       no acc_update warps.  How does it handle the scale multiplication?""",
hint="Single SM100 warp can issue TMA + S2T + MMA asynchronously.  prefetch_stages is a software pipelining hint.",
answer="""\
  (a) On SM100 (Blackwell), TMA, S2T, and MMA are all asynchronous
      hardware engines with independent completion paths.  A single warp
      can issue all three in a pipelined fashion:
        - Issue TMA (fires async GMEM→SMEM copy)
        - Spin / yield until TMA done
        - Issue S2T (fires async SMEM→TMEM copy)
        - Issue MMA (fires async TMEM multiply-accumulate)
      Because each instruction only consumes a few cycles to *issue*
      (not to *complete*), one warp can keep all three engines busy
      with software pipelining.

      The blockwise kernel uses warp specialization primarily to overlap
      SCALE loading (done with scalar async copy requiring many threads)
      with MMA computation.  NVFP4 loads scales via TMA (fast, one warp)
      so the overlap benefit of dedicating separate warps is smaller.

  (b) prefetch_stages=num_ab_stage - 2 tells the CuTe range to issue
      (num_ab_stage - 2) TMA loads *before* the first MMA invocation,
      pre-filling the SMEM pipeline buffer.  This hides the latency of
      the first few TMA loads so the MMA engine doesn't stall waiting
      for data on iteration 0.

      With num_ab_stage=4 and prefetch=2: by the time the mainloop body
      executes for k_tile=0, tiles 0 and 1 are already in flight.  The
      loop then issues tile 2 while processing tile 0, maintaining a
      2-tile lookahead that hides ~2× TMA latency.

  (c) In NVFP4, scale multiplication is embedded INTO the MMA hardware
      instruction itself (tcgen05.MmaMXF4NVF4Op).  The tcgen05.mma
      opcode reads FP4 operands from SMEM AND scale factors from TMEM
      in one fused operation:
        result += dequant(A_fp4, SFA) * dequant(B_fp4, SFB)
      There is no separate software "apply scale" step needed.

      In blockwise_gemm with MXFP8, the scale multiply is applied in
      software by the acc_update warps AFTER the MMA completes, because
      the tcgen05 FP8 MMA instruction does not support inline scaling
      in the same way.
""")

qa(9,
"""The ab_pipeline in this kernel uses PipelineTmaUmma with:
     ab_pipeline_consumer_group = pipeline.CooperativeGroup(
         pipeline.Agent.Thread, 1
     )

   Compare to blockwise_gemm where the consumer count is num_tma_producer.
   Also, the acc_pipeline consumer group is:
     pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)

   (a) Why does ab_pipeline have consumer count=1 (one thread)?
   (b) Why does acc_pipeline have consumer count=threads_per_cta=128?
   (c) The acc pipeline has num_acc_stage=1 (single-buffered).  What
       is the consequence of this for the mainloop?  When exactly does
       warp 0 release the acc_empty barrier (call acc_empty.commit())?""",
hint="Consumer count must match who calls consumer_release.  Single-stage acc means no overlap between MMA tiles.",
answer="""\
  (a) ab_pipeline consumer count=1 because only ONE thread (warp 0,
      lane 0, or equivalently the single warp operating the mainloop)
      calls ab_full.release() (via ab_full.release() = consumer_release).
      Setting count=1 means the barrier expects exactly one arrival
      before unblocking the TMA producer.  Using more threads would
      require all of them to call release, which doesn't happen here.

  (b) acc_pipeline consumer count=128 (= threads_per_cta) because ALL
      128 threads participate in the epilogue, and all of them call
      acc_full.release() (via acc_full.release() after epilogue store).
      The barrier must wait for all 128 threads to complete before
      it can be reset for the next use.  If count were 1, the barrier
      would trip prematurely after just the first thread releases,
      potentially allowing tmem.free() before other threads finish
      reading the accumulator.

  (c) With num_acc_stage=1, there is ONLY ONE accumulator buffer.
      The single-stage design means:
        - Warp 0 cannot start a new MMA tile until the epilogue has
          fully drained the current accumulator.
        - This eliminates MMA / epilogue overlap entirely.

      acc_empty.commit() is called at the END of the entire k_tile loop
      (after all k_tiles have been processed), not after each k_tile.
      The sequence is:
        acc_empty = acc_producer.acquire_and_advance()  # get the 1 slot
        for k_tile in range(k_tile_cnt):
            ... MMA accumulates into tCtAcc ...          # all k_tiles
        acc_empty.commit()                               # signal epilogue

      The epilogue then does acc_consumer.wait_and_advance() which
      unblocks once after warp 0 commits.  The single-buffer design
      trades performance (no MMA/epilogue pipeline overlap) for
      simplicity — acceptable for a tutorial kernel.
""")

# ---------------------------------------------------------------------------
section("PART 5 — Epilogue Shape: Ld32x32bOp(Repetition.x128)")
# ---------------------------------------------------------------------------

qa(10,
"""The epilogue uses a different TMEM load atom than blockwise_gemm:
     op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
     copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
     tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
     thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
     tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
     tTR_gC   = thr_copy_t2r.partition_D(tCgC)

   (a) blockwise_gemm uses Ld32x32bOp(Repetition(32)).  This uses
       Repetition.x128.  What does the 4× larger repetition count mean
       for the per-thread data volume and the epilogue shape?
   (b) tTR_tAcc and tTR_gC have shape (T2R_M, T2R_N, EPI_M, EPI_N)
       (no leading T2R mode like blockwise_gemm's (T2R, T2R_M, T2R_N, ...)).
       Why is the shape different from blockwise_gemm's epilogue?
   (c) The direct store uses:
         cute.copy(simt_atom, tTR_rC, tTR_gC)
       where simt_atom = CopyUniversalOp().  blockwise_gemm uses TMA
       store.  What are the tradeoffs?""",
hint="x128 repetitions = 128 × 32b per thread.  No T2R mode means the atom already covers the full per-thread tile.",
answer="""\
  (a) Ld32x32bOp(Repetition.x128) loads 128 × 32b = 4096 bits = 512 bytes
      per thread per invocation.  Compare to Repetition(32) = 32 × 32b =
      128 bytes per thread.

      With a 128×256 tile and 128 threads (4 warps):
        Per-thread output = (128 × 256 × 4 bytes) / 128 = 4096 bytes.
        This exactly matches Repetition.x128 × 32b = 512 bytes × ... 
        Actually: (128 × 256 × 4) / 128 = 1024 floats → 4096 bytes per thread,
        requiring 4096/4 = 1024 float32 values per thread.
        Ld32x32bOp(x128) delivers 128 × 32b = 512 bytes = 128 float32 per call,
        so EPI_M × EPI_N = 1024/128 = 8 subtile iterations (EPI_M=2, EPI_N=4 or similar).

      The larger repetition count means fewer TMEM load instructions are
      needed to drain the accumulator, reducing loop overhead in the
      epilogue.  The tradeoff is more registers per thread (128 float32
      in flight simultaneously).

  (b) blockwise_gemm has a leading T2R mode because it uses a more
      general partitioning that supports different accumulator tile sizes
      (64 or 128 M).  The T2R mode encodes the "vector length" of the
      TMEM load atom independently of the spatial tile.

      This NVFP4 kernel uses a fixed tile (128×256) and a fixed 128-thread
      CTA.  make_tmem_copy and partition_S/D flatten the T2R mode directly
      into the spatial (T2R_M, T2R_N) shape because the atom is chosen to
      exactly cover the per-thread tile without a separate vector axis.
      The spatial modes already subsume the "how many elements per atom
      invocation" information.

  (c) SIMT store (CopyUniversalOp) tradeoffs vs TMA store:

      SIMT pros:
        - Simple: each thread directly writes its portion of C to GMEM
        - No SMEM staging buffer needed (saves SMEM and eliminates the
          R→S→G copy chain)
        - Works for any output size without needing a TMA descriptor

      SIMT cons:
        - Each thread issues its own GMEM stores; ~128 separate store
          transactions vs a few TMA bulk transfers
        - Lower bandwidth utilization for large tiles (TMA coalesces better)
        - Must complete before CTA exits (synchronous from warp's perspective)

      TMA store pros:
        - Bulk transfer with hardware coalescing -> higher bandwidth
        - Asynchronous (fire and forget, C pipeline tracks completion)
        - Better for large epilogue tiles (blockwise N=128+)

      For a tutorial kernel targeting 8k×8k problem sizes, the SIMT
      store simplifies the code considerably.  For production kernels
      at this tile size, TMA store would give better epilogue throughput.
""")

# ---------------------------------------------------------------------------
section("PART 6 — cute.assume() Alignment Hints")
# ---------------------------------------------------------------------------

qa(11,
"""Tensors are built with alignment hints:
     a_tensor = cute.make_tensor(
         a_ptr,
         cute.make_layout(
             (m, cute.assume(k, 32), l),
             stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
         ),
     )

   (a) What does cute.assume(k, 32) tell the compiler?
   (b) Why is alignment 32 significant for Float4E2M1FN (4-bit) data?
   (c) The C tensor uses cute.assume(m, 32) and cute.assume(n, 16).
       Why are M and N aligned to different granularities?""",
hint="assume(x, n) hints x is a multiple of n.  32 FP4 values = 128 bits = 16 bytes = one TMA quantum.",
answer="""\
  (a) cute.assume(k, 32) is a compiler hint asserting that the runtime
      value of k is guaranteed to be a multiple of 32.  The JIT compiler
      (LLVM/NVCC backend) uses this to:
        - Eliminate modulo/remainder checks in address computations
        - Enable aligned vectorized loads (e.g., 128-bit LDG instead of
          32-bit)
        - Avoid loop remainder handling for tile boundary conditions
        - Allow the TMA descriptor to use aligned bulk transfer sizes

  (b) Float4E2M1FN is 4 bits per element, so:
        32 elements × 4 bits = 128 bits = 16 bytes.
      16 bytes is one cache line sector and the minimum TMA bulk transfer
      quantum.  Asserting k is a multiple of 32 guarantees that every
      K-tile boundary falls on a 16-byte alignment, which is required for
      TMA to issue correct bulk loads without partial-tile handling.

      Additionally, tcgen05.mma with NVFP4 operates on 32 FP4 elements
      per lane per instruction (64 total in a warpgroup).  Aligning K to
      multiples of 32 ensures the K loop has no remainder iterations.

  (c) C stores Float16 (c_dtype).
        M aligned to 32: 32 × 2 bytes (fp16) = 64 bytes = one cache line.
          This ensures each row of C starts on a cache-line boundary,
          enabling coalesced SIMT stores along the M axis.
        N aligned to 16: 16 × 2 bytes = 32 bytes.
          The epilogue SIMT store writes N-contiguous elements; 16 fp16
          values fit in a 256-bit (32-byte) vectorized store instruction
          (STG.E.128 or equivalent).  N=16 alignment enables these
          maximally-wide stores without padding or masking.

      The different alignments reflect that M is the "slow" axis
      (row start alignment for coalescing) while N is the "fast"
      axis (vector store width for throughput).
""")

qa(12,
"""The C tensor also has stride cute.assume(m * n, 512) for the batch dimension.
     stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512))

   (a) Why is the batch stride aligned to 512 elements (not bytes)?
   (b) The problem constraints require k % 256 == 0 (not k % 32 == 0
       as cute.assume suggests).  Why the discrepancy?
   (c) The argparse validation checks m % 128 == 0, n % 256 == 0,
       k % 256 == 0.  Map each constraint to one specific hardware
       requirement in the kernel.""",
hint="512 fp16 = 1024 bytes = cache-friendly batch offset.  k%256 is the mma_tiler_k.  m/n/k divisibility = no partial tiles.",
answer="""\
  (a) 512 Float16 elements × 2 bytes = 1024 bytes = 1 KB.
      1 KB alignment ensures each batch's C matrix starts on a 1KB
      boundary, which:
        - Aligns with TMA's preferred 1KB-aligned base addresses
        - Avoids L2 cache set conflicts between batches (L2 is typically
          set-associative with 256B or 512B sets)
        - Ensures vectorized TMA store descriptors have the same byte
          offsets for every batch, enabling descriptor reuse across L.

  (b) cute.assume(k, 32) is a *lower bound* alignment hint to the
      compiler — it says "k is at least a multiple of 32."  This is the
      minimum alignment needed for TMA quantum and FP4 atomicity.

      k % 256 == 0 is the *problem shape constraint* — required because
      mma_tiler_k = mma_inst_shape_k * mma_inst_tile_k = 64 * 4 = 256,
      and local_tile / gA_mkl tile the K dimension by 256.  If K is not
      a multiple of 256, the last K-tile would be partial, requiring
      predicated loads — which this tutorial kernel doesn't implement.

      256 is a multiple of 32, so cute.assume(k, 32) is conservative
      but correct; the runtime will always have k divisible by 256 ≥ 32.

  (c) Constraint → hardware requirement:
        m % 128 == 0 : mma_tiler_mn[0]=128.  local_tile partitions M
                       into 128-row tiles; partial M tiles would require
                       boundary masking in TMA and the SIMT epilogue.

        n % 256 == 0 : mma_tiler_mn[1]=256.  Same reason for N.  Also,
                       the SIMT epilogue store assumes a full N tile so
                       tTR_gC has no bounds-check predication.

        k % 256 == 0 : mma_tiler_k = 64 * 4 = 256.  k_tile_cnt =
                       K / mma_tiler_k must be an integer.  Also ensures
                       SFA/SFB scale groups (K/16 per tile) divide evenly
                       and TMA scale loads have aligned sizes.
""")

# ---------------------------------------------------------------------------
section("PART 7 — Synthesis: tracing tCtSFA through the kblock loop")
# ---------------------------------------------------------------------------

qa(13,
"""Walk through one complete k_tile iteration for k_tile=0 in the
   mainloop, focusing on the scale factor path.  Given:
     mma_tiler = (128, 256, 256)
     mma_inst_shape_k = 64
     num_kblocks = size(tCrA, mode=[2]) = mma_tiler_k / mma_inst_shape_k = 4
     sf_vec_size = 16

   (a) How many S2T copy operations are issued per k_tile?  What shape
       is tCsSFA_compact_s2t_staged for one k_tile?
   (b) For kblock_idx in range(4), what is sf_kblock_coord and what
       slice of tCtSFA does each kblock's MMA read?
   (c) If the S2T copy for k_tile=1 is issued before ab_consumer.wait
       for k_tile=1 returns, what goes wrong?""",
hint="One S2T per k_tile, before the kblock loop.  sf_kblock_coord selects K-group within the tile.  S2T must complete before MMA reads TMEM.",
answer="""\
  (a) One S2T copy is issued per k_tile (before the kblock loop), not
      per kblock.  The single S2T fills the entire tCtSFA TMEM buffer
      with all 4 kblocks' scale data at once.

      tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[(None,None,None,None, ab_full.index)]
      This selects the SMEM stage corresponding to the just-filled
      k_tile.  The shape is approximately:
        ((ATOM_V, REST_V), Rest_Tiler, MMA_M, MMA_K_compact)
      where MMA_K_compact = mma_tiler_k / sf_vec_size = 256/16 = 16
      (the 16 distinct scale groups in this tile, after filter_zeros
      removed the inner 16-element broadcast mode).

  (b) sf_kblock_coord = (None, None, kblock_idx):
        Mode 0: None -> keep MMA mode
        Mode 1: None -> keep MMA_M mode
        Mode 2: kblock_idx -> select one K-group within tCtSFA

      Each of the 4 kblocks covers mma_inst_shape_k/sf_vec_size = 64/16 = 4
      scale groups along K.  tCtSFA was written by the S2T copy with
      all 16 K-groups laid out contiguously.  kblock_coord slices:
        kblock 0: K-groups 0..3   (elements 0..63 of K)
        kblock 1: K-groups 4..7   (elements 64..127 of K)
        kblock 2: K-groups 8..11  (elements 128..191 of K)
        kblock 3: K-groups 12..15 (elements 192..255 of K)
      Field.SFA is set to tCtSFA[(None, None, kblock_idx)].iterator,
      pointing the MMA hardware to the right 4-group window.

  (c) If the S2T for k_tile=1 were issued before ab_consumer.wait for
      k_tile=1, then:
        - sSFA for k_tile=1 might not yet be in SMEM (TMA not done)
        - The S2T source (sSFA[stage=1]) contains stale data from the
          previous time that stage was used, or garbage
        - tCtSFA would be overwritten with wrong k_tile=1 scale values
          while the kblock loop for k_tile=0 is still running MMA
          instructions that read tCtSFA

      The MMA for kblocks 2 and 3 of k_tile=0 would read k_tile=1's
      (possibly partial/garbage) scale values instead of k_tile=0's.
      The result would be silently wrong output — no crash, just
      incorrect matrix values.

      The correct ordering enforced by the code is:
        ab_full = ab_consumer.wait_and_advance()  # sSFA k_tile=0 valid
        cute.copy(s2t, sSFA[ab_full.index], tCtSFA)  # fill TMEM
        for kblock in range(4):
            set Field.SFA from tCtSFA slice
            cute.gemm(...)                            # reads TMEM
        ab_full.release()                             # free stage
        # only NOW fetch k_tile=1
""")

qa(14,
"""Final question: compare the TMEM usage pattern of this NVFP4 kernel
   against blockwise_gemm.py across three dimensions.

   Fill in this comparison table (answer each cell):

   Dimension                   | NVFP4 nvfp4_gemm_0.py  | Blockwise blockwise_gemm.py
   ----------------------------+-------------------------+-----------------------------
   TMEM content                | acc + SFA + SFB         | ?
   TMEM stages (acc)           | 1                       | ?
   Who allocates TMEM          | epilog warps             | ?
   Scale in TMEM               | yes (S2T copy)          | ?
   Scale multiply location     | inside MMA instruction  | ?
   TMEM load atom              | Ld32x32bOp(x128)        | ?

   Also: why does blockwise_gemm use multiple acc TMEM stages while
   this kernel uses only 1?""",
hint="blockwise has persistent scheduling with MMA/epilogue overlap; NVFP4 is simpler with sequential tile execution.",
answer="""\
  Dimension                   | NVFP4 (nvfp4_gemm_0)    | Blockwise (blockwise_gemm)
  ----------------------------+-------------------------+-----------------------------
  TMEM content                | acc + SFA + SFB         | acc only (+ final acc offset)
  TMEM stages (acc)           | 1                       | num_acc_stage (3 or 6)
  Who allocates TMEM          | warp 0 (shared alloc)   | epilog warps (warp 4)
  Scale in TMEM               | yes (S2T copy per tile) | no (stays in SMEM/registers)
  Scale multiply location     | inside MMA instruction  | acc_update warps (software)
  TMEM load atom              | Ld32x32bOp(x128, NONE)  | Ld32x32bOp(Rep(32)) or
                              |                         | Ld16x256bOp(Rep(8)) for M=64
  Scale format                | NVFP4: 1×16 per-row     | DeepSeek-style: 128×128 block
                              | (stride-0 on K-inner    | (stride-0 on BOTH M-inner
                              |  size-16 only)          |  and K-inner, size 128 each)

  Why blockwise uses multiple acc stages:
  blockwise_gemm uses a persistent tile scheduler where the MMA warp
  (warp 8) and the acc_update/epilog warps (warps 0-7) run simultaneously
  across DIFFERENT output tiles.  The acc pipeline allows:
    - MMA warp: writes tile T's accumulator to acc stage N
    - acc_update warp: reads tile T-1's accumulator from acc stage N-1
    - These can overlap in time

  With num_acc_stage=3 (for 128×128 tile) or 6 (for 64×64 tile), the
  MMA warp can be 2–5 tiles ahead of the epilogue warp, hiding the
  epilogue latency.

  This NVFP4 kernel has only ONE output tile per CTA (no persistent
  scheduling).  MMA and epilogue are strictly sequential within a CTA —
  MMA completes all K-tiles, then epilogue runs.  There is no tile-level
  overlap to pipeline, so one acc stage suffices.  The single-stage
  design also leaves more TMEM budget for SFA+SFB (which this kernel
  needs but blockwise_gemm does not store in TMEM).
""")

print("\n" + "="*72)
if QUIZ_MODE:
    print("  Self-test mode — re-run without --quiz to reveal all answers.")
else:
    print("  All 14 answers shown. Re-run with --quiz for self-test mode.")
print("="*72 + "\n")