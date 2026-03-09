from cutlass.cute.layout import Layout, Shape, Stride
from cutlass.cute import (
    zipped_divide, logical_divide, composition, size,
    right_inverse, complement
)
# ══════════════════════════════════════════════════
# Part A: local_partition = zipped_divide + slice
# ══════════════════════════════════════════════════
M, N = 128, 64
gmem = Layout(Shape(M, N), Stride(1, M))
BLK_M, BLK_N = 32, 32
bm, bn = 1, 0
# Explicit way (what you learned):
divided = zipped_divide(gmem, Shape(BLK_M, BLK_N))
my_tile_explicit = divided[:, (bm, bn)]
print(f"Explicit: {my_tile_explicit}")
# → (32, 32) : (1, 128)
# local_partition way (one line):
# local_partition(gmem, Shape(BLK_M, BLK_N), (bm, bn))
# → same result: (32, 32) : (1, 128)
# (In CuTe DSL Python, this is typically done via
#  the slice syntax on the divided layout)
# ══════════════════════════════════════════════════
# Part B: Layout composition for thread mapping
# ══════════════════════════════════════════════════
# The real power: compose a THREAD layout with a DATA layout
# to get each thread's view of the data.
# Data tile in shared memory: 16×16 column-major
smem_layout = Layout(Shape(16, 16), Stride(1, 16))
print(f"Shared mem: {smem_layout}")
# → (16, 16) : (1, 16)   — 256 elements
# Thread layout: how 128 threads are arranged over the tile
# 16 threads in M-dim × 8 threads in N-dim = 128 threads
# Each thread owns 1×2 elements (16/16=1 in M, 16/8=2 in N)
thr_layout = Layout(Shape(16, 8), Stride(1, 16))
print(f"Thread layout: {thr_layout}")
# → (16, 8) : (1, 16)  — maps thread_id to (m, n) position
# Value layout: elements per thread
val_layout = Layout(Shape(1, 2), Stride(1, 8))
# Each thread owns 1 row × 2 cols, cols are 8 apart
# ── Compose to get one thread's data view ──
# Thread 5: coordinates in thread layout
thr_id = 5
thr_m = thr_id % 16   # = 5
thr_n = thr_id // 16   # = 0
# This thread's elements in the 16×16 tile:
print(f"\nThread {thr_id} (pos {thr_m},{thr_n}):")
for j in range(2):
    for i in range(1):
        smem_offset = thr_m * 1 + (thr_n * 2 + j) * 16
        # equivalently: smem_layout(thr_m + i, thr_n*2 + j)
        print(f"  val({i},{j}) → smem offset {smem_offset}")
# val(0,0) → offset 5   (row 5, col 0)
# val(0,1) → offset 133  (row 5, col 8... wait)
# ── The composition approach (algebraic) ──
# compose(smem_layout, right_inverse(thr_layout))
# gives a layout that maps thread_id directly to smem offsets
inv_thr = right_inverse(thr_layout)
print(f"\nRight inverse of thr_layout: {inv_thr}")
# This maps a 1D thread index → (m, n) thread coordinate
thr_to_smem = composition(smem_layout, inv_thr)
print(f"Thread→smem composition: {thr_to_smem}")
# Now: thr_to_smem(thread_id) = smem offset for that thread
for t in [0, 1, 5, 16, 17]:
    print(f"  thread {t:2d} → smem offset {thr_to_smem(t)}")
# ══════════════════════════════════════════════════
# Part C: The full pattern in real kernels
# ══════════════════════════════════════════════════
# In a real CUTLASS kernel, the flow is:
#
#   1. blk_tile = zipped_divide(gmem, blk_shape)[:, blockIdx]
#   2. thr_data = composition(blk_tile, thread_layout_inv)
#   3. Each thread reads/writes via thr_data(threadIdx)
#
# Or equivalently with local_partition:
#   1. blk_tile = local_partition(gmem, blk_shape, blockIdx)
#   2. thr_tile = local_partition(blk_tile, thr_shape, threadIdx)
#   3. for i in range(size(thr_tile)): process(thr_tile(i))
#
# Both approaches use zipped_divide under the hood.
# Composition is more algebraic; local_partition is more direct.