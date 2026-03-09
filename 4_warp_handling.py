from cutlass.cute.layout import Layout, Shape, Stride
from cutlass.cute import zipped_divide, size
# ── 512 elements, 128 threads (4 warps × 32 lanes) ──
# Each thread handles 512/128 = 4 elements
N_BLK = 512
N_WARPS = 4
WARP_SIZE = 32
N_THR = N_WARPS * WARP_SIZE  # 128
blk = Layout(N_BLK, 1)  # (512) : (1)
# ═══ Flat tile (loses warp structure) ═══
flat = zipped_divide(blk, N_THR)
print(f"Flat tile: {flat}")
# → ((128), (4)) : ((1), (128))
#     ↑       ↑
#  all threads  iterations
# mode-0 is just (128) — no warp/lane distinction!
# ═══ Hierarchical tile (preserves warp structure) ═══
hier = zipped_divide(blk, Shape(Shape(WARP_SIZE, N_WARPS)))
print(f"Hierarchical tile: {hier}")
# → (((32, 4)), (4)) : (((1, 32)), (128))
#      ↑   ↑      ↑
#    lane  warp   iters
# Now mode-0 has shape (32, 4) — directly (lane, warp)!
# ── Indexing by (lane_id, warp_id, iteration) ──
lane_id = 5
warp_id = 2
print(f"\nWarp {warp_id}, Lane {lane_id} accesses:")
for i in range(size(hier, 1)):
    offset = hier((lane_id, warp_id), i)
    print(f"  iter {i} → offset {offset}")
# iter 0 → 5 + 2*32 = 69
# iter 1 → 69 + 128 = 197
# iter 2 → 69 + 256 = 325
# iter 3 → 69 + 384 = 453
# ── Slice one warp's data ──
# With hierarchical tile, you can reason about warp 2's elements:
#   All of warp 2's elements across all iterations:
#   Lanes 0-31 → offsets [64..95], [192..223], [320..351], [448..479]
#   These are 4 contiguous 32-element chunks at stride 128
#
# With flat tile (128), you'd need:
#   thread_id = lane_id + warp_id * 32  (manual!)
#   No way to slice "all of warp 2" from the layout
# ── Why this matters for warp shuffle ──
# After processing, each thread has a partial sum.
# Warp shuffle reduction needs lanes 0-31 of the SAME warp.
# Hierarchical tile guarantees warp 2's lanes are at:
#   offsets 64, 65, 66, ..., 95  (iter 0) — consecutive!
# So __shfl_down_sync naturally matches the memory layout.