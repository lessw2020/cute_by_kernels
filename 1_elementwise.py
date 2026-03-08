from cutlass.cute.layout import Layout, Shape, Stride
from cutlass.cute import zipped_divide, size
# ── Setup: 1024 elements, 256 threads per block ──
N = 1024
BLOCK_SIZE = 256
gmem = Layout(N, 1)  # (1024) : (1)  — contiguous 1D
print(f"Global layout: {gmem}")
# → (1024):(1)
# ── Level 1: Tile for thread blocks ──
blocked = zipped_divide(gmem, BLOCK_SIZE)
print(f"After block divide: {blocked}")
# → ((256), (4)) : ((1), (256))
#      ↑      ↑
#   one blk  which blk (1024/256 = 4 blocks)
# Simulate blockIdx.x = 2
blockIdx_x = 2
blk_layout = blocked[:, blockIdx_x]
print(f"Block {blockIdx_x} layout: {blk_layout}")
# → (256) : (1)  but offset by 2*256 = 512
# ── Level 2: Tile for threads (1 element per thread) ──
threaded = zipped_divide(blk_layout, 1)
print(f"After thread divide: {threaded}")
# → ((1), (256)) : ((1), (1))
#     ↑      ↑
#  per-thr  which thread
# Each thread just does: C[idx] = A[idx] + B[idx]
# Thread t in block b accesses offset: b*256 + t
for threadIdx_x in range(4):  # show first 4 threads
    idx = threaded(0, threadIdx_x)
    print(f"  thread {threadIdx_x} → global offset {idx}")
# thread 0 → 512, thread 1 → 513, thread 2 → 514, ...
