from cutlass.cute.layout import Layout, Shape, Stride
from cutlass.cute import zipped_divide, size
# ── Global: 64×64 column-major matrix ──
M, N = 64, 64
gmem = Layout(Shape(M, N), Stride(1, M))
print(f"Global: {gmem}")
# → (64, 64) : (1, 64)
# ══════════════════════════════════════════════
# Level 1: Tile for blocks — (16, 16) block tiles
# ══════════════════════════════════════════════
BLK_M, BLK_N = 16, 16
blk_tiled = zipped_divide(gmem, Shape(BLK_M, BLK_N))
print(f"Block-tiled: {blk_tiled}")
# → ((16, 16), (4, 4)) : ((1, 64), (16, 1024))
#      ↑   ↑     ↑  ↑
#    one tile   4×4 grid of tiles
print(f"Tile grid: {size(blk_tiled, 1)} blocks")
# → 16 blocks total (4 × 4)
# Select block (2, 1)
bm, bn = 2, 1
my_tile = blk_tiled[:, (bm, bn)]
print(f"Block ({bm},{bn}): {my_tile}")
# → (16, 16) : (1, 64)
# base = 2×16 + 1×1024 = 32 + 1024 = 1056
# ══════════════════════════════════════════════
# Level 2: Tile for threads — (4, 4) per thread
# ══════════════════════════════════════════════
THR_M, THR_N = 4, 4
thr_tiled = zipped_divide(my_tile, Shape(THR_M, THR_N))
print(f"Thread-tiled: {thr_tiled}")
# → ((4, 4), (4, 4)) : ((1, 64), (4, 256))
#     ↑  ↑     ↑  ↑
#  per-thr   4×4 thread grid = 16 threads
# Thread (1, 2) in the 4×4 grid
thr_m, thr_n = 1, 2
my_frag = thr_tiled[:, (thr_m, thr_n)]
print(f"Thread ({thr_m},{thr_n}): {my_frag}")
# → (4, 4) : (1, 64)
# base within tile = 1×4 + 2×256 = 4 + 512 = 516
# global base = 1056 + 516 = 1572
# ── Verify: trace every element this thread owns ──
print(f"\nThread ({thr_m},{thr_n}) in block ({bm},{bn}) owns:")
for j in range(THR_N):
    for i in range(THR_M):
        global_offset = my_frag(i, j)
        row = global_offset % M
        col = global_offset // M
        print(f"  ({i},{j}) → offset {global_offset}  (row={row}, col={col})")
# Output:
#   (0,0) → offset 1572  (row=36, col=24)
#   (1,0) → offset 1573  (row=37, col=24)
#   (2,0) → offset 1574  (row=38, col=24)
#   (3,0) → offset 1575  (row=39, col=24)
#   (0,1) → offset 1636  (row=36, col=25)
#   ...
#
# Rows [36..39], Cols [24..27] — a 4×4 submatrix ✓
# row 36 = block_row*16 + thr_m*4 = 2*16 + 1*4 = 36 ✓
# col 24 = block_col*16 + thr_n*4 = 1*16 + 2*4 = 24 ✓