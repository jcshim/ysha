---

## 1. 문제 상황

* 큰 도형(외곽 사각형) 안에 \*\*구멍(금지영역)\*\*이 있음.
* 작은 직사각형 부품(가로/세로 두 방향 가능)을 **겹치지 않게 최대한 많이 배치**하고 싶음.
* 부품 간/경계와의 \*\*클리어런스(여유거리)\*\*와 \*\*가로-세로 배치 간 최소 간격(gap\_hv)\*\*도 고려해야 함.

---

## 2. 코드의 주요 흐름

1. **배치 영역 정의**

   * `outer`: PCB 외곽
   * `hole`: 금지 구역
   * `area`: 외곽 – 구멍
   * `inner_area = area.buffer(-clearance)` → 클리어런스를 적용한 실제 배치 영역

2. **부품 크기와 피치(pitch)**

   * 가로 방향 부품: `(w=3.35, h=1.10)`
   * 세로 방향 부품: `(w=1.10, h=3.35)`
   * pitch = 부품 크기 × 1.05 → 5% 여유

3. **배치 함수 `place_rectangles_with_shift()`**

   * 격자 좌표를 돌면서 직사각형(rect)을 생성.
   * `rect.within(area_poly)` → 배치 가능 영역 안에 있는지 확인.
   * 기존 배치(`placed_list`)와 겹치면 버림.
   * 세로 배치 시에는 `gap_hv`만큼 버퍼 처리해서 가로 부품과의 최소 간격 보장.

4. **Packing 알고리즘 `run_packing()`**

   * **가로 부품 먼저 배치(H-first)**
   * 이후 남는 공간에 **세로 부품 채우기(V-fill)**
   * 각각 `0` 또는 `0.5 pitch`만큼 시작점을 시프트해 보면서 **최적 조합** 탐색.

5. **결과 출력**

   * 총 배치된 부품 개수, 가로/세로 개수, 사용한 시프트 값 표시.
   * 옵션으로 CSV 저장 가능.
   * `matplotlib`으로 외곽, 구멍, 클리어런스 영역, 배치된 부품 시각화.

---

## 3. 핵심 아이디어

* **클리어런스**: 경계와 구멍에서 일정 거리 떨어뜨려 안전한 배치.
* **가로 우선 배치**: 효율적으로 공간 채우기.
* **세로 채우기**: 가로로 못 쓰는 좁은 공간을 활용.
* **시프트 탐색**: 시작점을 살짝 옮겨 가장 많은 부품이 들어가는 배치 선택.
* **gap\_hv**: 가로와 세로 부품이 **겹쳐 보이거나 붙는 문제**를 방지.

---

👉 한마디로:
**“가로로 꽉 채우고, 남은 틈새를 세로로 메우되, 경계와 서로 일정 간격을 두는” 자동 배치기**입니다.

```
# -*- coding: utf-8 -*-
"""
H-first + V-fill (with 0/½ pitch shift), clearance, and horizontal–vertical gap
- Fixes the “overlapping line” look by enforcing min gap between H/V sets.
- Compatible with Shapely 1.x and 2.x (safe STRtree query wrapper that always returns a list).
"""

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import csv
import sys

# -----------------------------
# Parameters
# -----------------------------
# Placement area (outer with a rectangular hole)
outer = Polygon([(0,0),(100,0),(100,80),(0,80)])
hole  = Polygon([(40,20),(60,20),(60,40),(40,40)])
area  = Polygon(outer.exterior.coords, holes=[hole.exterior.coords])

# Part size (H orientation)
w, h = 3.35, 1.10
pitch_x, pitch_y = w*1.05, h*1.05   # add ~5% spacing

# Part size for V orientation (90° rotated)
w_v, h_v = h, w
pitch_x_v, pitch_y_v = w_v*1.05, h_v*1.05

# Manufacturing clearance from outer walls / hole (mm)
clearance = 0.10
inner_area = area.buffer(-clearance)

# If clearance too large, nothing to place
if inner_area.is_empty:
    print(f"[ERROR] inner_area is empty. Reduce clearance (current={clearance}).")
    sys.exit(1)

# Enforce a small visual gap between horizontal and vertical sets (mm)
gap_hv = 0.02

# Search grid phase shifts (fractions of pitch)
SHIFT_CAND = [0.0, 0.5]

# Optional: save CSV
SAVE_CSV = False
CSV_PATH = "placed_components.csv"

# -----------------------------
# Utilities
# -----------------------------
def strtree_hits(tree, geom):
    """
    Safe STRtree query for Shapely 1.x/2.x:
    - Try predicate='intersects' (Shapely 2.x)
    - Fallback to bbox query then exact intersects filter (Shapely 1.x)
    Always returns a Python list (never a numpy array) to make len(...) safe.
    """
    if tree is None:
        return []
    try:
        return list(tree.query(geom, predicate='intersects'))  # Shapely 2.x
    except TypeError:
        cands = tree.query(geom)                               # Shapely 1.x
        return [g for g in cands if isinstance(g, BaseGeometry) and geom.intersects(g)]

def place_rectangles_with_shift(area_poly, rect_w, rect_h, step_x, step_y,
                                shift_x=0.0, shift_y=0.0,
                                placed_list=None, rebuild_every=400,
                                avoid_tree=None, min_gap_to_avoid=0.0):
    """
    Grid-scan placement of axis-aligned rect_w x rect_h rectangles inside area_poly.
    - placed_list: existing geometries to avoid (will add into this list)
    - avoid_tree + min_gap_to_avoid: ensure candidate rect (buffered by gap/2) does NOT
      intersect the avoid_tree set (used to keep a minimum H–V gap).
    """
    if placed_list is None:
        placed_list = []

    minx, miny, maxx, maxy = area_poly.bounds

    # Start at half-extent plus shift * pitch; clamp into bounds
    x0 = rect_w/2.0 + shift_x*step_x
    y0 = rect_h/2.0 + shift_y*step_y
    while x0 < minx + rect_w/2.0:
        x0 += step_x
    while y0 < miny + rect_h/2.0:
        y0 += step_y

    tree = STRtree(placed_list) if placed_list else None

    y = y0
    while y <= maxy - rect_h/2.0:
        x = x0
        while x <= maxx - rect_w/2.0:
            rect = box(x - rect_w/2.0, y - rect_h/2.0, x + rect_w/2.0, y + rect_h/2.0)

            if rect.within(area_poly):
                ok = True

                # Avoid existing placed_list (both H and already-placed V)
                if tree:
                    hits = strtree_hits(tree, rect)
                    if len(hits) > 0:
                        ok = False
                else:
                    for other in placed_list:
                        if isinstance(other, BaseGeometry) and rect.intersects(other):
                            ok = False
                            break

                # Keep a min gap to the "avoid" set (all H parts)
                if ok and avoid_tree is not None and min_gap_to_avoid > 0.0:
                    rect_g = rect.buffer(min_gap_to_avoid/2.0, join_style=2)
                    hits_gap = strtree_hits(avoid_tree, rect_g)
                    if len(hits_gap) > 0:
                        ok = False

                if ok:
                    placed_list.append(rect)
                    if len(placed_list) % rebuild_every == 0:
                        tree = STRtree(placed_list)

            x += step_x
        y += step_y

    return placed_list

# -----------------------------
# Packing: H-first then V-fill with shift search
# -----------------------------
def run_packing():
    best = None
    best_total = -1

    for sx_h in SHIFT_CAND:
        for sy_h in SHIFT_CAND:
            # 1) Place horizontal first
            placed_h = []
            placed_h = place_rectangles_with_shift(
                inner_area, w, h, pitch_x, pitch_y,
                shift_x=sx_h, shift_y=sy_h, placed_list=placed_h
            )
            horiz_tree = STRtree(placed_h) if placed_h else None

            # 2) Fill remaining with vertical; enforce H–V min gap
            best_v_cfg = None
            best_v_total = -1

            for sx_v in SHIFT_CAND:
                for sy_v in SHIFT_CAND:
                    placed_all = placed_h.copy()
                    placed_all = place_rectangles_with_shift(
                        inner_area, w_v, h_v, pitch_x_v, pitch_y_v,
                        shift_x=sx_v, shift_y=sy_v,
                        placed_list=placed_all,
                        avoid_tree=horiz_tree, min_gap_to_avoid=gap_hv
                    )
                    vertical_only = placed_all[len(placed_h):]
                    total = len(placed_all)
                    if total > best_v_total:
                        best_v_total = total
                        best_v_cfg = {
                            "placed_all": placed_all,
                            "placed_h": placed_h,
                            "placed_v": vertical_only,
                            "shift_h": (sx_h, sy_h),
                            "shift_v": (sx_v, sy_v),
                        }

            if best_v_total > best_total:
                best_total = best_v_total
                best = best_v_cfg

    return best

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    best = run_packing()

    print("[Clearance + H/V split + H–V min gap]")
    print(f"- clearance: {clearance} mm")
    print(f"- gap_hv   : {gap_hv} mm  (min distance between H and V sets)")
    print(f"- H shift  : {best['shift_h']}")
    print(f"- V shift  : {best['shift_v']}")
    print(f"- H count  : {len(best['placed_h'])}")
    print(f"- V count  : {len(best['placed_v'])}")
    print(f"- Total    : {len(best['placed_all'])}")

    # Optional CSV
    if SAVE_CSV:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["type","x_min","y_min","x_max","y_max"])
            for r in best["placed_h"]:
                x0,y0,x1,y1 = r.bounds
                wri.writerow(["H",x0,y0,x1,y1])
            for r in best["placed_v"]:
                x0,y0,x1,y1 = r.bounds
                wri.writerow(["V",x0,y0,x1,y1])
        print(f"CSV saved: {CSV_PATH}")

    # Plot (single plot, default colors)
    fig, ax = plt.subplots(figsize=(8,6))
    ex_x, ex_y = area.exterior.xy
    ax.plot(ex_x, ex_y, linewidth=2, label="area")
    for hole_poly in area.interiors:
        hx, hy = hole_poly.xy
        ax.plot(hx, hy, linewidth=2, label="hole")
    (iex, iey) = inner_area.exterior.xy
    ax.plot(iex, iey, linestyle='--', linewidth=1, label="inner (clearance)")

    for r in best["placed_h"]:
        rx, ry = r.exterior.xy
        ax.plot(rx, ry, linewidth=0.6)
    for r in best["placed_v"]:
        rx, ry = r.exterior.xy
        ax.plot(rx, ry, linewidth=0.6)

    ax.set_aspect('equal')
    ax.set_title(f"Total={len(best['placed_all'])} (gap_hv={gap_hv})")
    ax.legend(loc="upper right")
```
    plt.show()
