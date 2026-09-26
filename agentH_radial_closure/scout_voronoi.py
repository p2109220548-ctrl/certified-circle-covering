# -*- coding: utf-8 -*-
"""
scout_voronoi.py — Empirical scouting for the structural (Fejes Tóth) roadmap.

For the current best configuration: compute Voronoi cells clipped to D,
per-cell area / side count / circumradius slack, and the boundary waste
(total circle area outside D).  Purpose: measure the constants that the
finite-n hexagonal-tiling lower bound would need (§5 of OPTIMALITY_THEOREM).
No claims — data only.
"""
import json
import os
import sys
import numpy as np
from scipy.spatial import Voronoi

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from geometry import covering_radius  # noqa: E402


def cell_area_clip(center, neighbors, n_arc=720):
    """Area of Voronoi cell of `center` (bisector half-planes vs `neighbors`)
    intersected with the unit disk.  Start from a fine disk polygon, clip by
    each bisector half-plane (keep p: (p-mid)·(q-center) <= 0)."""
    th = np.linspace(0, 2 * np.pi, n_arc, endpoint=False)
    poly = np.stack([np.cos(th), np.sin(th)], axis=1)   # fine disk boundary
    for q in neighbors:
        n_vec = q - center
        mid = 0.5 * (q + center)
        keep = (poly - mid) @ n_vec <= 0
        out = []
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            ka, kb = keep[i], keep[(i + 1) % m]
            if ka:
                out.append(a)
            if ka != kb:
                denom = (b - a) @ n_vec
                t = ((mid - a) @ n_vec) / denom
                out.append(a + t * (b - a))
        poly = np.array(out) if len(out) >= 3 else np.zeros((0, 2))
        if len(poly) < 3:
            return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    d = json.load(open(os.path.join(BASE, "agentD_marathon", "centers_gpu.json")))
    C = np.array(d["centers"], float)
    n = len(C)
    R = covering_radius(C)
    print(f"config: n={n}  R={R:.9f}")
    vor = Voronoi(C)
    rng = np.random.default_rng(7)
    areas, waste_out, in_frac = [], 0.0, []
    n_bnd = 0
    for i in range(n):
        rows, cols = np.where(vor.ridge_points == i)
        others = vor.ridge_points[rows, 1 - cols]
        nb = C[others]
        if len(nb) == 0:
            areas.append(0.0)
            continue
        a = cell_area_clip(C[i], nb)
        areas.append(a)
        di = np.linalg.norm(C[i])
        # circle area inside D (exact-ish cap formula via sampling fine)
        if di + R <= 1:
            fin = np.pi * R * R
        elif di >= 1 + R:
            fin = 0.0
        else:
            th = rng.uniform(0, 2 * np.pi, 400_000)
            pts = C[i] + R * np.stack([np.cos(th), np.sin(th)], axis=1)
            fin = np.pi * R * R * float((np.linalg.norm(pts, axis=1) <= 1).mean())
        waste_out += np.pi * R * R - fin
        in_frac.append(fin / (np.pi * R * R))
        if di > 1 - R:
            n_bnd += 1
    areas = np.array(areas)
    in_frac = np.array(in_frac)
    print(f"sum cell areas          = {areas.sum():.6f}  (pi = {np.pi:.6f})")
    print(f"hexagon benchmark n*f6*R^2 = {n * (3*np.sqrt(3)/2) * R * R:.6f}")
    print(f"boundary centers (d>1-R): {n_bnd}")
    print(f"total circle area outside D (waste) = {waste_out:.6f} "
          f"= {waste_out/(np.pi*R*R*n)*100:.1f}% of total circle area")
    print(f"min/med/max cell area = {areas.min():.5f}/{np.median(areas):.5f}/{areas.max():.5f}")
    print(f"mean circle-frac inside D = {in_frac.mean():.4f}")
    # the empirical 'structural slack': n*f6*R^2 - pi  (what a FT bound must fill)
    slack = n * (3 * np.sqrt(3) / 2) * R * R - np.pi
    print(f"n*f6*R^2 - pi = {slack:.6f}  ->  if waste+cell-shape could be charged, "
          f"R would drop to ~{np.sqrt((slack+np.pi)/(n*(3*np.sqrt(3)/2))):.6f}")


if __name__ == "__main__":
    main()
