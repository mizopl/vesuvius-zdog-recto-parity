#!/usr/bin/env python3
"""
Builds smooth, prize-sized (>= 4.5 - 8.5 cm^2) .tifxyz surface meshes directly
from S3 `normal-grids/xy/{z:06d}.grid` (VCGS big-endian binary format) for any
unread Herculaneum scroll.

Key facts verified on S3:
1. Header is big-endian `>11I`:
   `magic, version, bx, by, bw, bh, cell_size, num_buckets, num_paths, buckets_offset, paths_offset`
2. Each path starts at `paths_offset` with big-endian `>3I` (`sx, sy, n_off`),
   followed by `n_off` int8 delta bytes (`(n_off // 2, 2)`), where consecutive
   points are ALREADY spaced at `spiral-step = 20.0` voxels!
3. Files `xy/{z:06d}.grid` are indexed directly by voxel slice `z` (`0 .. volZ-1`).
"""
import concurrent.futures
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import tifffile


def load_grid_paths(url: str, min_pts: int = 40):
    try:
        data = urllib.request.urlopen(url, timeout=15).read()
    except Exception:
        return []
    if len(data) < 44:
        return []
    magic, version, bx, by, bw, bh, cell_size, num_buckets, num_paths, buckets_offset, paths_offset = struct.unpack(">11I", data[:44])
    pos = paths_offset
    paths = []
    for _ in range(num_paths):
        if pos + 12 > len(data):
            break
        sx, sy, n_off = struct.unpack(">3I", data[pos:pos + 12])
        if pos + 12 + n_off > len(data):
            break
        deltas = np.frombuffer(data, dtype=np.int8, count=n_off, offset=pos + 12).reshape(-1, 2)
        pts = np.vstack([[sx, sy], [sx, sy] + np.cumsum(deltas, axis=0)]).astype(np.float32)
        if len(pts) >= min_pts:
            paths.append(pts)
        pos += 12 + n_off
    return paths


def build_meshes_for_scroll(scroll: str, ng_prefix: str, shape_zyx, um: float, out_root: Path,
                            max_cols: int = 160, half_steps: int = 24, z_step: int = 60, top_k_wraps: int = 3):
    volZ, volY, volX = shape_zyx
    base_xy_url = f"https://vesuvius-challenge-open-data.s3.amazonaws.com/{ng_prefix}/xy"

    # Probe 3 candidate Z centers (38%, 50%, 62% of volZ) and pick the richest slice
    cand_zs = [int(volZ * 0.38), int(volZ * 0.50), int(volZ * 0.62)]
    z0 = cand_zs[1]
    paths0 = []
    for cz in cand_zs:
        ps = load_grid_paths(f"{base_xy_url}/{cz:06d}.grid", min_pts=90)
        if len(ps) > len(paths0):
            paths0 = ps
            z0 = cz
    if not paths0:
        paths0 = load_grid_paths(f"{base_xy_url}/{z0:06d}.grid", min_pts=50)

    paths0.sort(key=lambda p: len(p), reverse=True)
    if not paths0:
        print(f"[{scroll}] WARNING: No paths found near z={z0}", flush=True)
        return []

    # Pick top_k_wraps distinct wraps separated by >= 140 voxels in XY
    seeds = []
    cx, cy = volX / 2.0, volY / 2.0
    for p in paths0:
        if len(p) > max_cols:
            s0 = (len(p) - max_cols) // 2
            p = p[s0:s0 + max_cols].copy()
        # Orient counter-clockwise consistently
        v0 = p[0] - np.array([cx, cy])
        v1 = p[-1] - np.array([cx, cy])
        if (v0[0] * v1[1] - v0[1] * v1[0]) < 0:
            p = p[::-1].copy()
        c_p = p.mean(axis=0)
        if all(np.linalg.norm(c_p - s.mean(axis=0)) > 140.0 for s in seeds):
            seeds.append(p)
        if len(seeds) >= top_k_wraps:
            break

    zs = [z0 + k * z_step for k in range(-half_steps, half_steps + 1) if 100 <= z0 + k * z_step < volZ - 100]
    z0_idx = zs.index(z0)

    print(f"[{scroll}] Fetching {len(zs)} slices ({zs[0]}..{zs[-1]}, step={z_step}) around z={z0} ...", flush=True)
    grid_trees = {}

    def _fetch(z_val):
        ps = load_grid_paths(f"{base_xy_url}/{z_val:06d}.grid", min_pts=15)
        all_p = np.vstack(ps) if ps else np.zeros((1, 2), dtype=np.float32)
        return z_val, (cKDTree(all_p), all_p)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for z_val, tree_tup in ex.map(_fetch, zs):
            grid_trees[z_val] = tree_tup

    z_ks = np.array(zs, dtype=np.float32)
    z_rows = np.arange(z_ks[0], z_ks[-1] + 0.1, 20.0, dtype=np.float32)
    saved = []

    for w_idx, seed in enumerate(seeds):
        U = len(seed)
        K = len(zs)
        xy_k = np.zeros((K, U, 2), dtype=np.float32)
        xy_k[z0_idx] = seed

        for direction in [1, -1]:
            idx = z0_idx
            while 0 <= idx + direction < K:
                prev = xy_k[idx]
                nxt_idx = idx + direction
                tree, pts = grid_trees[zs[nxt_idx]]
                # Compute unit normal of prev curve so points track radially without tangential slip
                diffs = np.gradient(prev, axis=0)
                norms = np.hypot(diffs[:, 0], diffs[:, 1])[:, None] + 1e-6
                tan_vec = diffs / norms
                nor_vec = np.stack([-tan_vec[:, 1], tan_vec[:, 0]], axis=-1)

                dists, indices = tree.query(prev, k=1)
                matched = pts[indices]
                valid = dists < 18.0
                shift_n = np.zeros(U, dtype=np.float32)
                if valid.any():
                    delta = matched[valid] - prev[valid]
                    shift_n[valid] = np.sum(delta * nor_vec[valid], axis=1)
                    # Fill invalid entries from neighbors and smooth along U
                    if not valid.all():
                        good_u = np.where(valid)[0]
                        bad_u = np.where(~valid)[0]
                        shift_n[bad_u] = np.interp(bad_u, good_u, shift_n[good_u])
                    shift_n = gaussian_filter(shift_n, sigma=3.0, mode="nearest")
                xy_k[nxt_idx] = prev + shift_n[:, None] * nor_vec
                idx = nxt_idx

        # Interpolate from z_ks (step 60) to z_rows (step 20)
        R = len(z_rows)
        x_grid = np.zeros((R, U), dtype=np.float32)
        y_grid = np.zeros((R, U), dtype=np.float32)
        z_grid = np.repeat(z_rows[:, None], U, axis=1)
        for u in range(U):
            x_grid[:, u] = np.interp(z_rows, z_ks, xy_k[:, u, 0])
            y_grid[:, u] = np.interp(z_rows, z_ks, xy_k[:, u, 1])

        x_grid = gaussian_filter(x_grid, sigma=(1.0, 1.0), mode="nearest").astype(np.float32)
        y_grid = gaussian_filter(y_grid, sigma=(1.0, 1.0), mode="nearest").astype(np.float32)

        H_px, W_px = R * 20, U * 20
        area_cm2 = (H_px * um * 1e-4) * (W_px * um * 1e-4)
        seg_name = f"grid_wrap{w_idx:02d}_z{z0:05d}"
        seg_dir = out_root / scroll / seg_name
        seg_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(seg_dir / "x.tif"), x_grid)
        tifffile.imwrite(str(seg_dir / "y.tif"), y_grid)
        tifffile.imwrite(str(seg_dir / "z.tif"), z_grid)
        meta = {
            "uuid": seg_name,
            "scroll": scroll,
            "format": "tifxyz",
            "scale": [0.05, 0.05],
            "area_cm2": round(float(area_cm2), 3),
            "shape_grid": [int(R), int(U)],
            "shape_px": [int(H_px), int(W_px)],
            "z_range": [int(z_rows[0]), int(z_rows[-1])],
        }
        (seg_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        dx = np.diff(x_grid, axis=1)
        dy = np.diff(y_grid, axis=1)
        step_u = np.hypot(dx, dy)
        print(f"  -> {scroll}/{seg_name}: grid={R}x{U} ({H_px}x{W_px} px), area={area_cm2:.2f} cm^2, step_u_med={np.median(step_u):.1f} vox, jumps(>40)={np.mean(step_u>40)*100:.2f}%", flush=True)
        saved.append((seg_name, seg_dir, float(area_cm2)))
    return saved


if __name__ == "__main__":
    m_path = Path("./manifest_19_scrolls.json")
    manifest = json.load(open(m_path))
    test_scroll = sys.argv[1] if len(sys.argv) > 1 else "PHerc0125"
    info = manifest[test_scroll]
    out_dir = Path("./meshes")
    build_meshes_for_scroll(test_scroll, info["normal_grids_prefix"], info["shape_zyx"], info["um"], out_dir)
