#!/usr/bin/env python3
"""
Stage I (Etap I.1 v3) — Local CPU Candidate Builder + Thin-Lamina 3D m7-Zarr Wavefront Snap
+ Inward Normal Canonicalization (`Bet 5` >= 28 candidates with `m7_prob_after >= 120` & `atten_pm4 >= 62%`).
"""
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import imagecodecs
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from build_tifxyz_from_grids import load_grid_paths
from normalize_pherc1203_meshes import normalize_tifxyz
from snap_meshes_to_m7_zarr import compute_smooth_mesh_normals

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_MESH_ROOT = BASE_DIR / "meshes" / "stage_i_m7snap"
OUT_MESH_ROOT.mkdir(parents=True, exist_ok=True)

SCROLL_CONFIGS = {
    "PHerc1203_77keV": {
        "ct_zarr_url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr/2",
        "m7_zarr_0": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr/0",
        "ng_prefix": "PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.normal-grids",
        "shape_zyx": (3785, 6624, 6624),
        "um": 9.612,
        "umbilicus_xy": (3312.0, 3312.0),
    },
    "PHerc0846A_77keV": {
        "ct_zarr_url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr/2",
        "m7_zarr_0": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/representations/predictions/surfaces/20260319102732-surface-20260413222639-surface-m7-L2-th0.2.zarr/0",
        "ng_prefix": "PHerc0846A/representations/predictions/surfaces/20260319102732-surface-20260413222639-surface-m7-L2-th0.2.normal-grids",
        "shape_zyx": (3785, 7397, 7397),
        "um": 9.612,
        "umbilicus_xy": (3698.0, 3698.0),
    },
    "PHerc1203": {
        "ct_zarr_url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/0",
        "m7_zarr_0": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/representations/predictions/surfaces/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr/0",
        "ng_prefix": "PHerc1203/representations/predictions/surfaces/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.normal-grids",
        "shape_zyx": (18977, 6844, 6844),
        "um": 9.362,
        "umbilicus_xy": (3364.0, 3947.0),
    },
    "PHerc0800": {
        "ct_zarr_url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0",
        "m7_zarr_0": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/representations/predictions/surfaces/20250521135224-surface-20260413222639-surface-m7-L0-th0.2.zarr/0",
        "ng_prefix": "PHerc0800/representations/predictions/surfaces/20250521135224-surface-20260413222639-surface-m7-L0-th0.2.normal-grids",
        "shape_zyx": (24298, 9867, 9867),
        "um": 8.640,
        "umbilicus_xy": (4442.0, 5492.0),
    },
}

_m19 = json.loads((BASE_DIR / "manifest_19_scrolls.json").read_text())
for sc_name in [
    "PHerc0125", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
    "PHerc0358", "PHerc0813", "PHerc0826", "PHerc1218", "PHerc1545",
]:
    inf = _m19[sc_name]
    ng_pref = inf["normal_grids_prefix"]
    m7_pref = ng_pref.replace(".normal-grids", ".zarr/0")
    SCROLL_CONFIGS[sc_name] = {
        "ct_zarr_url": f"https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/{sc_name}/volumes/{inf['volume']}/0",
        "m7_zarr_0": f"https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/{m7_pref}",
        "ng_prefix": ng_pref,
        "shape_zyx": tuple(inf["shape_zyx"]),
        "um": float(inf["um"]),
        "umbilicus_xy": (inf["shape_zyx"][2] / 2.0, inf["shape_zyx"][1] / 2.0),
    }


def fetch_m7_chunks_generic(scroll_key: str, chunk_keys: set[tuple[int, int, int]]) -> dict[tuple[int, int, int], np.ndarray]:
    cfg = SCROLL_CONFIGS[scroll_key]
    base_url = cfg["m7_zarr_0"]
    cache_root = BASE_DIR / "m7_cache" / scroll_key
    cache_root.mkdir(parents=True, exist_ok=True)

    chunks_dict: dict[tuple[int, int, int], np.ndarray] = {}
    to_download = []

    for (cz, cy, cx) in sorted(chunk_keys):
        local_p = cache_root / f"{cz}_{cy}_{cx}.blosc"
        if local_p.exists():
            raw = local_p.read_bytes()
            if len(raw) > 0:
                arr = np.frombuffer(imagecodecs.blosc_decode(raw), dtype=np.uint8).reshape(192, 192, 192)
                chunks_dict[(cz, cy, cx)] = arr
        else:
            to_download.append((cz, cy, cx, local_p))

    def _dl_one(item):
        cz, cy, cx, local_p = item
        url = f"{base_url}/{cz}/{cy}/{cx}"
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "vesuvius-stage-i/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                local_p.write_bytes(data)
                arr = np.frombuffer(imagecodecs.blosc_decode(data), dtype=np.uint8).reshape(192, 192, 192)
                return (cz, cy, cx), arr
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    local_p.write_bytes(b"")
                    return (cz, cy, cx), None
                time.sleep(0.3 * (attempt + 1))
            except Exception:
                time.sleep(0.3 * (attempt + 1))
        return (cz, cy, cx), None

    if to_download:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex:
            for key, arr in ex.map(_dl_one, to_download):
                if arr is not None:
                    chunks_dict[key] = arr
        print(f"  [m7 S3] Downloaded {len(to_download)} chunks for {scroll_key} in {time.time()-t0:.1f}s (active={len(chunks_dict)})", flush=True)

    return chunks_dict


def sample_m7_profiles_generic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    s_grid: np.ndarray,
    scroll_key: str,
) -> np.ndarray:
    S = len(s_grid)
    rows, cols = x.shape
    Z_max, Y_max, X_max = SCROLL_CONFIGS[scroll_key]["shape_zyx"]

    s_b = s_grid[:, None, None].astype(np.float32)
    Xs = np.clip(x[None, :, :] + s_b * nx[None, :, :], 0.0, float(X_max - 2.001))
    Ys = np.clip(y[None, :, :] + s_b * ny[None, :, :], 0.0, float(Y_max - 2.001))
    Zs = np.clip(z[None, :, :] + s_b * nz[None, :, :], 0.0, float(Z_max - 2.001))

    cz_all = (Zs.astype(np.int32)) // 192
    cy_all = (Ys.astype(np.int32)) // 192
    cx_all = (Xs.astype(np.int32)) // 192
    unique_triplets = np.unique(np.stack([cz_all.ravel(), cy_all.ravel(), cx_all.ravel()], axis=1), axis=0)
    chunk_keys = {(int(t[0]), int(t[1]), int(t[2])) for t in unique_triplets}

    chunks_dict = fetch_m7_chunks_generic(scroll_key, chunk_keys)

    M = np.zeros((S, rows, cols), dtype=np.float32)
    Zi = np.rint(Zs).astype(np.int32)
    Yi = np.rint(Ys).astype(np.int32)
    Xi = np.rint(Xs).astype(np.int32)

    cz_i = Zi // 192
    cy_i = Yi // 192
    cx_i = Xi // 192
    lz_i = Zi % 192
    ly_i = Yi % 192
    lx_i = Xi % 192

    for (cz, cy, cx), chunk in chunks_dict.items():
        mask = (cz_i == cz) & (cy_i == cy) & (cx_i == cx)
        if not np.any(mask):
            continue
        lz = lz_i[mask]
        ly = ly_i[mask]
        lx = lx_i[mask]
        v0 = chunk[lz, ly, lx].astype(np.float32)
        vz1 = chunk[np.minimum(191, lz + 1), ly, lx].astype(np.float32)
        vz0 = chunk[np.maximum(0, lz - 1), ly, lx].astype(np.float32)
        vy1 = chunk[lz, np.minimum(191, ly + 1), lx].astype(np.float32)
        vy0 = chunk[lz, np.maximum(0, ly - 1), lx].astype(np.float32)
        vx1 = chunk[lz, ly, np.minimum(191, lx + 1)].astype(np.float32)
        vx0 = chunk[lz, ly, np.maximum(0, lx - 1)].astype(np.float32)
        M[mask] = 0.40 * v0 + 0.10 * (vz1 + vz0 + vy1 + vy0 + vx1 + vx0)

    return M


def solve_thin_lamina_m7_snap(
    M: np.ndarray,
    s_grid: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    S, rows, cols = M.shape
    M_smooth = gaussian_filter(M, sigma=(0.75, 1.15, 1.15))

    step_s = float(s_grid[1] - s_grid[0])
    shift_4 = int(round(4.0 / step_s))
    M_m4 = np.pad(M_smooth, ((shift_4, 0), (0, 0), (0, 0)), mode="edge")[:S]
    M_p4 = np.pad(M_smooth, ((0, shift_4), (0, 0), (0, 0)), mode="edge")[shift_4:]
    R_thin = M_smooth - 0.58 * 0.5 * (M_m4 + M_p4)

    s_star = np.zeros((rows, cols), dtype=np.float32)
    snapped = seed_mask.copy()

    seed_band = np.abs(s_grid)[:, None, None] <= 5.5
    prior_seed = -1.2 * ((s_grid[:, None, None] / 5.5) ** 2)
    score_seed = np.where(seed_band, R_thin + 0.35 * M_smooth + prior_seed * 10.0, -1e9)
    idx_seed = np.argmax(score_seed, axis=0)
    s_star[snapped] = s_grid[idx_seed[snapped]]
    if snapped.all():
        s_star = gaussian_filter(s_star, sigma=1.1)
    else:
        num = gaussian_filter(s_star * snapped.astype(np.float32), sigma=1.1)
        den = np.maximum(gaussian_filter(snapped.astype(np.float32), sigma=1.1), 1e-5)
        s_star[snapped] = (num / den)[snapped]

    if not snapped.all():
        dist_map = distance_transform_edt(~snapped)
        max_d = int(np.ceil(dist_map.max()))
        for d_step in range(1, max_d + 1):
            frontier = (~snapped) & (dist_map <= float(d_step) + 0.45)
            if not np.any(frontier):
                continue
            fy, fx = np.where(frontier)
            for r, c in zip(fy, fx):
                r0, r1 = max(0, r - 2), min(rows, r + 3)
                c0, c1 = max(0, c - 2), min(cols, c + 3)
                sub_m = snapped[r0:r1, c0:c1]
                s_pred = float(np.mean(s_star[r0:r1, c0:c1][sub_m])) if np.any(sub_m) else 0.0
                prof_r = R_thin[:, r, c] + 0.25 * M_smooth[:, r, c]
                band = np.abs(s_grid - s_pred) <= 3.25
                if np.any(band) and float(M_smooth[band, r, c].max()) >= 25.0:
                    reg_sc = prof_r - 2.2 * ((s_grid - s_pred) ** 2)
                    reg_sc[~band] = -1e9
                    s_star[r, c] = float(s_grid[int(np.argmax(reg_sc))])
                else:
                    s_star[r, c] = s_pred
            snapped[frontier] = True

    for _ in range(8):
        s_nbr = gaussian_filter(s_star, sigma=1.25)
        diff = s_grid[:, None, None] - s_nbr[None, :, :]
        band = np.abs(diff) <= 2.5
        obj = (R_thin + 0.30 * M_smooth) - 3.5 * (diff ** 2)
        obj = np.where(band, obj, -1e9)
        best_i = np.argmax(obj, axis=0)
        s_peak = s_grid[best_i]
        peak_val = np.take_along_axis(M_smooth, best_i[None, :, :], axis=0)[0]
        alpha_m7 = np.clip((peak_val - 15.0) / 50.0, 0.0, 1.0)
        s_star = alpha_m7 * s_peak + (1.0 - alpha_m7) * s_nbr

    s_star = gaussian_filter(s_star, sigma=0.75).astype(np.float32)

    idx_zero = int(np.argmin(np.abs(s_grid - 0.0)))
    m7_before = M[idx_zero]
    idx_after = np.argmin(np.abs(s_grid[:, None, None] - s_star[None, :, :]), axis=0)
    m7_after = np.take_along_axis(M, idx_after[None, :, :], axis=0)[0]

    idx_m4 = np.argmin(np.abs(s_grid[:, None, None] - (s_star[None, :, :] - 4.0)), axis=0)
    idx_p4 = np.argmin(np.abs(s_grid[:, None, None] - (s_star[None, :, :] + 4.0)), axis=0)
    m7_off4 = 0.5 * (
        np.take_along_axis(M, idx_m4[None, :, :], axis=0)[0]
        + np.take_along_axis(M, idx_p4[None, :, :], axis=0)[0]
    )

    stats = {
        "s_star_min_vox": round(float(s_star.min()), 3),
        "s_star_max_vox": round(float(s_star.max()), 3),
        "s_star_std_all_vox": round(float(s_star.std()), 3),
        "m7_prob_mean_before": round(float(m7_before.mean()), 2),
        "m7_prob_mean_after": round(float(m7_after.mean()), 2),
        "m7_prob_at_s_star_pm4_vox": round(float(m7_off4.mean()), 2),
        "m7_ridge_attenuation_pm4_pct": round(float((m7_after.mean() - m7_off4.mean()) / max(1.0, float(m7_after.mean())) * 100.0), 1),
    }
    return s_star, stats


def build_interior_wraps_for_scroll(
    scroll_key: str,
    out_root: Path,
    z_frac: float = 0.52,
    win_r: int = 116,
    win_c: int = 116,
    top_k_wraps: int = 4,
) -> list[Path]:
    cfg = SCROLL_CONFIGS[scroll_key]
    volZ, volY, volX = cfg["shape_zyx"]
    ng_prefix = cfg["ng_prefix"]
    ux, uy = cfg["umbilicus_xy"]
    R_max = 0.5 * min(volY, volX)
    base_xy_url = f"https://vesuvius-challenge-open-data.s3.amazonaws.com/{ng_prefix}/xy"

    z0 = int(round(volZ * z_frac))
    paths0 = load_grid_paths(f"{base_xy_url}/{z0:06d}.grid", min_pts=win_c + 6)
    if not paths0:
        for dz in [-200, 200, -400, 400]:
            paths0 = load_grid_paths(f"{base_xy_url}/{z0 + dz:06d}.grid", min_pts=win_c + 4)
            if paths0:
                z0 = z0 + dz
                break

    interior_paths = []
    for p in paths0:
        if len(p) > win_c:
            s0 = (len(p) - win_c) // 2
            p_crop = p[s0:s0 + win_c].copy()
        else:
            p_crop = p.copy()
        r_mean = float(np.mean(np.hypot(p_crop[:, 0] - ux, p_crop[:, 1] - uy)))
        r_ratio = r_mean / R_max
        if 0.16 <= r_ratio <= 0.64:
            interior_paths.append((len(p), r_ratio, p_crop))

    if interior_paths:
        s_probe = np.array([-4.0, 0.0, +4.0], dtype=np.float32)
        scored_paths = []
        for length_p, r_rat, p_crop in interior_paths[:18]:
            diffs = np.gradient(p_crop, axis=0)
            norms = np.hypot(diffs[:, 0], diffs[:, 1]) + 1e-6
            nx_1d = (-diffs[:, 1] / norms).astype(np.float32)[None, :]
            ny_1d = (diffs[:, 0] / norms).astype(np.float32)[None, :]
            nz_1d = np.zeros_like(nx_1d)
            x_1d = p_crop[:, 0].astype(np.float32)[None, :]
            y_1d = p_crop[:, 1].astype(np.float32)[None, :]
            z_1d = np.full_like(x_1d, float(z0))
            M_1d = sample_m7_profiles_generic(x_1d, y_1d, z_1d, nx_1d, ny_1d, nz_1d, s_probe, scroll_key)
            m0 = float(M_1d[1].mean())
            moff = 0.5 * float(M_1d[0].mean() + M_1d[2].mean())
            atten = (m0 - moff) / max(m0, 1.0)
            thin_score = m0 * (max(0.0, atten) ** 1.3)
            scored_paths.append((thin_score, length_p, r_rat, p_crop))
        scored_paths.sort(key=lambda t: t[0], reverse=True)
    else:
        scored_paths = []

    seeds = []
    for thin_sc, _, r_rat, p in scored_paths:
        v0 = p[0] - np.array([ux, uy])
        v1 = p[-1] - np.array([ux, uy])
        if (v0[0] * v1[1] - v0[1] * v1[0]) < 0:
            p = p[::-1].copy()
        c_p = p.mean(axis=0)
        if all(np.linalg.norm(c_p - s.mean(axis=0)) > 85.0 for s in seeds):
            seeds.append(p)
        if len(seeds) >= top_k_wraps:
            break

    half_r = win_r // 2
    z_rows = [z0 + k * 20 for k in range(-half_r, win_r - half_r) if 40 <= z0 + k * 20 < volZ - 40]
    while len(z_rows) < win_r:
        z_rows.append(z_rows[-1] + 20)
    z_rows = z_rows[:win_r]
    z0_idx = int(np.argmin(np.abs(np.array(z_rows) - z0)))

    def _subdivide_1vox(pts_list):
        if not pts_list:
            return np.zeros((1, 2), dtype=np.float32)
        dense = []
        for p in pts_list:
            t = np.linspace(0.0, 1.0, 10, endpoint=False, dtype=np.float32)[:, None]
            segs = p[:-1, None, :] * (1.0 - t[None, :, :]) + p[1:, None, :] * t[None, :, :]
            dense.append(segs.reshape(-1, 2))
            dense.append(p[-1:])
        return np.vstack(dense)

    grid_trees = {}
    def _fetch(z_val):
        ps = load_grid_paths(f"{base_xy_url}/{z_val:06d}.grid", min_pts=12)
        all_p = _subdivide_1vox(ps)
        return z_val, (cKDTree(all_p), all_p)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for z_val, tup in ex.map(_fetch, z_rows):
            grid_trees[z_val] = tup

    saved_dirs = []
    for w_idx, seed in enumerate(seeds):
        U = len(seed)
        R = len(z_rows)
        xy_r = np.zeros((R, U, 2), dtype=np.float32)
        xy_r[z0_idx] = seed

        for direction in [1, -1]:
            idx = z0_idx
            while 0 <= idx + direction < R:
                prev = xy_r[idx]
                nxt_idx = idx + direction
                tree, pts = grid_trees[z_rows[nxt_idx]]
                diffs = np.gradient(prev, axis=0)
                norms = np.hypot(diffs[:, 0], diffs[:, 1])[:, None] + 1e-6
                tan_vec = diffs / norms
                nor_vec = np.stack([-tan_vec[:, 1], tan_vec[:, 0]], axis=-1)

                dists, indices = tree.query(prev, k=1)
                matched = pts[indices]
                valid = dists < 8.5
                shift_n = np.zeros(U, dtype=np.float32)
                if valid.any():
                    delta = matched[valid] - prev[valid]
                    shift_n[valid] = np.sum(delta * nor_vec[valid], axis=1)
                    if not valid.all():
                        good_u = np.where(valid)[0]
                        bad_u = np.where(~valid)[0]
                        shift_n[bad_u] = np.interp(bad_u, good_u, shift_n[good_u])
                    shift_n = gaussian_filter(shift_n, sigma=2.2, mode="nearest")
                xy_r[nxt_idx] = prev + shift_n[:, None] * nor_vec
                idx = nxt_idx

        x_grid = gaussian_filter(xy_r[:, :, 0], sigma=(0.8, 0.8), mode="nearest").astype(np.float32)
        y_grid = gaussian_filter(xy_r[:, :, 1], sigma=(0.8, 0.8), mode="nearest").astype(np.float32)
        z_grid = np.repeat(np.array(z_rows, dtype=np.float32)[:, None], U, axis=1)

        seg_name = f"thin_wrap{w_idx:02d}_z{z0:05d}"
        seg_dir = out_root / scroll_key / seg_name
        seg_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(seg_dir / "x.tif"), x_grid)
        tifffile.imwrite(str(seg_dir / "y.tif"), y_grid)
        tifffile.imwrite(str(seg_dir / "z.tif"), z_grid)
        saved_dirs.append(seg_dir)
    return saved_dirs


def crop_and_snap_window(
    scroll_key: str,
    pool: str,
    src_dir: Path,
    cand_id: str,
    r0: int,
    c0: int,
    win_r: int = 116,
    win_c: int = 116,
    max_s: float = 16.0,
    is_grid_wrap: bool = False,
) -> dict:
    x_full = tifffile.imread(str(src_dir / "x.tif")).astype(np.float32)
    y_full = tifffile.imread(str(src_dir / "y.tif")).astype(np.float32)
    z_full = tifffile.imread(str(src_dir / "z.tif")).astype(np.float32)

    if float(np.nanmean(z_full[-5:, :])) > float(np.nanmean(z_full[:5, :])):
        x_full = x_full[::-1, :].copy()
        y_full = y_full[::-1, :].copy()
        z_full = z_full[::-1, :].copy()

    H, W = x_full.shape
    r0 = max(0, min(r0, max(0, H - win_r)))
    c0 = max(0, min(c0, max(0, W - win_c)))
    x = x_full[r0:r0 + win_r, c0:c0 + win_c].copy()
    y = y_full[r0:r0 + win_r, c0:c0 + win_c].copy()
    z = z_full[r0:r0 + win_r, c0:c0 + win_c].copy()

    rows, cols = x.shape
    if is_grid_wrap:
        seed_mask = np.zeros((rows, cols), dtype=bool)
        rm = rows // 2
        seed_mask[max(0, rm - 6):min(rows, rm + 7), :] = True
    else:
        seed_mask = (x > 0) & np.isfinite(x)

    nx, ny, nz = compute_smooth_mesh_normals(x, y, z)

    ux, uy = SCROLL_CONFIGS[scroll_key]["umbilicus_xy"]
    rx = x - float(ux)
    ry = y - float(uy)
    r_dist = np.hypot(rx, ry) + 1e-6
    rx_hat = rx / r_dist
    ry_hat = ry / r_dist
    n_dot_rout = float(np.nanmean(nx * rx_hat + ny * ry_hat))
    inward_normal_sign = -1 if n_dot_rout > 0.0 else +1

    s_grid = np.linspace(-max_s, max_s, int(max_s * 4) + 1, dtype=np.float32)
    M = sample_m7_profiles_generic(x, y, z, nx, ny, nz, s_grid, scroll_key)
    s_star, stats = solve_thin_lamina_m7_snap(M, s_grid, seed_mask)

    x_snap = (x + s_star * nx).astype(np.float32)
    y_snap = (y + s_star * ny).astype(np.float32)
    z_snap = (z + s_star * nz).astype(np.float32)

    dst_dir = OUT_MESH_ROOT / scroll_key / cand_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(dst_dir / "x.tif"), x_snap)
    tifffile.imwrite(str(dst_dir / "y.tif"), y_snap)
    tifffile.imwrite(str(dst_dir / "z.tif"), z_snap)

    Z_max, Y_max, X_max = SCROLL_CONFIGS[scroll_key]["shape_zyx"]
    R_max = 0.5 * min(Y_max, X_max)
    z_frac_min = float(z_snap.min()) / float(Z_max)
    z_frac_max = float(z_snap.max()) / float(Z_max)
    r_umb_frac = float(np.mean(r_dist)) / float(R_max)

    meta = {
        "uuid": cand_id,
        "name": cand_id,
        "scroll_key": scroll_key,
        "pool": pool,
        "format": "tifxyz",
        "scale": [0.05, 0.05],
        "shape_grid": [int(rows), int(cols)],
        "shape_px": [int(rows * 20), int(cols * 20)],
        "um": SCROLL_CONFIGS[scroll_key]["um"],
        "n_dot_rout": round(n_dot_rout, 4),
        "inward_normal_sign": int(inward_normal_sign),
        "z_range_vox": [round(float(z_snap.min()), 1), round(float(z_snap.max()), 1)],
        "z_frac_range": [round(z_frac_min, 3), round(z_frac_max, 3)],
        "r_umb_frac": round(r_umb_frac, 3),
        "stage_i_m7_snap": stats,
    }
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"  [{pool:26s}] {scroll_key}/{cand_id:30s} | z={z_frac_min*100:4.1f}..{z_frac_max*100:4.1f}% "
        f"| r={r_umb_frac*100:4.1f}%R | inward={inward_normal_sign:+d} "
        f"| m7: {stats['m7_prob_mean_before']:5.1f}->{stats['m7_prob_mean_after']:5.1f} "
        f"| atten_pm4={stats['m7_ridge_attenuation_pm4_pct']:4.1f}%",
        flush=True,
    )
    return meta


def main():
    ph1203_raw = BASE_DIR / "meshes" / "PHerc1203"
    ph1203_up = BASE_DIR / "meshes" / "PHerc1203_upright"
    extra_ag = [
        "auto_grown_20250923164241632",
        "auto_grown_20250923164713356",
        "auto_grown_20250925223153537",
        "auto_grown_20251005230830030",
        "auto_grown_20251005231446963",
        "auto_grown_20250923163217042",
        "auto_grown_20250923164615417",
        "auto_grown_20250923163219195",
        "auto_grown_20250923163223390",
        "auto_grown_20250923163227956",
        "auto_grown_20250923163230146",
        "auto_grown_20250923164608661",
    ]
    for t in extra_ag:
        dst_u = ph1203_up / f"{t}_upright"
        if not (dst_u / "x.tif").exists():
            normalize_tifxyz(ph1203_raw / t, dst_u)

    int_grids_root = BASE_DIR / "meshes" / "stage_i_int_grids"
    int_grids_root.mkdir(parents=True, exist_ok=True)

    for sc_77, zf in [("PHerc1203_77keV", 0.50), ("PHerc0846A_77keV", 0.50)]:
        if len(list((int_grids_root / sc_77).glob("thin_wrap*"))) < 4:
            print(f"=== Building thin-screened 20-vox interior wraps for {sc_77} ===", flush=True)
            build_interior_wraps_for_scroll(sc_77, int_grids_root, z_frac=zf, top_k_wraps=4)

    for sc_c in [
        "PHerc0125", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
        "PHerc0358", "PHerc0813", "PHerc0826", "PHerc1218", "PHerc1545",
    ]:
        if len(list((int_grids_root / sc_c).glob("thin_wrap*"))) < 2:
            print(f"=== Building thin-screened 20-vox interior wraps for {sc_c} ===", flush=True)
            build_interior_wraps_for_scroll(sc_c, int_grids_root, z_frac=0.54, top_k_wraps=2)

    candidates_spec = []

    # Pool A: 8 candidates (4 PHerc1203_77keV + 4 PHerc0846A_77keV)
    for sc_77 in ["PHerc1203_77keV", "PHerc0846A_77keV"]:
        for w_dir in sorted((int_grids_root / sc_77).glob("thin_wrap*"))[:4]:
            candidates_spec.append((sc_77, "Pool_A_77keV_L2", w_dir, f"{w_dir.name}_77keV_m7", 0, 0, 116, 116, 18.0, True))

    # Pool B: 26 true 3D `auto_grown` windows across PHerc1203 (22 windows) & PHerc0800 (4 windows)
    b_ph1203 = [
        ("auto_grown_20250925223153537_upright", "ag_target_patch_77keVband_m7", 0, 0),
        ("auto_grown_20250923164241632_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923164713356_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20251005230830030_upright", "ag_target_patch_m7", 8, 4),
        ("auto_grown_20251005231446963_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923163217042_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923164615417_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923163219195_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923163223390_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923163227956_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923163230146_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250923164608661_upright", "ag_target_patch_m7", 0, 0),
        ("auto_grown_20250930000321811_upright", "ag_target_patch_m7", 12, 12),
        ("auto_grown_20251005230118636_upright", "ag_target_patch_m7", 20, 24),
        ("auto_grown_20251005230830031_upright", "ag_target_patch_top_m7", 4, 24),
        ("auto_grown_20251005230830031_upright", "ag_target_patch_bot_m7", 48, 24),
        ("auto_grown_20251005230830031_upright", "ag_target_patch_right_m7", 26, 50),
        ("auto_grown_20251005PANORAMA_TARGET_upright", "ag_PANORAMA_TARGET_left_m7", 24, 4),
        ("auto_grown_20251005PANORAMA_TARGET_upright", "ag_PANORAMA_TARGET_right_m7", 24, 48),
        ("auto_grown_20251005221856743_upright", "ag_target_patch_recto_m7", 0, 48),
        ("auto_grown_20250929222256117_upright", "ag_target_patch_recto_m7", 20, 32),
        ("auto_grown_20250930104534929_upright", "ag_target_patch_recto_m7", 30, 6),
    ]
    for src_name, cid, r0, c0 in b_ph1203:
        s_dir = ph1203_up / src_name
        x_tmp = tifffile.imread(str(s_dir / "x.tif"))
        wr = min(116, x_tmp.shape[0])
        wc = min(116, x_tmp.shape[1])
        candidates_spec.append(("PHerc1203", "Pool_B_AutoGrown", s_dir, cid, r0, c0, wr, wc, 12.0, False))

    ext_root = BASE_DIR / "meshes" / "PHerc0800_ext"
    candidates_spec.append(("PHerc0800", "Pool_B_AutoGrown", ext_root / "20251028220042_ext", "ag_target_patch_ext_m7", 6, 6, 116, 116, 16.0, False))
    candidates_spec.append(("PHerc0800", "Pool_B_AutoGrown", ext_root / "20251028225813_ext", "ag_target_patch_ext_m7", 6, 6, 116, 116, 16.0, False))
    candidates_spec.append(("PHerc0800", "Pool_B_AutoGrown", ext_root / "20251028222030_ext", "ag_target_patch_recto_m7", 8, 8, 116, 116, 16.0, False))
    candidates_spec.append(("PHerc0800", "Pool_B_AutoGrown", ext_root / "20251028220955_ext", "ag_target_patch_recto_m7", 6, 6, 116, 116, 16.0, False))

    # Pool C: 18 thin-screened candidates across the 10 unexplored scrolls with lasagna + m7-L0
    for sc_c in [
        "PHerc0125", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
        "PHerc0358", "PHerc0813", "PHerc0826", "PHerc1218", "PHerc1545",
    ]:
        for w_dir in sorted((int_grids_root / sc_c).glob("thin_wrap*"))[:2]:
            candidates_spec.append((sc_c, "Pool_C_MultiScroll_LasagnaM7", w_dir, f"{w_dir.name}_m7", 0, 0, 116, 116, 18.0, True))

    print(f"\n=== Snapping {len(candidates_spec)} Stage I v3 Candidate Windows in 3D to S3 m7-Zarr ===", flush=True)
    all_metas = []
    for spec in candidates_spec:
        sc_key, pool, src_dir, cid, r0, c0, wr, wc, ms, is_gw = spec
        dst_meta = OUT_MESH_ROOT / sc_key / cid / "meta.json"
        if dst_meta.exists() and not cid.startswith("thin_wrap"):
            meta = json.loads(dst_meta.read_text())
            stats = meta["stage_i_m7_snap"]
            print(
                f"  [CACHED {pool:19s}] {sc_key}/{cid:30s} | m7: {stats['m7_prob_mean_before']:5.1f}->{stats['m7_prob_mean_after']:5.1f} "
                f"| atten_pm4={stats['m7_ridge_attenuation_pm4_pct']:4.1f}%",
                flush=True,
            )
        else:
            meta = crop_and_snap_window(sc_key, pool, src_dir, cid, r0, c0, wr, wc, ms, is_grid_wrap=is_gw)
        all_metas.append(meta)

    passed_cpu = [
        m for m in all_metas
        if m["stage_i_m7_snap"]["m7_prob_mean_after"] >= 120.0
        and m["stage_i_m7_snap"]["m7_ridge_attenuation_pm4_pct"] >= 62.0
    ]
    summary = {
        "total_candidates_snapped": len(all_metas),
        "passed_bet5_m7_gate_count": len(passed_cpu),
        "bet5_status": "PASS" if len(passed_cpu) >= 28 else "FAIL",
        "candidates": all_metas,
    }
    out_path = BASE_DIR / "meshes" / "stage_i_scout_manifest.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"\n=== Stage I Etap I.1 v3 Complete: {len(passed_cpu)}/{len(all_metas)} candidates passed Bet 5 m7-Snap gate "
        f"-> Bet 5 = {summary['bet5_status']} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
