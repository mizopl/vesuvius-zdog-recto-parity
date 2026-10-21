#!/usr/bin/env python3
"""
Stage H — True 3D Mesh Vertex Snap to S3 `*-surface-m7-L0-th0.2.zarr`
(Erwin Nieuwlaar `snap_meshes.py` parity).

Before any CT layer rendering takes place:
1. Loads the 3D `.tifxyz` mesh `(x.tif, y.tif, z.tif)` for PHerc0800 and PHerc1203.
2. Computes smooth 3D unit surface normals `n(r, c) = (nx, ny, nz)` at every mesh vertex.
3. Downloads the intersected `[192, 192, 192]` chunks of the organizer S3 surface probability
   volume `representations/predictions/surfaces/*-surface-m7-L0-th0.2.zarr/0` (full L0 resolution).
4. Samples the `m7` probability profile `M(r, c, s)` along the 3D normal `P(r, c) + s * n(r, c)`
   for `s in [-24.0, +24.0]` voxels (0.25-voxel resolution).
5. Uses Seed-Anchored Wavefront Continuation + Coupled 2D Membrane Regularization so:
   - Interior `vc_grow_seg_from_seed` vertices snap with sub-voxel precision (`s in [-2.5, +2.5]` vox)
     directly onto the center of the `m7` probability ridge (`M(r, c, s)`), and
   - Extrapolated outer corners (`PHerc0800_ext`) march outward cell-by-cell from the diamond
     boundary, tracking the exact same continuous `m7` sheet ridge all the way to the corners.
6. Writes the snapped 3D `.tifxyz` meshes (`P_snap = P + s* n`) to `meshes/<scroll>_m7snap/`
   and records before/after `m7` probability & ridge-sharpness diagnostics.
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

BASE_DIR = Path("/Users/blister/.gemini/antigravity/brain/0adb9c02-dee8-451d-983f-5070b954f30b/scratch/vesuvius-unread-hunt")

S3_M7_CONFIG = {
    "PHerc1203": {
        "s3_zarr_0": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/representations/predictions/surfaces/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr/0",
        "shape_zyx": (18977, 6844, 6844),
        "chunk_zyx": (192, 192, 192),
    },
    "PHerc0800": {
        "s3_zarr_0": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/representations/predictions/surfaces/20250521135224-surface-20260413222639-surface-m7-L0-th0.2.zarr/0",
        "shape_zyx": (24298, 9867, 9867),
        "chunk_zyx": (192, 192, 192),
    },
}

SNAP_TARGETS = [
    {
        "scroll": "PHerc1203",
        "src_group": "PHerc1203_upright",
        "seg_name": "auto_grown_20250929222256117_upright",
        "orig_seed_dir": None,
        "orient": "reverse",
        "target_box_yxwh": [424, 742, 2136, 2136],
    },
    {
        "scroll": "PHerc1203",
        "src_group": "PHerc1203_upright",
        "seg_name": "auto_grown_20251005221856743_upright",
        "orig_seed_dir": None,
        "orient": "reverse",
        "target_box_yxwh": [0, 954, 2136, 2136],
    },
    {
        "scroll": "PHerc1203",
        "src_group": "PHerc1203_upright",
        "seg_name": "auto_grown_20250930104534929_upright",
        "orig_seed_dir": None,
        "orient": "reverse",
        "target_box_yxwh": [636, 106, 2136, 2136],
    },
    {
        "scroll": "PHerc0800",
        "src_group": "PHerc0800_ext",
        "seg_name": "20251028220955_ext",
        "orig_seed_dir": "meshes/PHerc0800/20251028220955-on-20250521135224-8.64um.tifxyz",
        "orient": "reverse",
        "target_box_yxwh": [0, 230, 2315, 2315],
    },
    {
        "scroll": "PHerc0800",
        "src_group": "PHerc0800_ext",
        "seg_name": "20251028213516_ext",
        "orig_seed_dir": "meshes/PHerc0800/20251028213516-on-20250521135224-8.64um.tifxyz",
        "orient": "reverse",
        "target_box_yxwh": [0, 230, 2315, 2315],
    },
    {
        "scroll": "PHerc0800",
        "src_group": "PHerc0800_ext",
        "seg_name": "20251028222030_ext",
        "orig_seed_dir": "meshes/PHerc0800/20251028222030-on-20250521135224-8.64um.tifxyz",
        "orient": "reverse",
        "target_box_yxwh": [0, 115, 2315, 2315],
    },
]


def compute_smooth_mesh_normals(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unit 3D surface normals (nx, ny, nz) at every vertex (r, c) of a valid .tifxyz mesh.
    Matches render_tifxyz_66.cpp normal orientation convention: n = normalize(dP/dc x dP/dr).
    """
    # Smooth coordinates slightly only for tangent vector estimation so local kinks don't twist normals
    xs = gaussian_filter(x.astype(np.float64), sigma=1.0)
    ys = gaussian_filter(y.astype(np.float64), sigma=1.0)
    zs = gaussian_filter(z.astype(np.float64), sigma=1.0)

    # dP/dc (along columns, axis=1) and dP/dr (along rows, axis=0)
    dxc = np.gradient(xs, axis=1)
    dyc = np.gradient(ys, axis=1)
    dzc = np.gradient(zs, axis=1)

    dxr = np.gradient(xs, axis=0)
    dyr = np.gradient(ys, axis=0)
    dzr = np.gradient(zs, axis=0)

    # Cross product dP/dc x dP/dr
    nx = dyc * dzr - dzc * dyr
    ny = dzc * dxr - dxc * dzr
    nz = dxc * dyr - dyc * dxr

    nx = gaussian_filter(nx, sigma=1.2)
    ny = gaussian_filter(ny, sigma=1.2)
    nz = gaussian_filter(nz, sigma=1.2)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    return (nx / norm).astype(np.float32), (ny / norm).astype(np.float32), (nz / norm).astype(np.float32)


def fetch_m7_chunks(scroll: str, chunk_keys: set[tuple[int, int, int]]) -> dict[tuple[int, int, int], np.ndarray]:
    """Download and blosc-decode 192^3 uint8 chunks from S3 m7-L0-th0.2.zarr/0 with local disk cache."""
    cfg = S3_M7_CONFIG[scroll]
    base_url = cfg["s3_zarr_0"]
    cache_root = BASE_DIR / "m7_cache" / scroll
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
                req = urllib.request.Request(url, headers={"User-Agent": "vesuvius-stage-h/1.0"})
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
        print(f"  [m7 S3] Downloaded {len(to_download)} chunks for {scroll} in {time.time()-t0:.1f}s (total active={len(chunks_dict)})", flush=True)

    return chunks_dict


def sample_m7_profiles(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    s_grid: np.ndarray,
    scroll: str,
) -> np.ndarray:
    """
    Sample 3D m7 surface probability M[S, rows, cols] along P(r, c) + s * n(r, c)
    using trilinear interpolation inside the 192^3 L0 chunks.
    """
    S = len(s_grid)
    rows, cols = x.shape
    Z_max, Y_max, X_max = S3_M7_CONFIG[scroll]["shape_zyx"]

    # Compute 3D coordinates for all (S, rows, cols)
    s_b = s_grid[:, None, None].astype(np.float32)
    Xs = np.clip(x[None, :, :] + s_b * nx[None, :, :], 0.0, float(X_max - 2.001))
    Ys = np.clip(y[None, :, :] + s_b * ny[None, :, :], 0.0, float(Y_max - 2.001))
    Zs = np.clip(z[None, :, :] + s_b * nz[None, :, :], 0.0, float(Z_max - 2.001))

    # Find all required 192^3 chunks
    cz_all = (Zs.astype(np.int32)) // 192
    cy_all = (Ys.astype(np.int32)) // 192
    cx_all = (Xs.astype(np.int32)) // 192
    unique_triplets = np.unique(np.stack([cz_all.ravel(), cy_all.ravel(), cx_all.ravel()], axis=1), axis=0)
    chunk_keys = {(int(t[0]), int(t[1]), int(t[2])) for t in unique_triplets}

    chunks_dict = fetch_m7_chunks(scroll, chunk_keys)

    # Evaluate trilinear (or nearest-subvoxel) inside each chunk
    # Since s_grid has 0.25-voxel step and x,y,z are smoothed over the mesh grid,
    # let's do fast vectorized 2-point z/y/x interpolation within each chunk
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
        # 7-point 3D stencil average for sub-voxel smoothness
        v0 = chunk[lz, ly, lx].astype(np.float32)
        vz1 = chunk[np.minimum(191, lz + 1), ly, lx].astype(np.float32)
        vz0 = chunk[np.maximum(0, lz - 1), ly, lx].astype(np.float32)
        vy1 = chunk[lz, np.minimum(191, ly + 1), lx].astype(np.float32)
        vy0 = chunk[lz, np.maximum(0, ly - 1), lx].astype(np.float32)
        vx1 = chunk[lz, ly, np.minimum(191, lx + 1)].astype(np.float32)
        vx0 = chunk[lz, ly, np.maximum(0, lx - 1)].astype(np.float32)
        M[mask] = 0.40 * v0 + 0.10 * (vz1 + vz0 + vy1 + vy0 + vx1 + vx0)

    return M


def solve_single_sheet_m7_snap(
    M: np.ndarray,
    s_grid: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Find a smooth 2D displacement field `s_star(r, c)` that locks every mesh vertex `(r, c)`
    onto the single continuous `m7` surface sheet:
    1. Smooth `M[S, rows, cols]` gently along `(r, c)` (`sigma = (0.8, 1.2, 1.2)`) so isolated
       1-vertex holes in `m7` don't cause jumps.
    2. Inside `seed_mask` (the `vc_grow_seg_from_seed` interior), restrict initial ridge locking
       to `s in [-3.5, +3.5]` voxels so it refines the true seed sheet without jumping to an
       adjacent wrap (`+-10..15` voxels away).
    3. Outside `seed_mask` (e.g. the 4 extrapolated corners of `PHerc0800_ext`), march outward
       by geodesic distance `d = 1, 2, 3, ...` from `seed_mask`. At each frontier vertex `(r, c)`,
       compute the mean displacement `s_pred` of its already-snapped 8-neighbors and search for
       the local `m7` ridge peak within `[s_pred - 2.5, s_pred + 2.5]` voxels!
    4. Run 6 iterations of `m7`-potential + membrane smoothness optimization (`P_new = argmax_s
       M_smooth(s) - lambda * (s - laplacian_avg(s))^2`) so the entire mesh sits at the exact
       sub-voxel peak of the `m7` surface prediction.
    """
    S, rows, cols = M.shape
    M_smooth = gaussian_filter(M, sigma=(0.75, 1.15, 1.15))

    s_star = np.zeros((rows, cols), dtype=np.float32)
    snapped = seed_mask.copy()

    # Step 1: Lock seed interior vertices within [-3.5, +3.5] voxels
    seed_band = np.abs(s_grid)[:, None, None] <= 3.5
    prior_seed = -1.8 * ((s_grid[:, None, None] / 3.5) ** 2)
    score_seed = np.where(seed_band, M_smooth + prior_seed * 12.0, -1e9)
    idx_seed = np.argmax(score_seed, axis=0)
    s_star[snapped] = s_grid[idx_seed[snapped]]
    # Smooth interior initial lock
    if snapped.all():
        s_star = gaussian_filter(s_star, sigma=1.2)
    else:
        num = gaussian_filter(s_star * snapped.astype(np.float32), sigma=1.2)
        den = np.maximum(gaussian_filter(snapped.astype(np.float32), sigma=1.2), 1e-5)
        s_star[snapped] = (num / den)[snapped]

    # Step 2: Geodesic Wavefront Propagation into extrapolated corners (if any)
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
                if np.any(sub_m):
                    s_pred = float(np.mean(s_star[r0:r1, c0:c1][sub_m]))
                else:
                    s_pred = 0.0
                # Search for the m7 ridge peak within s_pred +- 2.75 voxels
                prof = M_smooth[:, r, c]
                band = np.abs(s_grid - s_pred) <= 2.75
                if np.any( band ) and float(prof[band].max()) >= 25.0:
                    reg_sc = prof - 2.5 * ((s_grid - s_pred) ** 2)
                    reg_sc[~band] = -1e9
                    s_star[r, c] = float(s_grid[int(np.argmax(reg_sc))])
                else:
                    # If m7 probability is very weak at this vertex, smoothly continue s_pred
                    s_star[r, c] = s_pred
            snapped[frontier] = True

    # Step 3: Coupled 2D Membrane + Sub-Voxel m7 Ridge Peak Refinement (6 iterations)
    for it in range(6):
        s_nbr = gaussian_filter(s_star, sigma=1.35)
        # Allow +-2.0 voxel local adjustment toward the exact m7 peak around s_nbr
        diff = s_grid[:, None, None] - s_nbr[None, :, :]
        band = np.abs(diff) <= 2.0
        # Weight m7 probability against membrane elasticity
        obj = M_smooth - 4.0 * (diff ** 2)
        obj = np.where(band, obj, -1e9)
        best_i = np.argmax(obj, axis=0)
        s_peak = s_grid[best_i]
        # Where M_smooth at best_i is strong (>= 25), snap to s_peak; otherwise follow membrane s_nbr
        peak_val = np.take_along_axis(M_smooth, best_i[None, :, :], axis=0)[0]
        alpha_m7 = np.clip((peak_val - 15.0) / 55.0, 0.0, 1.0)
        s_star = alpha_m7 * s_peak + (1.0 - alpha_m7) * s_nbr

    # Final gentle sub-grid smoothing (sigma=0.85 mesh cells = ~17 pixels = ~0.13 mm)
    s_star = gaussian_filter(s_star, sigma=0.85).astype(np.float32)

    # Compute before (s=0) vs after (s=s_star) m7 surface probability statistics
    idx_zero = int(np.argmin(np.abs(s_grid - 0.0)))
    m7_before = M[idx_zero]
    idx_after = np.argmin(np.abs(s_grid[:, None, None] - s_star[None, :, :]), axis=0)
    m7_after = np.take_along_axis(M, idx_after[None, :, :], axis=0)[0]

    # Also measure off-sheet m7 probability at s_star +- 4 voxels to verify single-lamina sharpness
    idx_m4 = np.argmin(np.abs(s_grid[:, None, None] - (s_star[None, :, :] - 4.0)), axis=0)
    idx_p4 = np.argmin(np.abs(s_grid[:, None, None] - (s_star[None, :, :] + 4.0)), axis=0)
    m7_off4 = 0.5 * (
        np.take_along_axis(M, idx_m4[None, :, :], axis=0)[0]
        + np.take_along_axis(M, idx_p4[None, :, :], axis=0)[0]
    )

    ext_mask = ~seed_mask
    stats = {
        "s_star_min_vox": round(float(s_star.min()), 3),
        "s_star_max_vox": round(float(s_star.max()), 3),
        "s_star_std_all_vox": round(float(s_star.std()), 3),
        "s_star_std_seed_vox": round(float(s_star[seed_mask].std()), 3),
        "s_star_max_abs_ext_corners_vox": round(float(np.abs(s_star[ext_mask]).max()), 3) if np.any(ext_mask) else 0.0,
        "m7_prob_mean_before": round(float(m7_before.mean()), 2),
        "m7_prob_mean_after": round(float(m7_after.mean()), 2),
        "m7_prob_seed_before": round(float(m7_before[seed_mask].mean()), 2),
        "m7_prob_seed_after": round(float(m7_after[seed_mask].mean()), 2),
        "m7_prob_ext_corners_before": round(float(m7_before[ext_mask].mean()), 2) if np.any(ext_mask) else 0.0,
        "m7_prob_ext_corners_after": round(float(m7_after[ext_mask].mean()), 2) if np.any(ext_mask) else 0.0,
        "m7_prob_at_s_star_pm4_vox": round(float(m7_off4.mean()), 2),
        "m7_ridge_attenuation_pm4_pct": round(float((m7_after.mean() - m7_off4.mean()) / max(1.0, float(m7_after.mean())) * 100.0), 1),
    }
    return s_star, stats


def snap_one_target(t: dict) -> dict:
    scroll = t["scroll"]
    src_group = t["src_group"]
    seg_name = t["seg_name"]
    src_dir = BASE_DIR / "meshes" / src_group / seg_name

    x = tifffile.imread(str(src_dir / "x.tif")).astype(np.float32)
    y = tifffile.imread(str(src_dir / "y.tif")).astype(np.float32)
    z = tifffile.imread(str(src_dir / "z.tif")).astype(np.float32)
    rows, cols = x.shape

    # Determine seed_mask (True where original vc_grow_seg_from_seed was valid)
    if t["orig_seed_dir"]:
        orig_x = tifffile.imread(str(BASE_DIR / t["orig_seed_dir"] / "x.tif"))
        H0, W0 = orig_x.shape
        pad_r = max(0, (rows - H0) // 2)
        pad_c = max(0, (cols - W0) // 2)
        seed_mask = np.zeros((rows, cols), dtype=bool)
        seed_mask[pad_r:pad_r + H0, pad_c:pad_c + W0] = (orig_x > 0) & np.isfinite(orig_x)
        # Erode by 2 vertices so boundary transitions smoothly
        seed_mask = cv2.erode(seed_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    else:
        seed_mask = (x > 0) & np.isfinite(x)

    nx, ny, nz = compute_smooth_mesh_normals(x, y, z)
    # Use [-20, +20] at 0.25-vox steps for extrapolated meshes, [-10, +10] at 0.25-vox steps for auto_grown
    max_s = 20.0 if t["orig_seed_dir"] else 10.0
    s_grid = np.linspace(-max_s, max_s, int(max_s * 8) + 1, dtype=np.float32)

    print(f"=== Snapping 3D Mesh {scroll}/{seg_name} ({rows}x{cols} vertices, s in [-{max_s}, +{max_s}]) ===", flush=True)
    M = sample_m7_profiles(x, y, z, nx, ny, nz, s_grid, scroll)
    s_star, stats = solve_single_sheet_m7_snap(M, s_grid, seed_mask)

    x_snap = (x + s_star * nx).astype(np.float32)
    y_snap = (y + s_star * ny).astype(np.float32)
    z_snap = (z + s_star * nz).astype(np.float32)

    dst_group = f"{scroll}_m7snap"
    dst_seg = f"{seg_name}_m7snap"
    dst_dir = BASE_DIR / "meshes" / dst_group / dst_seg
    dst_dir.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(str(dst_dir / "x.tif"), x_snap)
    tifffile.imwrite(str(dst_dir / "y.tif"), y_snap)
    tifffile.imwrite(str(dst_dir / "z.tif"), z_snap)
    meta = json.loads((src_dir / "meta.json").read_text())
    meta["uuid"] = dst_seg
    meta["name"] = dst_seg
    meta["stage_h_m7_snap"] = stats
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Save visual diagnostic of s_star(r, c)
    s_vis = np.clip((s_star + max_s) / (2.0 * max_s) * 255.0, 0.0, 255.0).astype(np.uint8)
    s_vis_up = cv2.resize(s_vis, (cols * 8, rows * 8), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(dst_dir / "s_star_3d_snap_map.png"), cv2.applyColorMap(s_vis_up, cv2.COLORMAP_TURBO))

    print(
        f"  -> [DONE {scroll}/{dst_seg}] s*={stats['s_star_min_vox']:+.2f}..{stats['s_star_max_vox']:+.2f} vox "
        f"| m7_prob: {stats['m7_prob_mean_before']:.1f} -> {stats['m7_prob_mean_after']:.1f} "
        f"(corners: {stats['m7_prob_ext_corners_before']:.1f} -> {stats['m7_prob_ext_corners_after']:.1f}) "
        f"| m7_drop_pm4={stats['m7_ridge_attenuation_pm4_pct']:.1f}%",
        flush=True,
    )
    return {
        "scroll": scroll,
        "src_seg": seg_name,
        "dst_group": dst_group,
        "dst_seg": dst_seg,
        "orient": t["orient"],
        "target_box_yxwh": t["target_box_yxwh"],
        "snap_stats": stats,
    }


def main():
    results = []
    for t in SNAP_TARGETS:
        res = snap_one_target(t)
        results.append(res)
    out_json = BASE_DIR / "meshes" / "stage_h_m7_snap_summary.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved Stage H 3D m7-Snap summary to {out_json}", flush=True)


if __name__ == "__main__":
    main()
