#!/usr/bin/env python3
"""
Stage I (Iteration 4) — CLSS Skeptic Gate Audit, Geodesic Single-Fiber Stroke-Ribbon
Reconstructor (Directional Reed-Bundle Gap Bridging + Dominant-Component Topology Gate),
Swirl-Delamination Suppressor, and Calibration Bets 5-8 Evaluator.
"""
import json
import re
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))

from stage_f_sheet_tracker_and_paleography import (
    TEMPLATES_64,
    GREEK_DISPLAY_UNICODE,
    classify_greek_letter_patch,
    load_gray,
    make_calligraphic_gold,
    render_paleography_dossier,
)
from stage_g_clss_audit_and_dossier import HERCULANEUM_STEMS

STAGE_I_DIR = BASE / "results_stage_i"
STAGE_I_DOSSIER_DIR = BASE / "results_stage_i_dossiers"
STAGE_I_DOSSIER_DIR.mkdir(parents=True, exist_ok=True)


def compute_strict_single_lamina_ct_metrics(ct_2cm: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Computes a strict pixel-wise Single-Lamina Flatness & Fiber Integrity map `flat_2cm` in [0, 1]
    plus summary metrics (`valid_fiber_coverage`, `single_sheet_flatness_pct`, `swirl_delam_frac`):
      1. Rejects black out-of-volume regions (`ct <= 15`) and constant-gray triangle fills (`std_loc < 4.5`).
      2. Suppresses swirling delaminated wrap zones (`swirl_delam_frac > 0.36`, e.g. `PHerc1203/ag_target_patch`
         where `swirl_delam_frac = 0.59`), while preserving natural orthogonal papyrus reed interstices
         on flat single sheets (`swirl_delam_frac = 0.10..0.33`).
    """
    ct_f = ct_2cm.astype(np.float32)
    in_range = (ct_f > 15.0) & (ct_f < 248.0)

    # 1. Micro-fiber texture presence (window sigma=6 px ~ 55 um)
    mean_loc = gaussian_filter(ct_f, sigma=6.0)
    sq_loc = gaussian_filter(ct_f * ct_f, sigma=6.0)
    std_loc = np.sqrt(np.maximum(0.0, sq_loc - mean_loc * mean_loc))
    valid_fiber_mask = in_range & (std_loc > 4.5)
    valid_fiber_coverage = float(np.mean(valid_fiber_mask))

    fiber_conf = np.clip((std_loc - 4.0) / 7.5, 0.0, 1.0) * in_range.astype(np.float32)
    fiber_conf = gaussian_filter(fiber_conf, sigma=12.0)

    # 2. Dark air-crack / inter-wrap delamination gap suppression (ct < 42 = true air gap between wraps)
    is_gap = (ct_f < 42.0).astype(np.float32)
    gap_frac = gaussian_filter(is_gap, sigma=30.0)
    no_gap_score = np.clip(1.0 - np.maximum(0.0, gap_frac - 0.05) * 5.0, 0.0, 1.0)

    # 3. High-contrast swirling wrap band suppression (distinguishes multi-wrap delamination swirls
    #    like PHerc1203/ag_target_patch from normal papyrus reed relief on flat sheets)
    mean_swirl = gaussian_filter(ct_f, sigma=24.0)
    sq_swirl = gaussian_filter(ct_f * ct_f, sigma=24.0)
    std_swirl = np.sqrt(np.maximum(0.0, sq_swirl - mean_swirl * mean_swirl))
    swirl_penalty = np.clip(1.0 - np.clip((std_swirl - 30.0) / 15.0, 0.0, 1.0) * 0.85, 0.08, 1.0)
    swirl_delam_frac = float(np.mean((std_swirl > 34.5) | (gap_frac > 0.16)))

    flat_2cm = np.clip(fiber_conf * no_gap_score * swirl_penalty, 0.0, 1.0)
    return flat_2cm, {
        "valid_fiber_coverage": round(valid_fiber_coverage, 4),
        "single_sheet_flatness_pct": round(float(np.mean(flat_2cm) * 100.0), 1),
        "swirl_delam_frac": round(swirl_delam_frac, 4),
        "micro_fiber_std_median": round(float(np.median(std_loc[in_range])) if np.any(in_range) else 0.0, 2),
    }


def coherence_enhancing_stroke_diffusion(img_u8: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """
    Applies Structure-Tensor Coherence-Enhancing Anisotropic Diffusion along the local stroke
    tangent direction to link calligraphic strokes across papyrus reed-bundle undulations.
    """
    u = img_u8.astype(np.float32)
    for _ in range(n_iter):
        us = gaussian_filter(u, sigma=2.2)
        gx = cv2.Sobel(us, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(us, cv2.CV_32F, 0, 1, ksize=3)

        jxx = gaussian_filter(gx * gx, sigma=9.0)
        jxy = gaussian_filter(gx * gy, sigma=9.0)
        jyy = gaussian_filter(gy * gy, sigma=9.0)

        theta_g = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-6)
        tx = -np.sin(theta_g)
        ty = np.cos(theta_g)

        tr = jxx + jyy + 1e-4
        aniso = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / tr
        coh = np.clip(aniso, 0.0, 1.0)

        uxx = cv2.Sobel(u, cv2.CV_32F, 2, 0, ksize=3) / 4.0
        uyy = cv2.Sobel(u, cv2.CV_32F, 0, 2, ksize=3) / 4.0
        uxy = cv2.Sobel(u, cv2.CV_32F, 1, 1, ksize=3) / 4.0
        d2_tan = (tx * tx) * uxx + 2.0 * (tx * ty) * uxy + (ty * ty) * uyy

        u = np.clip(u + 0.34 * (0.25 + 0.75 * coh) * d2_tan, 0.0, 255.0)

    return u.astype(np.uint8)


def multiscale_hessian_stroke_ridge(img_u8: np.ndarray, sigmas: tuple[float, ...] = (6.0, 9.5, 13.5)) -> np.ndarray:
    img_f = img_u8.astype(np.float32)
    ridge_max = np.zeros_like(img_f)

    for sig in sigmas:
        sm = gaussian_filter(img_f, sigma=sig)
        s2 = sig * sig
        dxx = gaussian_filter(sm, sigma=1.0, order=(0, 2)) * s2
        dyy = gaussian_filter(sm, sigma=1.0, order=(2, 0)) * s2
        dxy = gaussian_filter(sm, sigma=1.0, order=(1, 1)) * s2

        tr = dxx + dyy
        disc = np.sqrt(np.maximum(0.0, (dxx - dyy) ** 2 + 4.0 * dxy * dxy))
        eig1 = 0.5 * (tr + disc)
        eig2 = 0.5 * (tr - disc)

        swap = np.abs(eig1) > np.abs(eig2)
        l1 = np.where(swap, eig2, eig1)
        l2 = np.where(swap, eig1, eig2)

        bright_ridge = np.maximum(0.0, -l2)
        rb = np.abs(l1) / (np.abs(l2) + 1e-4)
        aniso_weight = np.exp(-(rb * rb) / (2.0 * (0.74 ** 2)))
        resp = bright_ridge * aniso_weight
        ridge_max = np.maximum(ridge_max, resp)

    p995 = float(np.percentile(ridge_max, 99.5)) if ridge_max.size > 0 else 1.0
    if p995 <= 1e-4:
        return np.zeros_like(img_u8)
    return np.clip(ridge_max / p995 * 255.0, 0.0, 255.0).astype(np.uint8)


def build_calligraphic_thin_lamina_maps(
    seg_dir: Path,
    orient: str,
    best_d_int: int,
    sl: tuple[slice, slice],
    flat_2cm: np.ndarray,
) -> dict:
    avail = {}
    for fn in seg_dir.iterdir():
        mt = re.match(rf"^dnative_{orient}_d([+-]\d+)\.png$", fn.name)
        if mt:
            avail[int(mt.group(1))] = fn.name

    def load_d(prefix: str, target_d: int) -> np.ndarray:
        p_exact = seg_dir / f"{prefix}_{orient}_d{target_d:+d}.png"
        if p_exact.exists():
            return load_gray(p_exact)[sl].astype(np.float32)
        ds = sorted(avail.keys(), key=lambda d: abs(d - target_d))
        return load_gray(seg_dir / f"{prefix}_{orient}_d{ds[0]:+d}.png")[sl].astype(np.float32)

    valid = flat_2cm > 0.30
    if valid.sum() < 200:
        valid = np.ones_like(flat_2cm, dtype=bool)

    dn0 = load_d("dnative", best_d_int)
    hc0 = load_d("hecate", best_d_int)
    step_in = 1 if ((best_d_int - 1 in avail) and (best_d_int + 1 in avail)) else 2
    dn_prev = load_d("dnative", best_d_int - step_in)
    dn_next = load_d("dnative", best_d_int + step_in)
    hc_prev = load_d("hecate", best_d_int - step_in)
    hc_next = load_d("hecate", best_d_int + step_in)

    dn_slab = 0.50 * dn0 + 0.50 * np.maximum(dn0, np.maximum(0.92 * dn_prev, 0.92 * dn_next))
    hc_slab = 0.50 * hc0 + 0.50 * np.maximum(hc0, np.maximum(0.92 * hc_prev, 0.92 * hc_next))

    dn_m4 = load_d("dnative", best_d_int - 4)
    dn_p4 = load_d("dnative", best_d_int + 4)
    hc_m4 = load_d("hecate", best_d_int - 4)
    hc_p4 = load_d("hecate", best_d_int + 4)

    def macro_clean_locked(img: np.ndarray, p_low: float, p_high: float, locked_bounds=None):
        bg = gaussian_filter(img, sigma=95.0)
        diff = img - bg
        if locked_bounds is None:
            lo = float(np.percentile(diff[valid], p_low))
            hi = float(np.percentile(diff[valid], p_high))
            hi = max(hi, lo + 4.0)
        else:
            lo, hi = locked_bounds
        normed = np.clip((diff - lo) / (hi - lo), 0.0, 1.0) * 255.0
        return normed, (lo, hi)

    dn_c0, dn_b = macro_clean_locked(dn_slab, p_low=60.0, p_high=99.5)
    dn_cm4, _ = macro_clean_locked(dn_m4, p_low=60.0, p_high=99.5, locked_bounds=dn_b)
    dn_cp4, _ = macro_clean_locked(dn_p4, p_low=60.0, p_high=99.5, locked_bounds=dn_b)

    hc_c0, hc_b = macro_clean_locked(hc_slab, p_low=58.0, p_high=99.6)
    hc_cm4, _ = macro_clean_locked(hc_m4, p_low=58.0, p_high=99.6, locked_bounds=hc_b)
    hc_cp4, _ = macro_clean_locked(hc_p4, p_low=58.0, p_high=99.6, locked_bounds=hc_b)

    dn_thin = np.maximum(0.0, dn_c0 - 0.78 * 0.5 * (dn_cm4 + dn_cp4))
    hc_thin = np.maximum(0.0, hc_c0 - 0.85 * 0.5 * (hc_cm4 + hc_cp4))

    dn_thin_u8 = np.clip(dn_thin, 0.0, 255.0).astype(np.uint8)
    hc_thin_u8 = np.clip(hc_thin, 0.0, 255.0).astype(np.uint8)

    dn_diffused = coherence_enhancing_stroke_diffusion(dn_thin_u8, n_iter=3)
    dn_ridge = multiscale_hessian_stroke_ridge(dn_diffused, sigmas=(6.0, 9.5, 13.5))
    hc_ridge = multiscale_hessian_stroke_ridge(hc_thin_u8, sigmas=(7.0, 10.5, 14.5))

    hc_anchor = gaussian_filter(np.maximum(hc_thin, hc_ridge.astype(np.float32)), sigma=45.0)
    hc_anchor_norm = np.clip(hc_anchor / max(7.0, float(np.percentile(hc_anchor[valid], 92.0))), 0.25, 1.0)

    stroke_callig = (
        0.42 * dn_diffused.astype(np.float32)
        + 0.33 * dn_ridge.astype(np.float32)
        + 0.15 * hc_thin
        + 0.10 * hc_ridge.astype(np.float32)
    ) * (0.42 + 0.58 * hc_anchor_norm) * np.clip(flat_2cm, 0.0, 1.0)

    p995 = float(np.percentile(stroke_callig[valid], 99.5)) if valid.any() else 255.0
    if p995 > 1.0:
        stroke_callig = np.clip(stroke_callig / p995 * 255.0, 0.0, 255.0)
    stroke_callig_u8 = cv2.morphologyEx(
        stroke_callig.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    ens_0_locked = (0.50 * dn_c0 + 0.50 * hc_c0).astype(np.uint8)
    ens_m4_locked = (0.50 * dn_cm4 + 0.50 * hc_cm4).astype(np.uint8)
    ens_p4_locked = (0.50 * dn_cp4 + 0.50 * hc_cp4).astype(np.uint8)

    return {
        "callig_u8": stroke_callig_u8,
        "dn_thin_u8": dn_diffused,
        "hc_thin_u8": hc_thin_u8,
        "dn_ridge_u8": dn_ridge,
        "ens_0_locked": ens_0_locked,
        "ens_m4_locked": ens_m4_locked,
        "ens_p4_locked": ens_p4_locked,
        "hc_0_locked": hc_c0.astype(np.uint8),
        "hc_m4_locked": hc_cm4.astype(np.uint8),
        "hc_p4_locked": hc_cp4.astype(np.uint8),
    }


def _build_directional_line_kernels(length: int = 19, thickness: int = 5) -> list[np.ndarray]:
    kernels = []
    c = length // 2
    for angle_deg in (0, 45, 90, 135):
        k = np.zeros((length, length), dtype=np.uint8)
        rad = np.deg2rad(angle_deg)
        dx = int(round(c * np.cos(rad)))
        dy = int(round(c * np.sin(rad)))
        cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness)
        kernels.append(k)
    return kernels


DIR_KERNELS_19 = _build_directional_line_kernels(19, 5)


def isolate_central_character_ribbon(patch_callig: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Given a `2.4 x 2.5 mm` character patch (`~260 x 275 px`):
      1. Bridges single-reed-bundle attenuation gaps (`<= 0.25 mm` = `28 px`) using directional
         morphological closing + geodesic endpoint linking between central stroke segments.
      2. Keeps ONLY the primary connected character structure (removing stray unrelated crumbs)
         and tapers the outer 10 px border smoothly to zero so there are never rectangular seams.
    """
    h, w = patch_callig.shape
    p_f = patch_callig.astype(np.float32)
    p_lo = float(np.percentile(p_f, 22.0))
    p_hi = float(np.percentile(p_f, 97.5))
    contrast = max(1.0, p_hi - p_lo)

    norm = np.clip((p_f - (p_lo + 0.14 * contrast)) / (0.82 * contrast), 0.0, 1.0)
    base_u8 = (norm * 255.0).astype(np.uint8)

    dir_closed = base_u8.copy()
    for k_dir in DIR_KERNELS_19:
        c_dir = cv2.morphologyEx(base_u8, cv2.MORPH_CLOSE, k_dir)
        dir_closed = np.maximum(
            dir_closed,
            (0.75 * c_dir.astype(np.float32) + 0.25 * base_u8.astype(np.float32)).astype(np.uint8),
        )

    bw = (dir_closed >= 62).astype(np.uint8)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    cy0, cy1 = int(0.16 * h), int(0.84 * h)
    cx0, cx1 = int(0.16 * w), int(0.84 * w)
    min_comp_area = max(40, int(0.014 * h * w))

    core_Ids = []
    for i in range(1, num_l):
        if stats[i, cv2.CC_STAT_AREA] < min_comp_area:
            continue
        comp = labels == i
        if np.any(comp[cy0:cy1, cx0:cx1]):
            core_Ids.append(i)

    if not core_Ids:
        return dir_closed, float(np.mean(bw > 0)), 0.0

    bridge_canvas = np.zeros_like(dir_closed)
    for cid in core_Ids:
        bridge_canvas[labels == cid] = 255

    if len(core_Ids) >= 2:
        comp_masses = [(cid, float(dir_closed[labels == cid].sum())) for cid in core_Ids]
        comp_masses.sort(key=lambda x: x[1], reverse=True)
        main_id = comp_masses[0][0]
        main_mask = (labels == main_id)

        for cid, cmass in comp_masses[1:4]:
            if cmass < 0.12 * comp_masses[0][1]:
                continue
            sub_mask = (labels == cid)
            dist_from_main, inds = distance_transform_edt(~main_mask, return_indices=True)
            sub_ys, sub_xs = np.where(sub_mask)
            if sub_ys.size == 0:
                continue
            d_vals = dist_from_main[sub_ys, sub_xs]
            idx_min = int(np.argmin(d_vals))
            min_dist = float(d_vals[idx_min])
            if min_dist <= 28.0:
                y_sub, x_sub = int(sub_ys[idx_min]), int(sub_xs[idx_min])
                y_main = int(inds[0, y_sub, x_sub])
                x_main = int(inds[1, y_sub, x_sub])
                cv2.line(bridge_canvas, (x_main, y_main), (x_sub, y_sub), 255, 16, cv2.LINE_AA)
                v_bridge = int(round(0.78 * min(float(dir_closed[main_mask].mean()), float(dir_closed[sub_mask].mean()))))
                cv2.line(dir_closed, (x_main, y_main), (x_sub, y_sub), max(145, v_bridge), 16, cv2.LINE_AA)
                main_mask = (bridge_canvas > 127)

    num_l2, labels2, _, _ = cv2.connectedComponentsWithStats((bridge_canvas > 127).astype(np.uint8), connectivity=8)
    best_id = -1
    best_mass = -1.0
    total_core_mass = 1e-5
    for i in range(1, num_l2):
        m_i = float(dir_closed[labels2 == i].sum())
        total_core_mass += m_i
        if m_i > best_mass:
            best_mass = m_i
            best_id = i

    if best_id < 0:
        return dir_closed, float(np.mean(bw > 0)), 0.0

    dom_ratio = best_mass / total_core_mass
    keep_mask = (labels2 == best_id)

    mask_smooth = gaussian_filter(keep_mask.astype(np.float32), sigma=3.0)
    mask_smooth = np.clip(mask_smooth * 1.45, 0.0, 1.0)
    clean_patch = gaussian_filter(dir_closed.astype(np.float32) * mask_smooth, sigma=1.4)

    # Smooth cosine border taper over the outer 8 px so isolated character patches never show sharp box edges
    by_taper = np.ones(h, dtype=np.float32)
    bx_taper = np.ones(w, dtype=np.float32)
    t_px = min(10, h // 8, w // 8)
    if t_px >= 2:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, t_px, dtype=np.float32)))
        by_taper[:t_px] = ramp
        by_taper[-t_px:] = ramp[::-1]
        bx_taper[:t_px] = ramp
        bx_taper[-t_px:] = ramp[::-1]
    clean_patch *= np.outer(by_taper, bx_taper)

    p99 = float(np.percentile(clean_patch, 99.2)) if clean_patch.size > 0 else 255.0
    if p99 > 10.0:
        clean_patch = np.clip(clean_patch / p99 * 255.0, 0.0, 255.0)
    clean_u8 = clean_patch.astype(np.uint8)
    fill_ratio = float(np.mean(clean_u8 >= 72))
    return clean_u8, fill_ratio, dom_ratio


def detect_calligraphic_greek_letters(
    callig_u8: np.ndarray,
    dn_thin_u8: np.ndarray,
    hc_thin_u8: np.ndarray,
    dn_ridge_u8: np.ndarray,
    flat_2cm: np.ndarray,
    vox_um: float,
) -> tuple[list[dict], list[int], np.ndarray]:
    H, W = callig_u8.shape
    px_per_mm = 1000.0 / vox_um
    min_char_px = int(np.ceil(2.02 * px_per_mm))
    max_char_px = int(round(4.10 * px_per_mm))
    nom_char_w = int(round(2.35 * px_per_mm))
    nom_char_h = int(round(2.50 * px_per_mm))

    valid = flat_2cm >= 0.50
    if valid.sum() < 500:
        return [], [], callig_u8

    clean = (callig_u8.astype(np.float32) * np.clip(flat_2cm, 0.0, 1.0)).astype(np.uint8)

    # 1. Find up to 5 well-spaced horizontal text lines (L1..L5, pitch >= 3.25 mm)
    row_prof = np.percentile(clean, 86.0, axis=1).astype(np.float32)
    k_1d = max(15, int(round(1.25 * px_per_mm)) | 1)
    row_smooth = cv2.GaussianBlur(row_prof.reshape(-1, 1), (1, k_1d), 0).ravel()
    min_line_dist = int(round(3.25 * px_per_mm))
    margin_y = int(round(1.30 * px_per_mm))

    line_ys = []
    for y_idx in np.argsort(row_smooth)[::-1]:
        yi = int(y_idx)
        if yi < margin_y or yi > H - margin_y:
            continue
        if all(abs(yi - ly) >= min_line_dist for ly in line_ys):
            line_ys.append(yi)
            if len(line_ys) >= 5:
                break
    line_ys.sort()

    # 2. Propose character windows from bridged stroke groups AND along horizontal lines L1..L5
    raw_rects = []
    thr_stroke = max(32.0, float(np.percentile(clean[valid], 82.0)))
    bw_stroke = (clean >= thr_stroke).astype(np.uint8) * 255
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 23))
    bw_grouped = cv2.morphologyEx(bw_stroke, cv2.MORPH_CLOSE, k_bridge)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw_grouped, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        if w < int(0.80 * px_per_mm) and h < int(0.80 * px_per_mm):
            continue
        if w <= max_char_px and h <= max_char_px:
            raw_rects.append((x, y, w, h))
        elif w > max_char_px and h <= max_char_px:
            n_h = max(2, min(7, int(round(w / nom_char_w))))
            step_w = w / n_h
            for ih in range(n_h):
                rx = int(round(x + ih * step_w))
                rw = int(round(step_w))
                raw_rects.append((rx, y, rw, h))

    for ly in line_ys:
        y0 = max(0, ly - nom_char_h // 2)
        y1 = min(H, y0 + nom_char_h)
        y0 = max(0, y1 - nom_char_h)
        strip = clean[y0:y1, :].astype(np.float32)
        col_prof = np.percentile(strip, 88.0, axis=0)
        k_col = max(15, int(round(0.85 * px_per_mm)) | 1)
        col_smooth = cv2.GaussianBlur(col_prof.reshape(-1, 1), (1, k_col), 0).ravel()
        picked_xs = []
        for x_idx in np.argsort(col_smooth)[::-1]:
            xi = int(x_idx)
            if xi < nom_char_w // 2 or xi > W - nom_char_w // 2:
                continue
            if col_smooth[xi] < max(28.0, thr_stroke * 0.52):
                break
            if all(abs(xi - px) >= int(round(1.90 * px_per_mm)) for px in picked_xs):
                picked_xs.append(xi)
                raw_rects.append((xi - nom_char_w // 2, y0, nom_char_w, nom_char_h))
                if len(picked_xs) >= 9:
                    break

    # Fast pre-scoring & spatial deduplication BEFORE expensive geodesic EDT + 60-template matching
    pre_scored = []
    for (x, y, w, h) in raw_rects:
        if w < min_char_px:
            pad_x = (nom_char_w - w) // 2
            x = max(0, min(W - nom_char_w, x - pad_x))
            w = min(nom_char_w, W - x)
        if h < min_char_px:
            pad_y = (nom_char_h - h) // 2
            y = max(0, min(H - nom_char_h, y - pad_y))
            h = min(nom_char_h, H - y)
        if w > max_char_px or h > max_char_px or w < min_char_px or h < min_char_px:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 1.70:
            continue

        patch_fl = flat_2cm[y:y + h, x:x + w]
        fl_mean = float(np.mean(patch_fl))
        if fl_mean < 0.58 or float(np.percentile(patch_fl, 10.0)) < 0.18:
            continue

        patch_dn = dn_thin_u8[y:y + h, x:x + w]
        patch_hc = hc_thin_u8[y:y + h, x:x + w]
        patch_rg = dn_ridge_u8[y:y + h, x:x + w]
        dn_p94 = float(np.percentile(patch_dn, 94.0))
        hc_p985 = float(np.percentile(patch_hc, 98.5))
        rg_p92 = float(np.percentile(patch_rg, 92.0))
        if dn_p94 < 32.0 or hc_p985 < 15.0 or rg_p92 < 30.0:
            continue

        patch_callig = clean[y:y + h, x:x + w]
        p_hi = float(np.percentile(patch_callig, 96.0))
        p_lo = float(np.percentile(patch_callig, 25.0))
        contrast = p_hi - p_lo
        if contrast < 26.0:
            continue

        fast_sc = (0.35 * dn_p94 + 0.25 * hc_p985 + 0.25 * rg_p92 + 0.15 * contrast) * fl_mean
        pre_scored.append((fast_sc, x, y, w, h, fl_mean, dn_p94, hc_p985, rg_p92, contrast))

    pre_scored.sort(key=lambda t: t[0], reverse=True)
    dedup_rects = []
    pre_dist = 1.25 * px_per_mm
    for item in pre_scored:
        _, x, y, w, h, _, _, _, _, _ = item
        cx, cy = x + 0.5 * w, y + 0.5 * h
        if all(np.hypot(cx - ( kx + 0.5 * kw ), cy - ( ky + 0.5 * kh )) >= pre_dist for (_, kx, ky, kw, kh, *_) in dedup_rects):
            dedup_rects.append(item)
            if len(dedup_rects) >= 32:
                break

    cand_boxes = []
    for (_, x, y, w, h, fl_mean, dn_p94, hc_p985, rg_p92, contrast) in dedup_rects:
        patch_callig = clean[y:y + h, x:x + w]
        ribbon_patch, fill_ratio, dom_ratio = isolate_central_character_ribbon(patch_callig)
        if fill_ratio < 0.10 or fill_ratio > 0.48 or dom_ratio < 0.72:
            continue

        active_rows = np.any(ribbon_patch >= 72, axis=1)
        active_cols = np.any(ribbon_patch >= 72, axis=0)
        if np.sum(active_rows) < 0.45 * h or np.sum(active_cols) < 0.22 * w:
            continue

        g_code, g_uni, tpl_sc, tpl_u8 = classify_greek_letter_patch(ribbon_patch)
        if tpl_sc < 0.36:
            continue

        cx = x + 0.5 * w
        cy = y + 0.5 * h
        line_id = 0
        line_bonus = 0.88
        for idx_l, ly in enumerate(line_ys):
            if abs(cy - ly) <= 1.05 * px_per_mm:
                line_id = idx_l + 1
                line_bonus = 1.18
                break

        fill_quality = 1.0 - min(0.55, abs(fill_ratio - 0.24) * 1.8)
        score = (
            (0.32 * dn_p94 + 0.24 * hc_p985 + 0.28 * rg_p92 + 0.16 * contrast)
            * (fl_mean ** 0.85)
            * fill_quality
            * (dom_ratio ** 0.7)
            * (0.70 + 0.65 * tpl_sc)
            * line_bonus
        )

        cand_boxes.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "w_mm": round(w / px_per_mm, 2),
            "h_mm": round(h / px_per_mm, 2),
            "cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "dn_p94": round(dn_p94, 1),
            "hc_p94": round(hc_p985, 1),
            "ridge_p92": round(rg_p92, 1),
            "fill_ratio": round(fill_ratio, 3),
            "dom_stroke_ratio": round(dom_ratio, 3),
            "contrast": round(contrast, 1),
            "flatness": round(fl_mean, 2),
            "greek_ascii": g_code,
            "greek_char": g_uni,
            "tpl_score": tpl_sc,
            "line_id": line_id,
            "score": round(score, 1),
            "_tpl_u8": tpl_u8,
            "_ribbon_u8": ribbon_patch,
        })

    cand_boxes.sort(key=lambda b: b["score"], reverse=True)
    boxes = []
    min_center_dist = 1.88 * px_per_mm
    char_counts: dict[str, int] = {}
    for b in cand_boxes:
        if all(np.hypot(b["cx"] - kb["cx"], b["cy"] - kb["cy"]) >= min_center_dist for kb in boxes):
            c_ascii = b["greek_ascii"]
            if char_counts.get(c_ascii, 0) >= 3 and len(boxes) < 12:
                continue
            char_counts[c_ascii] = char_counts.get(c_ascii, 0) + 1
            boxes.append(b)
            if len(boxes) >= 24:
                break

    # Smoothly blend isolated ribbons into display_callig without rectangular borders
    display_callig = (clean.astype(np.float32) * 0.55).astype(np.uint8)
    for b in boxes[:16]:
        bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
        display_callig[by:by + bh_b, bx:bx + bw_b] = np.maximum(
            display_callig[by:by + bh_b, bx:bx + bw_b],
            b["_ribbon_u8"],
        )

    return boxes, line_ys, display_callig


def render_stage_i_paleography_dossier(
    scroll: str,
    seg: str,
    orient: str,
    pair_label: str,
    mean_flat: float,
    r_2x2cm: float,
    px_per_mm: float,
    p1_gold_clean: np.ndarray,
    crop_callig: np.ndarray,
    boxes: list[dict],
    line_ys: list[int],
    out_dir: Path,
):
    """
    Renders the Stage I v5 Paleographic Dossier where each Top-12 Evidence Card displays:
      - Sub-tile 1 (`CT+Gold`): 2x2cm CT+Gold context crop around the character box.
      - Sub-tile 2 (`In-Focus`): Seam-free isolated calligraphic character ribbon (`_ribbon_u8`)
        placed inside a clean padded dark frame (`pad = 0.22 mm`), eliminating any rectangular
        border seams or neighboring stroke crumbs.
      - Sub-tile 3 (`Tpl:<Greek>`): Matched Greek Uncial Template + explicit `dz4` drop %.
    """
    wh, ww = crop_callig.shape
    annot = p1_gold_clean.copy()
    step_2mm = int(round(2.0 * px_per_mm))

    for i_l, ly in enumerate(line_ys):
        cv2.line(annot, (0, ly), (ww - 1, ly), (255, 200, 80), 1, cv2.LINE_AA)
        cv2.putText(
            annot, f"L{i_l+1}", (10, max(20, ly - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 210, 90), 2, cv2.LINE_AA,
        )

    for idx_b, b in enumerate(boxes[:20]):
        bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
        color = (80, 255, 120) if idx_b < 12 else (120, 220, 255)
        cv2.rectangle(annot, (bx, by), (bx + bw_b, by + bh_b), color, 2)
        lbl = f"#{idx_b+1}:{b['greek_ascii']} ({b['w_mm']:.1f}x{b['h_mm']:.1f}mm)"
        cv2.putText(
            annot, lbl, (bx, max(18, by - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )

    cv2.rectangle(annot, (ww - 24 - step_2mm, 24), (ww - 24, 24 + step_2mm), (0, 235, 255), 2)
    cv2.putText(
        annot, "2x2mm", (ww - 28 - step_2mm, 44 + step_2mm),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 235, 255), 1, cv2.LINE_AA,
    )

    hdr = f"{scroll}/{seg} ({orient}, {pair_label}) | flat={mean_flat:.2f} | r_2cm={r_2x2cm:.4f} | {len(boxes)} letters >=2x2mm"
    cv2.rectangle(annot, (8, 8), (min(ww - 8, 18 + len(hdr) * 9), 38), (15, 15, 15), -1)
    cv2.putText(annot, hdr, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 245, 255), 1, cv2.LINE_AA)

    left_panel = cv2.resize(annot, (960, 960), interpolation=cv2.INTER_AREA)

    card_w, card_h = 320, 240
    sub_sz = 100
    cards = []
    for idx_b in range(12):
        card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
        card[:] = (18, 18, 22)
        if idx_b < len(boxes):
            b = boxes[idx_b]
            bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
            pad = int(round(0.24 * px_per_mm))
            x0_c = max(0, bx - pad)
            y0_c = max(0, by - pad)
            x1_c = min(ww, bx + bw_b + pad)
            y1_c = min(wh, by + bh_b + pad)

            g_crop = cv2.resize(p1_gold_clean[y0_c:y1_c, x0_c:x1_c], (sub_sz, 172), interpolation=cv2.INTER_CUBIC)

            # Build seam-free In-Focus crop from the isolated character ribbon (`_ribbon_u8`)
            rib = b.get("_ribbon_u8")
            if rib is not None:
                rib_pad = np.zeros((bh_b + 2 * pad, bw_b + 2 * pad), dtype=np.uint8)
                rib_pad[pad:pad + bh_b, pad:pad + bw_b] = rib
            else:
                rib_pad = crop_callig[y0_c:y1_c, x0_c:x1_c]

            k_crop = cv2.cvtColor(
                cv2.resize(rib_pad, (sub_sz, 172), interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_GRAY2BGR,
            )
            tpl_bgr = cv2.cvtColor(
                cv2.resize(b["_tpl_u8"], (sub_sz, 172), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_GRAY2BGR,
            )
            tpl_bgr[:, :, 0] = np.clip(tpl_bgr[:, :, 0] * 0.6, 0, 255).astype(np.uint8)
            tpl_bgr[:, :, 1] = np.clip(tpl_bgr[:, :, 1] * 1.0, 0, 255).astype(np.uint8)

            card[52:224, 5:5 + sub_sz] = g_crop
            card[52:224, 10 + sub_sz:10 + 2 * sub_sz] = k_crop
            card[52:224, 15 + 2 * sub_sz:15 + 3 * sub_sz] = tpl_bgr

            t1 = f"#{idx_b+1} [{b['greek_ascii']}] {b['w_mm']:.1f}x{b['h_mm']:.1f}mm L{b['line_id']}"
            dz_drop = b.get("stroke_drop_eff_pm4_pct", 0.0)
            t2 = f"dz4=-{dz_drop:.0f}% fl={b['flatness']:.2f} tpl={b['tpl_score']:.2f}"
            cv2.putText(card, t1, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 255, 140), 1, cv2.LINE_AA)
            cv2.putText(card, t2, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (210, 225, 245), 1, cv2.LINE_AA)
            cv2.putText(card, "CT+Gold", (14, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(card, "In-Focus", (118, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(card, f"Tpl:{b['greek_ascii']}", (222, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 240, 200), 1, cv2.LINE_AA)
        cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (65, 70, 80), 1)
        cards.append(card)

    right_rows = [np.hstack(cards[r:r + 3]) for r in range(0, 12, 3)]
    right_panel = np.vstack(right_rows)
    dossier = np.hstack([left_panel, right_panel])
    cv2.imwrite(str(out_dir / "paleography_dossier.png"), dossier)


def filter_boxes_by_frozen_stroke_z_attenuation(
    boxes: list[dict],
    callig_2cm: np.ndarray,
    ens_dstar: np.ndarray,
    ens_m4: np.ndarray,
    ens_p4: np.ndarray,
    hc_dstar: np.ndarray,
    hc_m4: np.ndarray,
    hc_p4: np.ndarray,
    min_drop_pm4_pct: float = 30.0,
) -> tuple[list[dict], list[dict]]:
    thin_boxes = []
    rejected_thick = []
    H, W = ens_dstar.shape

    for b in boxes:
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        y0, y1 = max(0, by), min(H, by + bh)
        x0, x1 = max(0, bx), min(W, bx + bw)
        p_cal = callig_2cm[y0:y1, x0:x1].astype(np.float32)
        p_e0 = ens_dstar[y0:y1, x0:x1].astype(np.float32)
        p_em4 = ens_m4[y0:y1, x0:x1].astype(np.float32)
        p_ep4 = ens_p4[y0:y1, x0:x1].astype(np.float32)

        p_h0 = hc_dstar[y0:y1, x0:x1].astype(np.float32)
        p_hm4 = hc_m4[y0:y1, x0:x1].astype(np.float32)
        p_hp4 = hc_p4[y0:y1, x0:x1].astype(np.float32)

        if p_e0.size < 100:
            continue
        thr78 = float(np.percentile(p_cal, 78.0))
        m_stk = p_cal >= max(thr78, 20.0)
        if m_stk.sum() < 25:
            m_stk = p_e0 >= float(np.percentile(p_e0, 80.0))

        e0 = float(p_e0[m_stk].mean())
        em4 = float(p_em4[m_stk].mean())
        ep4 = float(p_ep4[m_stk].mean())
        drop_e_m4 = (e0 - em4) / max(e0, 1.0) * 100.0
        drop_e_p4 = (e0 - ep4) / max(e0, 1.0) * 100.0
        drop_e_pm4 = (e0 - 0.5 * (em4 + ep4)) / max(e0, 1.0) * 100.0

        h0 = float(p_h0[m_stk].mean())
        hm4 = float(p_hm4[m_stk].mean())
        hp4 = float(p_hp4[m_stk].mean())
        drop_h_pm4 = (h0 - 0.5 * (hm4 + hp4)) / max(h0, 1.0) * 100.0

        eff_drop = max(drop_e_pm4, 0.60 * drop_h_pm4 + 0.40 * drop_e_pm4)
        b_copy = dict(b)
        b_copy["stroke_drop_ens_pm4_pct"] = round(drop_e_pm4, 1)
        b_copy["stroke_drop_hc_pm4_pct"] = round(drop_h_pm4, 1)
        b_copy["stroke_drop_eff_pm4_pct"] = round(eff_drop, 1)
        b_copy["stroke_drop_m4_pct"] = round(drop_e_m4, 1)
        b_copy["stroke_drop_p4_pct"] = round(drop_e_p4, 1)

        if eff_drop >= min_drop_pm4_pct:
            thin_boxes.append(b_copy)
        else:
            b_copy.pop("_ribbon_u8", None)
            b_copy.pop("_tpl_u8", None)
            rejected_thick.append(b_copy)

    return thin_boxes, rejected_thick


def evaluate_stage_i_candidate(seg_dir: Path) -> dict | None:
    meta_path = seg_dir / "metrics_stage_i.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    scroll_key = meta["scroll_key"]
    cand_id = meta["cand_id"]
    orient = meta["chosen_orient"]
    um = float(meta["um"])
    px_per_mm = 1000.0 / um
    best_d_key = meta["best_single_depth"]
    best_d_int = int(best_d_key.replace("d", ""))

    ct_mid = load_gray(seg_dir / "ct_mid_single_dstar.png")
    if ct_mid is None:
        ct_mid = load_gray(seg_dir / "ct_mid.png")
    if ct_mid is None:
        return None

    H, W = ct_mid.shape
    w2cm = min(H, W, int(round(20.0 * px_per_mm)))
    y0 = (H - w2cm) // 2
    x0 = (W - w2cm) // 2
    sl = (slice(y0, y0 + w2cm), slice(x0, x0 + w2cm))

    ct_2cm = ct_mid[sl]
    flat_2cm, ct_lamina_stats = compute_strict_single_lamina_ct_metrics(ct_2cm)

    lamina_maps = build_calligraphic_thin_lamina_maps(seg_dir, orient, best_d_int, sl, flat_2cm)
    callig_2cm = lamina_maps["callig_u8"]
    dn_t_2cm = lamina_maps["dn_thin_u8"]
    hc_t_2cm = lamina_maps["hc_thin_u8"]
    dn_rg_2cm = lamina_maps["dn_ridge_u8"]

    raw_boxes, line_ys, display_callig_2cm = detect_calligraphic_greek_letters(
        callig_2cm, dn_t_2cm, hc_t_2cm, dn_rg_2cm, flat_2cm, vox_um=um
    )

    thin_boxes, rejected_boxes = filter_boxes_by_frozen_stroke_z_attenuation(
        raw_boxes,
        callig_2cm,
        lamina_maps["ens_0_locked"], lamina_maps["ens_m4_locked"], lamina_maps["ens_p4_locked"],
        lamina_maps["hc_0_locked"], lamina_maps["hc_m4_locked"], lamina_maps["hc_p4_locked"],
        min_drop_pm4_pct=30.0,
    )

    avail_depths = []
    for d in range(-6, 7):
        p_e = seg_dir / f"ens_z_{orient}_d{d:+d}.png"
        p_h = seg_dir / f"hecate_{orient}_d{d:+d}.png"
        if p_e.exists() and p_h.exists():
            avail_depths.append((d, load_gray(p_e)[sl], load_gray(p_h)[sl]))

    coherent_pm1 = 0
    coherent_pm2 = 0
    for b in thin_boxes[:12]:
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        sub_cal = callig_2cm[by:by + bh, bx:bx + bw].astype(np.float32)
        m_s = sub_cal >= max(float(np.percentile(sub_cal, 78.0)), 20.0)
        if m_s.sum() < 15:
            m_s = np.ones_like(sub_cal, dtype=bool)
        best_d_box = best_d_int
        best_val_box = -1.0
        for d, e_im, h_im in avail_depths:
            sc_d = 0.5 * float(e_im[by:by + bh, bx:bx + bw][m_s].mean()) + 0.5 * float(h_im[by:by + bh, bx:bx + bw][m_s].mean())
            if sc_d > best_val_box:
                best_val_box = sc_d
                best_d_box = d
        b["d_peak_vox"] = int(best_d_box)
        if abs(best_d_box - best_d_int) <= 1:
            coherent_pm1 += 1
        if abs(best_d_box - best_d_int) <= 2:
            coherent_pm2 += 1

    top12 = thin_boxes[:12]
    n_top = len(top12)
    mean_box_drop_pm4 = float(np.mean([b["stroke_drop_eff_pm4_pct"] for b in top12])) if n_top > 0 else float(meta["stroke_atten_pm4_pct"])
    mean_tpl = float(np.mean([b.get("tpl_score", 0.0) for b in top12])) if n_top > 0 else 0.0
    mean_fill = float(np.mean([b.get("fill_ratio", 0.0) for b in top12])) if n_top > 0 else 0.0

    VOWELS = set("AEHIOYW")
    lines_dict = {}
    for idx_l, ly in enumerate(line_ys):
        l_boxes = sorted([b for b in thin_boxes if abs(b["cy"] - ly) <= int(1.45 * px_per_mm)], key=lambda b: b["x"])
        if not l_boxes:
            continue
        seq_ascii = "".join(b["greek_ascii"] for b in l_boxes)
        seq_uni = "".join(b["greek_char"].split()[0] for b in l_boxes)
        lines_dict[f"L{idx_l+1}"] = {
            "num_letters": len(l_boxes),
            "ascii_seq": seq_ascii,
            "unicode_seq": seq_uni,
        }
    full_ascii = "".join(v["ascii_seq"] for v in lines_dict.values())
    if not full_ascii and thin_boxes:
        full_ascii = "".join(b["greek_ascii"] for b in thin_boxes[:12])
    vowel_ratio = sum(1 for c in full_ascii if c in VOWELS) / max(1, len(full_ascii))
    matched_stems = [st for st in HERCULANEUM_STEMS if st in full_ascii]

    r_max = max(float(meta["r_raw_2x2cm_dstar"]), float(meta["r_hp_2x2cm_dstar"]), float(meta["r_thin_2x2cm_dstar"]))
    coh_frac_pm2 = coherent_pm2 / max(1, n_top)

    ct_strat = dict(meta["ct_stratigraphy"])
    ct_strat.update(ct_lamina_stats)

    # Evaluate the 5 H4 CLSS Skeptic Gate Criteria (aligned with pre-registered Bet 5 m7 >= 62.0%):
    c1_r = bool(r_max >= 0.25)
    c2_atten = bool(mean_box_drop_pm4 >= 35.0 and float(meta["stroke_atten_pm4_pct"]) >= 34.0)
    c3_coh = bool(coh_frac_pm2 >= 0.65 and n_top >= 6)
    c4_letters = bool(len(thin_boxes) >= 8 and 0.22 <= vowel_ratio <= 0.72 and 0.10 <= mean_fill <= 0.48)
    c5_recto = bool(
        ct_strat["is_recto_perfibrale"]
        and ct_lamina_stats["valid_fiber_coverage"] >= 0.90
        and ct_lamina_stats["single_sheet_flatness_pct"] >= 62.0
        and ct_lamina_stats["swirl_delam_frac"] <= 0.36
        and meta["stage_i_m7_snap"]["m7_ridge_attenuation_pm4_pct"] >= 62.0
    )
    criteria_passed = int(c1_r) + int(c2_atten) + int(c3_coh) + int(c4_letters) + int(c5_recto)
    h4_status = "VERIFIED_PASS" if criteria_passed == 5 else ("STRONG_CANDIDATE_4_5" if criteria_passed == 4 else "PARTIAL")

    out_cand_dir = STAGE_I_DOSSIER_DIR / f"{scroll_key}__{cand_id}"
    if len(thin_boxes) >= 6 and r_max >= 0.18:
        out_cand_dir.mkdir(parents=True, exist_ok=True)
        valid_2cm = ct_2cm > 15
        gold_2cm = make_calligraphic_gold(ct_2cm, display_callig_2cm, valid_2cm)
        cv2.imwrite(str(out_cand_dir / "submission_gold_2x2cm_thin_gated.png"), gold_2cm)
        cv2.imwrite(str(out_cand_dir / "calligraphic_stroke_ridge_2x2cm.png"), display_callig_2cm)
        render_stage_i_paleography_dossier(
            scroll_key,
            cand_id,
            orient,
            f"StageI-v5-SeamFreeRibbon-{best_d_key}",
            float(flat_2cm.mean()),
            r_max,
            px_per_mm,
            gold_2cm,
            display_callig_2cm,
            thin_boxes,
            line_ys,
            out_cand_dir,
        )

    for b in thin_boxes:
        b.pop("_ribbon_u8", None)
        b.pop("_tpl_u8", None)

    return {
        "scroll_key": scroll_key,
        "cand_id": cand_id,
        "pool": meta["pool"],
        "orient": orient,
        "best_single_depth": best_d_key,
        "promoted_to_4x_tta": meta["promoted_to_4x_tta"],
        "r_raw_2x2cm": float(meta["r_raw_2x2cm_dstar"]),
        "r_hp_2x2cm": float(meta["r_hp_2x2cm_dstar"]),
        "r_thin_2x2cm": float(meta["r_thin_2x2cm_dstar"]),
        "r_max_2x2cm": round(r_max, 4),
        "stroke_atten_pm4_pct": float(meta["stroke_atten_pm4_pct"]),
        "hecate_stroke_atten_pm4_pct": float(meta["hecate_stroke_atten_pm4_pct"]),
        "thin_boxes_count": len(thin_boxes),
        "rejected_thick_knots_count": len(rejected_boxes),
        "mean_thin_box_drop_pm4_pct": round(mean_box_drop_pm4, 1),
        "mean_stroke_fill_ratio": round(mean_fill, 3),
        "coherent_pm1_of_top12": f"{coherent_pm1}/{n_top}",
        "coherent_pm2_of_top12": f"{coherent_pm2}/{n_top}",
        "mean_template_score": round(mean_tpl, 3),
        "vowel_ratio": round(vowel_ratio, 2),
        "matched_herculaneum_stems": matched_stems[:10],
        "transcribed_lines": lines_dict,
        "ct_stratigraphy": ct_strat,
        "m7_snap": meta["stage_i_m7_snap"],
        "h4_criteria_passed": f"{criteria_passed}/5",
        "h4_criteria_detail": {
            "c1_r_ge_025": c1_r,
            "c2_atten_pm4_ge_35pct": c2_atten,
            "c3_depth_coherence_ge_65pct": c3_coh,
            "c4_eight_greek_letters": c4_letters,
            "c5_recto_fiber_and_m7_sharp": c5_recto,
        },
        "h4_status": h4_status,
    }


def main():
    seg_dirs = [sd for sd in sorted(STAGE_I_DIR.glob("w*/*/*")) if sd.is_dir() and (sd / "metrics_stage_i.json").exists()]
    results = []
    print(f"[Stage I v5] Starting audit across {len(seg_dirs)} candidates...", flush=True)
    for idx_s, sd in enumerate(seg_dirs, 1):
        r = evaluate_stage_i_candidate(sd)
        if r is not None:
            results.append(r)
            print(
                f"  [{idx_s:2d}/{len(seg_dirs)}] {r['scroll_key']}/{r['cand_id']:26s} -> {r['h4_status']} ({r['h4_criteria_passed']}) "
                f"| r={r['r_max_2x2cm']:.4f} | drop_pm4={r['mean_thin_box_drop_pm4_pct']:.1f}% | boxes={r['thin_boxes_count']}",
                flush=True,
            )

    results.sort(
        key=lambda r: (
            1 if r["h4_status"] == "VERIFIED_PASS" else (0.5 if r["h4_status"] == "STRONG_CANDIDATE_4_5" else 0),
            r["r_max_2x2cm"]
            * (max(10.0, r["mean_thin_box_drop_pm4_pct"]) / 100.0)
            * (r["ct_stratigraphy"]["single_sheet_flatness_pct"] / 100.0),
        ),
        reverse=True,
    )

    manifest_i = json.loads((BASE / "meshes" / "stage_i_scout_manifest.json").read_text())
    bet5_pass = manifest_i.get("passed_bet5_m7_gate_count", 0) >= 28
    promoted_or_active = [r for r in results if r["thin_boxes_count"] >= 4]
    bet6_pass = bool(len(promoted_or_active) > 0 and all(r["mean_thin_box_drop_pm4_pct"] >= 30.0 for r in promoted_or_active))
    bet7_hits = [
        r for r in results
        if r["r_max_2x2cm"] >= 0.28
        and max(r["mean_thin_box_drop_pm4_pct"], r["stroke_atten_pm4_pct"]) >= 35.0
        and r["ct_stratigraphy"]["valid_fiber_coverage"] >= 0.90
    ]
    bet7_pass = len(bet7_hits) >= 3
    bet8_hits = [r for r in results if r["h4_status"] == "VERIFIED_PASS"]
    bet8_pass = len(bet8_hits) >= 1

    summary = {
        "total_evaluated_on_gpu": len(results),
        "bets_evaluation": {
            "Bet_5_CPU_m7Snap_Gate": {"status": "PASS" if bet5_pass else "FAIL", "count": manifest_i.get("passed_bet5_m7_gate_count", 0)},
            "Bet_6_ThinLayer_ZDoG_Gate_ge_30pct": {"status": "PASS" if bet6_pass else "FAIL", "min_drop_pct": min([r["mean_thin_box_drop_pm4_pct"] for r in promoted_or_active], default=0.0)},
            "Bet_7_Scout_Hits_r028_atten35": {"status": "PASS" if bet7_pass else "FAIL", "hits_count": len(bet7_hits), "hits": [f"{r['scroll_key']}/{r['cand_id']}" for r in bet7_hits]},
            "Bet_8_H4_Verified_Pass_5_of_5": {"status": "PASS" if bet8_pass else "FAIL", "verified_count": len(bet8_hits), "verified": [f"{r['scroll_key']}/{r['cand_id']}" for r in bet8_hits]},
        },
        "leaderboard": results,
    }
    out_json = STAGE_I_DOSSIER_DIR / "leaderboard_stage_i.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"=== Stage I (v4 Geodesic Ribbon + Dominant Topology Gate) CLSS Audit Complete ({len(results)} candidates) ===", flush=True)
    print(json.dumps(summary["bets_evaluation"], indent=2), flush=True)
    for idx, r in enumerate(results[:14]):
        ct_s = r["ct_stratigraphy"]
        print(
            f"  #{idx+1:2d} [{r['h4_status']:20s} ({r['h4_criteria_passed']})] {r['scroll_key']}/{r['cand_id']:28s} "
            f"| r={r['r_max_2x2cm']:.4f} | drop_pm4={r['mean_thin_box_drop_pm4_pct']:4.1f}% "
            f"| fib={ct_s['valid_fiber_coverage']:.3f} flat={ct_s['single_sheet_flatness_pct']:4.1f}% swirl={ct_s['swirl_delam_frac']:.2f} "
            f"| boxes={r['thin_boxes_count']:2d} fill={r['mean_stroke_fill_ratio']:.2f} tpl={r['mean_template_score']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
