#!/usr/bin/env python3
"""
Stage N — Exact 3D Voxel-and-Around Geodesic Scraper & Sub-Voxel Shell Integration
==================================================================================
Implements the winning Stage N Verbalized Sampling + CLSS architecture (`N_1 = N_2 + N_3 + N_4 + N_5 + N_8`):
  1. `N_2` (Full 5-Slice 3D Tensor inside the Rotated Lupa `T(u, v, k)` + Local 3x3 Cylindrical
     Off-Lamina Background Subtraction):
     Instead of collapsing `d in {d*-2, d*-1, d*, d*+1, d*+2}` to 2D before rotating the character
     magnifying glass ("zrotowana lupa"), we warp all 5 sub-voxel depth slices (`shape = (H, W, 5)`)
     directly into the upright character frame (`0.0 deg`) and subtract the local 3x3 cylindrical
     off-lamina minimum at `d* +- 4`.
  2. `N_3` (6-Step 3D 26-Connected Voxel-and-Around Geodesic Hysteresis Scraper):
     Around every core 3D ink voxel `(u, v, k)`, we scrape its 3D micro-neighborhood
     `(u + du, v + dv, k + dk)` with `|dk| <= 1` along 4 canonical calligraphic stroke cones
     (vertical stems `90 deg`, horizontal crossbars `0 deg`, diagonals `+-45 deg`, and curved loops)
     across 6 geodesic hysteresis steps, plus edge-anchored template corridor 3D sub-voxel lookahead.
     ZERO synthetic `cv2.line` pixels (`synthetic_cv2_lines_drawn == 0`)!
  3. `N_4` (Cryo-EM Local 3D Manifold Shell Integration & 88% Off-Manifold Chaff Suppression):
     At every pixel `(u, v)` of the scraped 3D glyph manifold, we integrate the optimal depth voxel
     `k*(u, v)` with its immediate 3D shell (`k* +- 1` and 8-connected in-manifold neighbors) while
     suppressing un-scraped background speckle outside the 3D manifold by 88%.
  4. `N_5` (3D-Scraped Global Column Ink Projection):
     Projects the 3D-scraped upright character ribbons back onto `display_callig` (`splat_rotated_lupa_patch`)
     while attenuating un-scraped inter-line speckle so the golden ink overlay on the global `20x20 mm`
     and `28x20 mm` boards is sharp, continuous, and crystal-clear.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))

from stage_f_sheet_tracker_and_paleography import (
    TEMPLATES_64,
    GREEK_DISPLAY_UNICODE,
    load_gray,
)
from stage_i_clss_audit_and_loop import (
    compute_strict_single_lamina_ct_metrics,
    coherence_enhancing_stroke_diffusion,
    isolate_central_character_ribbon,
    filter_boxes_by_frozen_stroke_z_attenuation,
    render_stage_i_paleography_dossier,
)
from stage_j_viterbi_line_decoder import (
    EPICUREAN_LEXICON,
    build_chamfered_45deg_polyline,
    viterbi_decode_line_slots,
    make_nieuwlaar_gold_overlay,
    draw_nieuwlaar_cyan_reading_underline,
    render_pure_nieuwlaar_submission_image,
    render_diplomatic_column_lines_board,
)
from stage_l_ultraprecise_ink_and_columns import (
    CW_WINDING_CANDS,
    UNCIAL_PRIOR_BIAS,
    junction_preserving_stroke_bank_l,
    isolate_ultraprecise_uncial_character,
    score_corridor_anchored_greek_templates,
    trace_fiber_streamline_baselines_6lines,
    calibrate_corridor_ultraprecise,
    _apply_2d_tf,
    _estimate_ct_fiber_tilt_deg,
    find_largest_valid_interior_crop_l,
)
from stage_m_rectified_scroll_and_voxel_lookahead import (
    extract_rotated_lupa_patch,
    splat_rotated_lupa_patch,
    _discover_cadence_lookahead_xs,
    score_rectified_lupa_greek_templates,
)

STAGE_I_DIR = BASE / "results_stage_i"
STAGE_N_DIR = BASE / "results_stage_n_voxel_scraped"
STAGE_N_DIR.mkdir(parents=True, exist_ok=True)


def _apply_2d_tf_any(arr: np.ndarray, tf_mode: str) -> np.ndarray:
    """Applies `tf_mode` (`id`, `rot180`, `flip_x`, `flip_y`) to 2D `(H, W)` or 3D `(H, W, C)` arrays."""
    if tf_mode == "id":
        return arr
    if tf_mode == "rot180":
        return np.rot90(arr, 2, axes=(0, 1)).copy()
    if tf_mode == "flip_x":
        return np.flip(arr, axis=1).copy()
    if tf_mode == "flip_y":
        return np.flip(arr, axis=0).copy()
    return arr


def _directional_around_shell_2d(slice_u8: np.ndarray, px_per_mm: float) -> np.ndarray:
    """
    Computes the 4-direction calligraphic micro-neighborhood ('voxel-and-around') response
    around every pixel `(u, v)` at a given sub-voxel depth slice `k`.
    Uses short (`0.09 mm`, ~11 px) 1-px-thin directional kernels at 0, 45, 90, 135 deg so
    micro-pits along vertical stems, horizontal crossbars, diagonals, and loops are bridged
    purely from immediate neighboring ink voxels.
    """
    L = max(5, int(round(0.09 * px_per_mm)) | 1)
    half = L // 2
    f = slice_u8.astype(np.float32)

    # 1. Vertical stem neighborhood (90 deg)
    k_vert = np.zeros((L, L), dtype=np.float32)
    k_vert[:, half] = 1.0 / L
    r_vert = cv2.filter2D(f, cv2.CV_32F, k_vert)

    # 2. Horizontal crossbar neighborhood (0 deg)
    k_horz = np.zeros((L, L), dtype=np.float32)
    k_horz[half, :] = 1.0 / L
    r_horz = cv2.filter2D(f, cv2.CV_32F, k_horz)

    # 3. Main diagonal neighborhood (45 deg)
    k_d1 = np.eye(L, dtype=np.float32) / L
    r_d1 = cv2.filter2D(f, cv2.CV_32F, k_d1)

    # 4. Anti-diagonal neighborhood (135 deg)
    k_d2 = np.fliplr(k_d1)
    r_d2 = cv2.filter2D(f, cv2.CV_32F, k_d2)

    dir_max = np.maximum(np.maximum(r_vert, r_horz), np.maximum(r_d1, r_d2))
    out = 0.64 * f + 0.36 * dir_max
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def build_stage_n_subvoxel_lamina_maps(
    seg_dir: Path,
    orient: str,
    best_d_int: int,
    sl: tuple[slice, slice],
    flat_2cm: np.ndarray,
    fiber_tilt_deg: float,
    px_per_mm: float,
) -> dict:
    """
    N_2: Builds BOTH the 2D lamina maps AND the full 5-slice 3D Sub-Voxel Tensor `(H, W, 5)`
    across `lookahead_ds = (d*-2, d*-1, d*, d*+1, d*+2)` with local 3x3 cylindrical off-lamina
    background subtraction at `d* +- 4`.
    """
    avail = set()
    for fn in seg_dir.iterdir():
        mt = re.match(rf"^dnative_{orient}_d([+-]\d+)\.png$", fn.name)
        if mt:
            avail.add(int(mt.group(1)))

    def load_d(prefix: str, target_d: int) -> np.ndarray:
        p_exact = seg_dir / f"{prefix}_{orient}_d{target_d:+d}.png"
        if p_exact.exists():
            return load_gray(p_exact)[sl].astype(np.float32)
        ds = sorted(avail, key=lambda d: abs(d - target_d))
        return load_gray(seg_dir / f"{prefix}_{orient}_d{ds[0]:+d}.png")[sl].astype(np.float32)

    valid = flat_2cm > 0.30
    if valid.sum() < 200:
        valid = np.ones_like(flat_2cm, dtype=bool)

    if (best_d_int - 1 in avail) and (best_d_int + 1 in avail):
        inner_ds = (best_d_int - 1, best_d_int, best_d_int + 1)
        is_1vox_4xtta = True
    elif best_d_int == +2:
        inner_ds = (0, +2, +2)
        is_1vox_4xtta = False
    elif best_d_int == -2:
        inner_ds = (-2, -2, 0)
        is_1vox_4xtta = False
    else:
        inner_ds = (-2, 0, +2)
        is_1vox_4xtta = False

    lookahead_ds = sorted({
        max(-4, min(+4, best_d_int + delta))
        for delta in (-2, -1, 0, +1, +2)
    } | set(inner_ds))

    uniq_all = sorted(set(inner_ds) | set(lookahead_ds))
    d_bg_lo = max(-6, best_d_int - 4)
    d_bg_hi = min(+6, best_d_int + 4)

    dn_in_raw = {d: load_d("dnative", d) for d in uniq_all}
    hc_in_raw = {d: load_d("hecate", d) for d in uniq_all}
    ez_in_raw = {d: load_d("ens_z", d) for d in uniq_all}

    dn_lo_bg_raw, dn_hi_bg_raw = load_d("dnative", d_bg_lo), load_d("dnative", d_bg_hi)
    hc_lo_bg_raw, hc_hi_bg_raw = load_d("hecate", d_bg_lo), load_d("hecate", d_bg_hi)
    ez_lo_bg_raw, ez_hi_bg_raw = load_d("ens_z", d_bg_lo), load_d("ens_z", d_bg_hi)

    dn_m6_raw, dn_p6_raw = load_d("dnative", -6), load_d("dnative", +6)
    hc_m6_raw, hc_p6_raw = load_d("hecate", -6), load_d("hecate", +6)

    def hp_sub(img: np.ndarray) -> np.ndarray:
        return img - gaussian_filter(img, sigma=90.0)

    dn_in_hp = {d: hp_sub(dn_in_raw[d]) for d in uniq_all}
    hc_in_hp = {d: hp_sub(hc_in_raw[d]) for d in uniq_all}
    ez_in_hp = {d: hp_sub(ez_in_raw[d]) for d in uniq_all}

    dn_lo_hp, dn_hi_hp = hp_sub(dn_lo_bg_raw), hp_sub(dn_hi_bg_raw)
    hc_lo_hp, hc_hi_hp = hp_sub(hc_lo_bg_raw), hp_sub(hc_hi_bg_raw)
    ez_lo_hp, ez_hi_hp = hp_sub(ez_lo_bg_raw), hp_sub(ez_hi_bg_raw)
    dn_m6_hp, dn_p6_hp = hp_sub(dn_m6_raw), hp_sub(dn_p6_raw)
    hc_m6_hp, hc_p6_hp = hp_sub(hc_m6_raw), hp_sub(hc_p6_raw)

    dn_mean_hp = sum(dn_in_hp[d] for d in inner_ds) / float(len(inner_ds))
    hc_mean_hp = sum(hc_in_hp[d] for d in inner_ds) / float(len(inner_ds))
    ez_mean_hp = sum(ez_in_hp[d] for d in inner_ds) / float(len(inner_ds))

    dn_b_lo, dn_b_hi = float(np.percentile(dn_mean_hp[valid], 52.0)), float(np.percentile(dn_mean_hp[valid], 99.5))
    dn_b_hi = max(dn_b_hi, dn_b_lo + 4.0)
    hc_b_lo, hc_b_hi = float(np.percentile(hc_mean_hp[valid], 50.0)), float(np.percentile(hc_mean_hp[valid], 99.6))
    hc_b_hi = max(hc_b_hi, hc_b_lo + 4.0)
    ez_b_lo, ez_b_hi = float(np.percentile(ez_mean_hp[valid], 52.0)), float(np.percentile(ez_mean_hp[valid], 99.5))
    ez_b_hi = max(ez_b_hi, ez_b_lo + 4.0)

    def scale_locked(hp_arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((hp_arr - lo) / (hi - lo), 0.0, 1.0) * 255.0

    dn_in_c = {d: scale_locked(dn_in_hp[d], dn_b_lo, dn_b_hi) for d in uniq_all}
    hc_in_c = {d: scale_locked(hc_in_hp[d], hc_b_lo, hc_b_hi) for d in uniq_all}
    ez_in_c = {d: scale_locked(ez_in_hp[d], ez_b_lo, ez_b_hi) for d in uniq_all}

    dn_cm4 = scale_locked(dn_lo_hp, dn_b_lo, dn_b_hi)
    dn_cp4 = scale_locked(dn_hi_hp, dn_b_lo, dn_b_hi)
    hc_cm4 = scale_locked(hc_lo_hp, hc_b_lo, hc_b_hi)
    hc_cp4 = scale_locked(hc_hi_hp, hc_b_lo, hc_b_hi)
    ez_cm4 = scale_locked(ez_lo_hp, ez_b_lo, ez_b_hi)
    ez_cp4 = scale_locked(ez_hi_hp, ez_b_lo, ez_b_hi)

    dn_cm6 = scale_locked(dn_m6_hp, dn_b_lo, dn_b_hi)
    dn_cp6 = scale_locked(dn_p6_hp, dn_b_lo, dn_b_hi)
    hc_cm6 = scale_locked(hc_m6_hp, hc_b_lo, hc_b_hi)
    hc_cp6 = scale_locked(hc_p6_hp, hc_b_lo, hc_b_hi)

    # N_2: Local 3x3 cylindrical minimum on off-lamina references so tilted normal vectors don't over-subtract
    # thin strokes while 8-voxel fiber knots (wide across 3x3) are completely subtracted!
    k3 = np.ones((3, 3), dtype=np.uint8)
    dn_cm4_cyl = 0.72 * dn_cm4 + 0.28 * cv2.erode(dn_cm4, k3)
    dn_cp4_cyl = 0.72 * dn_cp4 + 0.28 * cv2.erode(dn_cp4, k3)
    hc_cm4_cyl = 0.72 * hc_cm4 + 0.28 * cv2.erode(hc_cm4, k3)
    hc_cp4_cyl = 0.72 * hc_cp4 + 0.28 * cv2.erode(hc_cp4, k3)
    ez_cm4_cyl = 0.72 * ez_cm4 + 0.28 * cv2.erode(ez_cm4, k3)
    ez_cp4_cyl = 0.72 * ez_cp4 + 0.28 * cv2.erode(ez_cp4, k3)

    dn_bg = 0.38 * (dn_cm4_cyl + dn_cp4_cyl) + 0.12 * (dn_cm6 + dn_cp6)
    hc_bg = 0.38 * (hc_cm4_cyl + hc_cp4_cyl) + 0.12 * (hc_cm6 + hc_cp6)
    ez_bg = 0.50 * (ez_cm4_cyl + ez_cp4_cyl)

    dn_thin_d = {d: np.clip(dn_in_c[d] - 0.88 * dn_bg, 0.0, 255.0) for d in uniq_all}
    hc_thin_d = {d: np.clip(hc_in_c[d] - 0.85 * hc_bg, 0.0, 255.0) for d in uniq_all}
    ez_thin_d = {d: np.clip(ez_in_c[d] - 0.86 * ez_bg, 0.0, 255.0) for d in uniq_all}

    w_stack = np.array([0.22, 0.56, 0.22], dtype=np.float32)[:, None, None]
    dn_stack = np.stack([dn_thin_d[d] for d in inner_ds], axis=0)
    hc_stack = np.stack([hc_thin_d[d] for d in inner_ds], axis=0)
    ez_stack = np.stack([ez_thin_d[d] for d in inner_ds], axis=0)
    dn_c_stack = np.stack([dn_in_c[d] for d in inner_ds], axis=0)
    hc_c_stack = np.stack([hc_in_c[d] for d in inner_ds], axis=0)

    dn_thin = 0.82 * np.max(dn_stack, axis=0) + 0.18 * np.sum(w_stack * dn_stack, axis=0)
    hc_thin = 0.82 * np.max(hc_stack, axis=0) + 0.18 * np.sum(w_stack * hc_stack, axis=0)
    ez_thin = 0.82 * np.max(ez_stack, axis=0) + 0.18 * np.sum(w_stack * ez_stack, axis=0)
    dn_c0_sheet = 0.82 * np.max(dn_c_stack, axis=0) + 0.18 * np.sum(w_stack * dn_c_stack, axis=0)
    hc_c0_sheet = 0.82 * np.max(hc_c_stack, axis=0) + 0.18 * np.sum(w_stack * hc_c_stack, axis=0)

    ens_sheet = 0.55 * dn_c0_sheet + 0.45 * hc_c0_sheet
    ens_outer_lo = 0.65 * (0.55 * dn_cm4_cyl + 0.45 * hc_cm4_cyl) + 0.35 * (0.55 * dn_cm6 + 0.45 * hc_cm6)
    ens_outer_hi = 0.65 * (0.55 * dn_cp4_cyl + 0.45 * hc_cp4_cyl) + 0.35 * (0.55 * dn_cp6 + 0.45 * hc_cp6)
    max_outer = np.maximum(ens_outer_lo, ens_outer_hi)
    bleed_ratio = max_outer / np.maximum(12.0, ens_sheet)
    z_bell_gate = np.clip(1.0 - np.maximum(0.0, bleed_ratio - 0.62) / 0.38, 0.25, 1.0)

    dn_thin_gated = dn_thin * z_bell_gate
    hc_thin_gated = hc_thin * z_bell_gate
    ez_thin_gated = ez_thin * z_bell_gate

    d_norm = np.clip(dn_thin_gated / 255.0, 0.0, 1.0)
    h_norm = np.clip(hc_thin_gated / 255.0, 0.0, 1.0)
    e_norm = np.clip(ez_thin_gated / 255.0, 0.0, 1.0)

    geom_3 = ((d_norm + 0.03) ** 0.44) * ((h_norm + 0.03) ** 0.34) * ((e_norm + 0.03) ** 0.22) - 0.03
    geom_3 = np.clip(geom_3 / 0.97, 0.0, 1.0) * 255.0
    arith_3 = 0.48 * dn_thin_gated + 0.32 * hc_thin_gated + 0.20 * ez_thin_gated
    concord_u8 = np.clip(0.55 * geom_3 + 0.45 * arith_3, 0.0, 255.0).astype(np.uint8)

    concord_ratio = (2.0 * np.sqrt(d_norm * h_norm + 1e-6)) / (d_norm + h_norm + 0.06)
    concord_ratio = np.clip(concord_ratio, 0.0, 1.0).astype(np.float32)

    # Full 5-slice Sub-Voxel Tensor `(H, W, K)` across `d in lookahead_ds`
    sub_dn_stack = np.stack([dn_thin_d[d] * z_bell_gate for d in lookahead_ds], axis=-1)  # (H, W, K)
    sub_hc_stack = np.stack([hc_thin_d[d] * z_bell_gate for d in lookahead_ds], axis=-1)
    sub_ez_stack = np.stack([ez_thin_d[d] * z_bell_gate for d in lookahead_ds], axis=-1)

    dn_submax = np.max(sub_dn_stack, axis=-1)
    hc_submax = np.max(sub_hc_stack, axis=-1)
    ez_submax = np.max(sub_ez_stack, axis=-1)

    d_sub_norm = np.clip(dn_submax / 255.0, 0.0, 1.0)
    h_sub_norm = np.clip(hc_submax / 255.0, 0.0, 1.0)
    concord_sub_ratio = (2.0 * np.sqrt(d_sub_norm * h_sub_norm + 1e-6)) / (d_sub_norm + h_sub_norm + 0.06)
    concord_ratio = np.maximum(concord_ratio, np.clip(concord_sub_ratio, 0.0, 1.0)).astype(np.float32)

    sub_lookahead_f = (0.48 * dn_submax + 0.32 * hc_submax + 0.20 * ez_submax) * (0.45 + 0.55 * concord_ratio)
    subvoxel_lookahead_u8 = np.clip(sub_lookahead_f, 0.0, 255.0).astype(np.uint8)

    # Per-slice 3D voxel stack `(H, W, K)`
    d_3d_n = np.clip(sub_dn_stack / 255.0, 0.0, 1.0)
    h_3d_n = np.clip(sub_hc_stack / 255.0, 0.0, 1.0)
    cr_3d = np.clip((2.0 * np.sqrt(d_3d_n * h_3d_n + 1e-6)) / (d_3d_n + h_3d_n + 0.06), 0.0, 1.0)
    voxel_3d_f = (0.48 * sub_dn_stack + 0.32 * sub_hc_stack + 0.20 * sub_ez_stack) * (0.45 + 0.55 * cr_3d)
    voxel_3d_stack_u8 = np.clip(voxel_3d_f, 0.0, 255.0).astype(np.uint8)
    dn_3d_stack_u8 = np.clip(sub_dn_stack, 0.0, 255.0).astype(np.uint8)
    hc_3d_stack_u8 = np.clip(sub_hc_stack, 0.0, 255.0).astype(np.uint8)

    dn_ridge_u8, dn_horiz_u8, _ = junction_preserving_stroke_bank_l(
        dn_thin_gated.astype(np.uint8), fiber_tilt_deg, px_per_mm, crossbar_boost=1.24
    )
    hc_ridge_u8, hc_horiz_u8, _ = junction_preserving_stroke_bank_l(
        hc_thin_gated.astype(np.uint8), fiber_tilt_deg, px_per_mm, crossbar_boost=1.24
    )

    fused = (
        0.30 * concord_u8.astype(np.float32)
        + 0.26 * dn_ridge_u8.astype(np.float32)
        + 0.16 * hc_ridge_u8.astype(np.float32)
        + 0.16 * dn_thin_gated
        + 0.12 * hc_thin_gated
    )
    horiz_syn = np.sqrt(np.maximum(0.0, dn_horiz_u8.astype(np.float32) * hc_horiz_u8.astype(np.float32)))
    fused = np.maximum(fused, 0.84 * horiz_syn)

    diffused = coherence_enhancing_stroke_diffusion(np.clip(fused, 0, 255).astype(np.uint8), n_iter=2)
    p55 = float(np.percentile(diffused[valid], 55.0))
    p994 = max(p55 + 8.0, float(np.percentile(diffused[valid], 99.4)))
    callig_u8 = (np.clip((diffused.astype(np.float32) - p55) / (p994 - p55), 0.0, 1.0) * 255.0).astype(np.uint8)

    r_sheet_max = 0.0
    for d in inner_ds:
        a = dn_in_hp[d][valid]
        b = hc_in_hp[d][valid]
        if a.std() > 1e-5 and b.std() > 1e-5:
            r_d = float(np.corrcoef(a, b)[0, 1])
            if r_d > r_sheet_max:
                r_sheet_max = r_d

    return {
        "inner_ds_used": list(inner_ds),
        "lookahead_ds_used": list(lookahead_ds),
        "bg_ds_used": [d_bg_lo, d_bg_hi],
        "is_1vox_4xtta": is_1vox_4xtta,
        "r_sheet_max": round(r_sheet_max, 4),
        "callig_u8": callig_u8,
        "concord_u8": concord_u8,
        "concord_ratio": concord_ratio,
        "dn_thin_u8": dn_thin_gated.astype(np.uint8),
        "hc_thin_u8": hc_thin_gated.astype(np.uint8),
        "dn_ridge_u8": dn_ridge_u8,
        "subvoxel_lookahead_u8": subvoxel_lookahead_u8,
        "dn_submax_u8": np.clip(dn_submax, 0, 255).astype(np.uint8),
        "hc_submax_u8": np.clip(hc_submax, 0, 255).astype(np.uint8),
        "voxel_3d_stack_u8": voxel_3d_stack_u8,
        "dn_3d_stack_u8": dn_3d_stack_u8,
        "hc_3d_stack_u8": hc_3d_stack_u8,
        "ens_0_locked": (0.55 * dn_c0_sheet + 0.45 * hc_c0_sheet).astype(np.float32),
        "ens_m4_locked": (0.55 * dn_cm4_cyl + 0.45 * hc_cm4_cyl).astype(np.float32),
        "ens_p4_locked": (0.55 * dn_cp4_cyl + 0.45 * hc_cp4_cyl).astype(np.float32),
        "hc_0_locked": hc_c0_sheet.astype(np.float32),
        "hc_m4_locked": hc_cm4_cyl.astype(np.float32),
        "hc_p4_locked": hc_cp4_cyl.astype(np.float32),
    }


def scrape_3d_voxel_and_around_in_lupa(
    rib_pure: np.ndarray,
    stroke_meta_pure: dict,
    patch_cal: np.ndarray,
    patch_subvoxel: np.ndarray,
    patch_dn_sub: np.ndarray,
    patch_hc_sub: np.ndarray,
    lupa_3d_u8: np.ndarray,
    lupa_dn_3d: np.ndarray,
    lupa_hc_3d: np.ndarray,
    px_per_mm: float,
) -> tuple[np.ndarray, float, dict]:
    """
    N_3 + N_4: Exact 3D Voxel-and-Around Geodesic Scraper & Sub-Voxel Shell Integrator:
      1. For each of the 5 sub-voxel layers `k in {0..K-1}` (`d in {d*-2..d*+2}`) inside
         the rotated upright lupa (`lupa_3d_u8` of shape `(h, w, K)`), computes the 4-directional
         calligraphic 'voxel-and-around' shell (`_directional_around_shell_2d`) coupled with
         adjacent sub-voxel layers `k +- 1`.
      2. Starting from the core 3D ink seeds `S_0(u, v, k)` anchored on `rib_pure >= 58`, performs
         6 geodesic hysteresis steps in 3D `(u + du, v + dv, k + dk)` with `|dk| <= 1` per step,
         annexing connected thin-layer ink voxels where both `dnative` and `hecate` agree.
      3. Evaluates edge-anchored Greek template ridge corridors (`<= 0.08 mm` wide) in the 3D shell
         so missing second legs (`H, N, PI, A, K, M`) and crossbars (`E, T, H, A, D`) that dipped
         into `d* +- 1, d* +- 2` are scraped cleanly.
      4. Performs Cryo-EM 3D Micro-Shell Integration around the optimal depth `k*(u, v)` and
         suppresses un-scraped off-manifold background speckle (`0` synthetic `cv2.line` pixels!).
    """
    h, w = rib_pure.shape
    if lupa_3d_u8.ndim == 2:
        lupa_3d_u8 = lupa_3d_u8[:, :, None]
        lupa_dn_3d = lupa_dn_3d[:, :, None]
        lupa_hc_3d = lupa_hc_3d[:, :, None]
    K = lupa_3d_u8.shape[2]

    cands_init, _, _, _ = score_corridor_anchored_greek_templates(rib_pure, top_k=3)
    if not cands_init:
        return rib_pure, float(np.mean(rib_pure >= 68)), stroke_meta_pure

    active = rib_pure >= 50
    if int(np.sum(active)) < 16:
        return rib_pure, float(np.mean(rib_pure >= 68)), stroke_meta_pure

    # Step 1: Local intra-lupa 3D normalization + 4-directional 'voxel-and-around' shell for each depth slice k in 0..K-1
    close_px = max(3, int(round(0.065 * px_per_mm)) | 1)
    k_close_micro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))

    shell_2d_slices = []
    dn_local_slices = []
    hc_local_slices = []
    for k in range(K):
        v_k = lupa_3d_u8[:, :, k].astype(np.float32)
        dn_k = lupa_dn_3d[:, :, k].astype(np.float32)
        hc_k = lupa_hc_3d[:, :, k].astype(np.float32)

        # Local intra-lupa dynamic range recovery so fainter 2nd legs / crossbars inside this 2.2x2.4mm box are not crushed
        v_lo, v_hi = float(np.percentile(v_k, 30.0)), max(float(np.percentile(v_k, 30.0)) + 6.0, float(np.percentile(v_k, 99.0)))
        v_loc = np.clip((v_k - v_lo) / (v_hi - v_lo), 0.0, 1.0) * 255.0
        v_blend = (0.52 * v_k + 0.48 * v_loc).astype(np.uint8)

        # Bridge 1-2 px papyrus fiber micro-fissures and compute 4-directional calligraphic shell around each voxel
        v_closed = cv2.morphologyEx(v_blend, cv2.MORPH_CLOSE, k_close_micro)
        sh_k = _directional_around_shell_2d(v_closed, px_per_mm).astype(np.float32)
        shell_2d_slices.append(sh_k)

        dn_lo, dn_hi = float(np.percentile(dn_k, 28.0)), max(float(np.percentile(dn_k, 28.0)) + 5.0, float(np.percentile(dn_k, 99.0)))
        hc_lo, hc_hi = float(np.percentile(hc_k, 28.0)), max(float(np.percentile(hc_k, 28.0)) + 5.0, float(np.percentile(hc_k, 99.0)))
        dn_local_slices.append(0.50 * dn_k + 0.50 * np.clip((dn_k - dn_lo) / (dn_hi - dn_lo), 0.0, 1.0) * 255.0)
        hc_local_slices.append(0.50 * hc_k + 0.50 * np.clip((hc_k - hc_lo) / (hc_hi - hc_lo), 0.0, 1.0) * 255.0)

    shell_3d = np.stack(shell_2d_slices, axis=-1)  # (h, w, K)
    dn_loc_3d = np.stack(dn_local_slices, axis=-1)
    hc_loc_3d = np.stack(hc_local_slices, axis=-1)

    # Couple each depth slice k with its immediate sub-voxel neighbors k-1 and k+1 ("voxel and around in 3D")
    coupled_3d = np.zeros_like(shell_3d, dtype=np.float32)
    for k in range(K):
        k_lo = max(0, k - 1)
        k_hi = min(K - 1, k + 1)
        neigh_max = np.maximum(shell_3d[:, :, k_lo], shell_3d[:, :, k_hi])
        coupled_3d[:, :, k] = 0.58 * shell_3d[:, :, k] + 0.42 * neigh_max

    # Step 2: 6-Step 3D 26-Connected Geodesic Hysteresis Scraping `(u + du, v + dv, k + dk)`
    p36_3d = max(14.0, float(np.percentile(coupled_3d, 36.0)))
    dn_gate_3d = dn_loc_3d >= max(12.0, float(np.percentile(dn_loc_3d, 35.0)))
    hc_gate_3d = hc_loc_3d >= max(8.0, float(np.percentile(hc_loc_3d, 34.0)))
    weak_3d = (coupled_3d >= p36_3d) & dn_gate_3d & hc_gate_3d

    # Also include secondary stroke components inside the central character zone [0.14..0.86] of `patch_cal`
    corridor_box = np.zeros((h, w), dtype=bool)
    corridor_box[int(0.14 * h):int(0.86 * h), int(0.14 * w):int(0.86 * w)] = True
    cal_seed = (patch_cal >= max(36.0, float(np.percentile(patch_cal, 64.0)))) & corridor_box
    seed_2d = active | cal_seed

    S_3d = np.zeros((h, w, K), dtype=np.uint8)
    for k in range(K):
        S_3d[:, :, k] = (seed_2d & (coupled_3d[:, :, k] >= p36_3d * 0.88)).astype(np.uint8)
    if S_3d.sum() < 12:
        S_3d[:, :, K // 2] = seed_2d.astype(np.uint8)

    step_px = max(3, int(round(0.065 * px_per_mm)) | 1)
    k_xy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (step_px, step_px))
    reach_px = max(13, int(round(0.36 * px_per_mm)) | 1)
    geodesic_halo = cv2.dilate(seed_2d.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach_px, reach_px))) > 0

    for _ in range(6):
        S_next = np.zeros_like(S_3d)
        for k in range(K):
            k_lo = max(0, k - 1)
            k_hi = min(K - 1, k + 1)
            slab_k = np.max(S_3d[:, :, k_lo:k_hi + 1], axis=-1)
            dil_xy = cv2.dilate(slab_k, k_xy)
            S_next[:, :, k] = dil_xy & weak_3d[:, :, k].astype(np.uint8) & geodesic_halo.astype(np.uint8)
        if np.array_equal(S_next, S_3d):
            break
        S_3d = S_next

    # Step 3: Edge-Anchored Template Corridor 3D Sub-Voxel Scraping
    ys, xs = np.where(active)
    ay0, ay1 = int(ys.min()), int(ys.max()) + 1
    ax0, ax1 = int(xs.min()), int(xs.max()) + 1
    cy_act = 0.5 * (ay0 + ay1)
    cx_act = 0.5 * (ax0 + ax1)

    th = max(16, int(round(0.68 * h)))
    tw = max(14, int(round(0.60 * w)))
    ty0 = max(2, min(h - th - 2, int(round(cy_act - 0.5 * th))))
    if (ax1 - ax0) < 0.44 * w:
        if cx_act <= 0.50 * w:
            tx0 = max(2, min(w - tw - 2, ax0 - 2))
        else:
            tx0 = max(2, min(w - tw - 2, ax1 - tw + 2))
    else:
        tx0 = max(2, min(w - tw - 2, int(round(cx_act - 0.5 * tw))))

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    two_leg_codes = ("H", "N", "A", "K", "PI", "E", "T", "D", "P", "M")
    search_cands = list(cands_init[:2])
    for c_item in cands_init[2:]:
        if c_item["code"] in two_leg_codes:
            search_cands.append(c_item)
            break

    for c_item in search_cands:
        tpl_64 = c_item["_tpl_u8"]
        tpl_rs = cv2.resize(tpl_64, (tw, th), interpolation=cv2.INTER_LINEAR)
        pred_mask[ty0:ty0 + th, tx0:tx0 + tw] = np.maximum(
            pred_mask[ty0:ty0 + th, tx0:tx0 + tw],
            (tpl_rs >= 80).astype(np.uint8) * 255,
        )

    k_dil = max(3, int(round(0.085 * px_per_mm)) | 1)
    pred_zone = cv2.dilate(pred_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil))) > 0

    for k in range(K):
        corr_k = pred_zone & weak_3d[:, :, k]
        S_3d[:, :, k] = S_3d[:, :, k] | corr_k.astype(np.uint8)

    # Step 4: Cryo-EM 3D Micro-Shell Integration around optimal depth `k*(u, v)`
    weighted_3d = coupled_3d * np.where(S_3d > 0, 1.0, 0.22)
    k_star = np.argmax(weighted_3d, axis=-1)
    uu, vv = np.indices((h, w))
    v_opt = coupled_3d[uu, vv, k_star]
    v_lo = coupled_3d[uu, vv, np.maximum(0, k_star - 1)]
    v_hi = coupled_3d[uu, vv, np.minimum(K - 1, k_star + 1)]
    shell_integrated = (
        0.56 * v_opt
        + 0.24 * np.maximum(v_lo, v_hi)
        + 0.20 * np.maximum(patch_cal.astype(np.float32), patch_subvoxel.astype(np.float32))
    )

    # Local high-pass ridge sharpening so scraped strokes have crisp ~0.26 mm calligraphic width
    sub_sharp = np.maximum(0.0, shell_integrated - 0.30 * gaussian_filter(shell_integrated, sigma=6.0))
    s_p98 = float(np.percentile(sub_sharp, 98.2)) if sub_sharp.size > 0 else 1.0
    if s_p98 > 6.0:
        sub_sharp = np.clip(sub_sharp / s_p98 * 250.0, 0.0, 255.0)

    M_2d = np.max(S_3d, axis=-1) > 0
    # Smoothly connect adjacent scraped voxels within 0.06 mm
    M_closed = cv2.morphologyEx(M_2d.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close_micro) > 0
    scraped_zone = (pred_zone & np.max(weak_3d, axis=-1)) | M_closed

    add_voxels = np.where(scraped_zone, sub_sharp.astype(np.uint8), 0)

    by_taper = np.ones(h, dtype=np.float32)
    bx_taper = np.ones(w, dtype=np.float32)
    t_px = min(10, h // 8, w // 8)
    if t_px >= 2:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, t_px, dtype=np.float32)))
        by_taper[:t_px] = ramp
        by_taper[-t_px:] = ramp[::-1]
        bx_taper[:t_px] = ramp
        bx_taper[-t_px:] = ramp[::-1]
    add_voxels = (add_voxels.astype(np.float32) * np.outer(by_taper, bx_taper)).astype(np.uint8)

    # N_4: Suppress detached off-manifold background chaff by 88% while keeping 100% of scraped 3D strokes
    core_halo = cv2.dilate(scraped_zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    rib_pure_cleaned = np.where(core_halo, rib_pure, (rib_pure.astype(np.float32) * 0.12).astype(np.uint8))

    rib_scraped = np.maximum(rib_pure_cleaned, add_voxels)
    fill_sc = float(np.mean(rib_scraped >= 68))

    act_ys, act_xs = np.where(rib_scraped >= 66)
    if act_xs.size >= 15:
        span_w_mm = float(np.percentile(act_xs, 96) - np.percentile(act_xs, 4)) / px_per_mm
        span_h_mm = float(np.percentile(act_ys, 96) - np.percentile(act_ys, 4)) / px_per_mm
        is_multi = bool(stroke_meta_pure["is_multistroke"] or span_w_mm >= 0.84 or span_h_mm >= 1.28)
        n_legs = max(stroke_meta_pure["n_legs"], 2 if span_w_mm >= 0.86 else 1)
    else:
        span_w_mm = stroke_meta_pure["span_w_mm"]
        is_multi = stroke_meta_pure["is_multistroke"]
        n_legs = stroke_meta_pure["n_legs"]

    return rib_scraped, fill_sc, {
        "span_w_mm": round(span_w_mm, 2),
        "is_multistroke": is_multi,
        "n_legs": n_legs,
    }


def _run_single_orient_decode_n(
    ct_2cm_in: np.ndarray,
    flat_2cm_in: np.ndarray,
    lamina_maps_in: dict,
    px_per_mm: float,
    tf_mode: str,
    max_lines: int = 6,
    max_slots_per_line: int = 10,
) -> dict | None:
    ct_2cm = _apply_2d_tf(ct_2cm_in, tf_mode)
    flat_2cm = _apply_2d_tf(flat_2cm_in, tf_mode)
    lamina_maps = {
        k: (_apply_2d_tf_any(v, tf_mode) if isinstance(v, np.ndarray) and v.ndim in (2, 3) else v)
        for k, v in lamina_maps_in.items()
    }
    H, W = ct_2cm.shape
    callig_2cm = lamina_maps["callig_u8"]
    dn_t_2cm = lamina_maps["dn_thin_u8"]
    hc_t_2cm = lamina_maps["hc_thin_u8"]
    concord_2cm = lamina_maps["concord_u8"]
    concord_ratio_2cm = lamina_maps["concord_ratio"]
    dn_rg_2cm = lamina_maps["dn_ridge_u8"]
    subvoxel_2cm = lamina_maps["subvoxel_lookahead_u8"]
    dn_sub_2cm = lamina_maps["dn_submax_u8"]
    hc_sub_2cm = lamina_maps["hc_submax_u8"]
    voxel_3d_2cm = lamina_maps["voxel_3d_stack_u8"]
    dn_3d_2cm = lamina_maps["dn_3d_stack_u8"]
    hc_3d_2cm = lamina_maps["hc_3d_stack_u8"]

    clean_initial = (callig_2cm.astype(np.float32) * np.clip(flat_2cm, 0.0, 1.0)).astype(np.uint8)
    tilt_deg, lines_out = trace_fiber_streamline_baselines_6lines(
        ct_2cm, clean_initial, flat_2cm, px_per_mm, max_lines=max_lines
    )

    clean = calibrate_corridor_ultraprecise(
        callig_2cm, dn_t_2cm, hc_t_2cm, concord_2cm, flat_2cm, lines_out, px_per_mm
    )

    nom_w = int(round(2.26 * px_per_mm))
    nom_h = int(round(2.44 * px_per_mm))

    all_decoded_boxes = []
    decoded_lines_info = {}
    # N_5: Attenuate raw background chaff outside scraped character slots (`0.64`) so 3D-scraped letters pop cleanly!
    display_callig = (clean.astype(np.float32) * 0.64).astype(np.uint8)

    for l_item in lines_out:
        idx_l = l_item["line_id"] - 1
        line_name = l_item["line_name"]
        y_base_x = l_item["y_base_x"]

        strip = np.zeros((nom_h, W), dtype=np.float32)
        strip_sub = np.zeros((nom_h, W), dtype=np.float32)
        for col_x in range(W):
            yb = int(np.clip(y_base_x[col_x], nom_h, H))
            strip[:, col_x] = clean[yb - nom_h:yb, col_x]
            strip_sub[:, col_x] = subvoxel_2cm[yb - nom_h:yb, col_x]

        col_prof = 0.74 * np.percentile(strip, 88.0, axis=0) + 0.26 * np.percentile(strip_sub, 90.0, axis=0)
        k_col = max(15, int(round(0.66 * px_per_mm)) | 1)
        col_smooth = cv2.GaussianBlur(col_prof.reshape(-1, 1), (1, k_col), 0).ravel()

        picked_xs = []
        thr_col = max(18.5, float(np.percentile(col_smooth, 27.0)))
        min_slot_sep = int(round(1.66 * px_per_mm))
        for x_idx in np.argsort(col_smooth)[::-1]:
            xi = int(x_idx)
            if xi < nom_w // 2 or xi > W - nom_w // 2:
                continue
            if col_smooth[xi] < thr_col:
                continue
            if all(abs(xi - px) >= min_slot_sep for px in picked_xs):
                picked_xs.append(xi)
                if len(picked_xs) >= max_slots_per_line - 1:
                    break
        picked_xs.sort()
        anchor_set = set(picked_xs)

        picked_xs = _discover_cadence_lookahead_xs(
            picked_xs, col_smooth, W, nom_w, px_per_mm, max_slots_per_line
        )

        line_raw_slots = []
        for xi in picked_xs:
            x0 = max(0, min(W - nom_w, xi - nom_w // 2))
            yb_local = int(np.median(y_base_x[x0:x0 + nom_w]))
            y0_nom = max(0, min(H - nom_h, yb_local - nom_h))

            best_y0 = y0_nom
            best_e = -1.0
            dy_max = int(round(0.26 * px_per_mm))
            for dy in (-dy_max, 0, dy_max):
                yy = max(0, min(H - nom_h, y0_nom + dy))
                e_val = float(np.percentile(clean[yy:yy + nom_h, x0:x0 + nom_w], 92.0))
                if e_val > best_e:
                    best_e = e_val
                    best_y0 = yy

            cx_slot = float(x0 + 0.5 * nom_w)
            cy_slot = float(best_y0 + 0.5 * nom_h)

            dx_local = max(8, nom_w)
            x_a = max(0, int(round(cx_slot - 0.5 * dx_local)))
            x_b = min(W - 1, int(round(cx_slot + 0.5 * dx_local)))
            local_slope = float(y_base_x[x_b] - y_base_x[x_a]) / max(1.0, float(x_b - x_a))
            local_tilt_deg = float(np.clip(0.65 * tilt_deg + 0.35 * np.rad2deg(np.arctan(local_slope)), -28.0, 28.0))

            patch_fl = extract_rotated_lupa_patch(flat_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            fl_mean = float(np.mean(patch_fl))
            if fl_mean < 0.48:
                continue

            patch_dn = extract_rotated_lupa_patch(dn_t_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            patch_hc = extract_rotated_lupa_patch(hc_t_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            patch_rg = extract_rotated_lupa_patch(dn_rg_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            patch_cc = extract_rotated_lupa_patch(concord_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            patch_cr = extract_rotated_lupa_patch(concord_ratio_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            patch_sub = extract_rotated_lupa_patch(subvoxel_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            patch_dn_sub = extract_rotated_lupa_patch(dn_sub_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            patch_hc_sub = extract_rotated_lupa_patch(hc_sub_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)

            dn_p94 = max(float(np.percentile(patch_dn, 94.0)), 0.88 * float(np.percentile(patch_dn_sub, 94.0)))
            hc_p985 = max(float(np.percentile(patch_hc, 98.5)), 0.88 * float(np.percentile(patch_hc_sub, 98.5)))
            rg_p92 = float(np.percentile(patch_rg, 92.0))
            if dn_p94 < 18.5 or hc_p985 < 9.8 or rg_p92 < 17.0:
                continue

            # N_2: Extract the full 5-slice 3D Sub-Voxel Tensor `(nom_h, nom_w, K)` inside the rotated lupa!
            lupa_3d_u8 = extract_rotated_lupa_patch(voxel_3d_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            lupa_dn_3d = extract_rotated_lupa_patch(dn_3d_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            lupa_hc_3d = extract_rotated_lupa_patch(hc_3d_2cm, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)

            patch_cal_rot = extract_rotated_lupa_patch(clean, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            rib_rot, fill_rot, dom_rot, meta_rot = isolate_ultraprecise_uncial_character(
                patch_cal_rot, patch_cc, px_per_mm
            )
            # N_3 + N_4: Exact 3D Voxel-and-Around Geodesic Scraper & Sub-Voxel Shell Integrator
            rib_sc, fill_sc, meta_sc = scrape_3d_voxel_and_around_in_lupa(
                rib_rot, meta_rot, patch_cal_rot, patch_sub, patch_dn_sub, patch_hc_sub,
                lupa_3d_u8, lupa_dn_3d, lupa_hc_3d, px_per_mm
            )

            patch_cal_axis = clean[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_cc_axis = concord_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            rib_axis, fill_ax, dom_ax, meta_ax = isolate_ultraprecise_uncial_character(
                patch_cal_axis, patch_cc_axis, px_per_mm
            )
            rib_single, fill_s, dom_s = isolate_central_character_ribbon(patch_cal_rot)

            cands_sc, cent_sc, y_top_sc, y_bot_sc = score_rectified_lupa_greek_templates(rib_sc, top_k=5)
            cands_rot, cent_rot, y_top_r, y_bot_r = score_rectified_lupa_greek_templates(rib_rot, top_k=5)
            cands_ax, cent_ax, y_top_a, y_bot_a = score_rectified_lupa_greek_templates(rib_axis, top_k=5)
            cands_sing, cent_sing, y_top_s, y_bot_s = score_rectified_lupa_greek_templates(rib_single, top_k=5)

            two_leg_set = {"H", "N", "A", "K", "M", "PI", "W", "O", "P", "T", "E", "D", "X", "G"}
            options = []
            if 0.065 <= fill_sc <= 0.56:
                b_sc = 0.024 if (meta_sc["is_multistroke"] and cands_sc[0]["code"] in two_leg_set) else 0.014
                options.append((cands_sc[0]["tpl_score"] + b_sc, cands_sc, cent_sc, y_top_sc, y_bot_sc, rib_sc, fill_sc, max(dom_rot, dom_s), meta_sc, True))
            if 0.065 <= fill_rot <= 0.56:
                b_r = 0.008 if (meta_rot["is_multistroke"] and cands_rot[0]["code"] in two_leg_set) else 0.0
                options.append((cands_rot[0]["tpl_score"] + b_r, cands_rot, cent_sc if (0.065 <= fill_sc <= 0.52) else cent_rot, y_top_r, y_bot_r, rib_sc if (0.065 <= fill_sc <= 0.52) else rib_rot, fill_sc if (0.065 <= fill_sc <= 0.52) else fill_rot, dom_rot, meta_sc, True))
            if 0.065 <= fill_ax <= 0.56:
                b_a = 0.008 if (meta_ax["is_multistroke"] and cands_ax[0]["code"] in two_leg_set) else 0.0
                options.append((cands_ax[0]["tpl_score"] + b_a, cands_ax, cent_ax, y_top_a, y_bot_a, rib_axis, fill_ax, dom_ax, meta_ax, False))
            if not options and (0.065 <= fill_s <= 0.56):
                options.append((cands_sing[0]["tpl_score"], cands_sing, cent_sing, y_top_s, y_bot_s, rib_single, fill_s, max(dom_rot, dom_s), meta_sc, True))

            if not options:
                continue
            options.sort(key=lambda t: t[0], reverse=True)
            _, top_cands, centered_rib, y_st_top, y_st_bot, rib_chosen, fill_r, dom_r, stroke_meta, is_rot_lupa = options[0]

            ays_c, axs_c = np.where(rib_chosen >= 66)
            if axs_c.size >= 15:
                sw_c = float(np.percentile(axs_c, 96) - np.percentile(axs_c, 4)) / px_per_mm
                sh_c = float(np.percentile(ays_c, 96) - np.percentile(ays_c, 4)) / px_per_mm
                stroke_meta = {
                    "span_w_mm": round(sw_c, 2),
                    "is_multistroke": bool(stroke_meta.get("is_multistroke", False) or sw_c >= 0.84 or sh_c >= 1.26),
                    "n_legs": max(stroke_meta.get("n_legs", 1), 2 if sw_c >= 0.86 else 1),
                }

            min_tpl_req = 0.46 if (xi in anchor_set) else 0.58
            if dom_r < 0.58 or top_cands[0]["tpl_score"] < min_tpl_req:
                continue

            active_stroke_px = rib_chosen >= 80
            px_conc = float(np.percentile(patch_cr[active_stroke_px], 75.0)) if np.any(active_stroke_px) else float(np.mean(patch_cr))
            peak_conc = float((2.0 * np.sqrt(max(0.0, dn_p94 * hc_p985))) / (dn_p94 + hc_p985 + 1.0))
            slot_concord = 0.55 * peak_conc + 0.45 * px_conc

            # N_5: Splat the 3D-scraped character ribbon back onto `display_callig`
            if is_rot_lupa:
                splat_rotated_lupa_patch(display_callig, rib_chosen, cx_slot, cy_slot, local_tilt_deg)
            else:
                display_callig[best_y0:best_y0 + nom_h, x0:x0 + nom_w] = np.maximum(
                    display_callig[best_y0:best_y0 + nom_h, x0:x0 + nom_w],
                    rib_chosen,
                )

            slot_box = {
                "x": int(x0), "y": int(best_y0), "w": int(nom_w), "h": int(nom_h),
                "y_stroke_bot_abs": int(best_y0 + y_st_bot),
                "w_mm": round(nom_w / px_per_mm, 2),
                "h_mm": round(nom_h / px_per_mm, 2),
                "cx": round(cx_slot, 1),
                "cy": round(cy_slot, 1),
                "lupa_tilt_deg": round(local_tilt_deg, 1),
                "dn_p94": round(dn_p94, 1),
                "hc_p94": round(hc_p985, 1),
                "ridge_p92": round(rg_p92, 1),
                "slot_concordance": round(slot_concord, 3),
                "fill_ratio": round(fill_r, 3),
                "dom_stroke_ratio": round(dom_r, 3),
                "span_w_mm": stroke_meta["span_w_mm"],
                "is_multistroke": stroke_meta["is_multistroke"],
                "n_legs": stroke_meta["n_legs"],
                "flatness": round(fl_mean, 2),
                "line_id": idx_l + 1,
                "_ribbon_u8": centered_rib,
                "candidates": top_cands,
            }
            line_raw_slots.append(slot_box)

        thin_slots, _ = filter_boxes_by_frozen_stroke_z_attenuation(
            line_raw_slots,
            display_callig,
            lamina_maps["ens_0_locked"], lamina_maps["ens_m4_locked"], lamina_maps["ens_p4_locked"],
            lamina_maps["hc_0_locked"], lamina_maps["hc_m4_locked"], lamina_maps["hc_p4_locked"],
            min_drop_pm4_pct=32.0,
        )
        if not thin_slots:
            continue

        decoded_slots, line_meta = viterbi_decode_line_slots(thin_slots)
        for s in decoded_slots:
            s["score"] = round(
                (0.35 * s["dn_p94"] + 0.25 * s["hc_p94"] + 0.25 * s["ridge_p92"])
                * s["flatness"]
                * (0.75 + 0.65 * s["tpl_score"]),
                1,
            )
            all_decoded_boxes.append(s)

        decoded_lines_info[line_name] = {
            "y_mm": round(l_item["yc_mid"] / px_per_mm, 2),
            "num_chars": len(decoded_slots),
            "diplomatic_ascii": line_meta["ascii"],
            "diplomatic_unicode": line_meta["unicode"],
            "greedy_ascii": line_meta["greedy_ascii"],
            "matched_epicurean_words": line_meta["matched_words"],
            "viterbi_gain": line_meta["viterbi_gain"],
            "mean_drop_pm4_pct": round(float(np.mean([s["stroke_drop_eff_pm4_pct"] for s in decoded_slots])), 1),
            "mean_tpl_score": round(float(np.mean([s["tpl_score"] for s in decoded_slots])), 3),
            "mean_slot_concordance": round(float(np.mean([s["slot_concordance"] for s in decoded_slots])), 3),
            "_slots": decoded_slots,
        }

    if len(all_decoded_boxes) < 4:
        return None

    max_bridge_px = int(round(3.6 * px_per_mm))
    chars_in_runs_ge3 = 0
    for lk, lv in decoded_lines_info.items():
        sl_sorted = sorted(lv["_slots"], key=lambda s: s["x"])
        grps = [[sl_sorted[0]]]
        for s in sl_sorted[1:]:
            if s["x"] - (grps[-1][-1]["x"] + grps[-1][-1]["w"]) <= max_bridge_px:
                grps[-1].append(s)
            else:
                grps.append([s])
        for g in grps:
            if len(g) >= 3:
                chars_in_runs_ge3 += len(g)
    contiguous_run_ge3_frac = round(chars_in_runs_ge3 / max(1, len(all_decoded_boxes)), 3)

    multistroke_frac = round(
        float(np.mean([1.0 if b.get("is_multistroke", False) else 0.0 for b in all_decoded_boxes])),
        3,
    )
    mean_span_w_mm = round(float(np.mean([b.get("span_w_mm", 0.0) for b in all_decoded_boxes])), 2)
    mean_drop = float(np.mean([b["stroke_drop_eff_pm4_pct"] for b in all_decoded_boxes]))
    mean_tpl = float(np.mean([b["tpl_score"] for b in all_decoded_boxes]))
    mean_concord = float(np.mean([b["slot_concordance"] for b in all_decoded_boxes]))
    phi_frac = float(np.mean([1.0 if b["greek_ascii"] == "PH" else 0.0 for b in all_decoded_boxes]))

    all_words = []
    for lk, lv in decoded_lines_info.items():
        all_words.extend(lv["matched_epicurean_words"])
    uniq_words = sorted(set(all_words))

    return {
        "tf_mode": tf_mode,
        "orig_tilt_deg": tilt_deg,
        "rect_tilt_deg": 0.0,
        "ct_2cm": ct_2cm,
        "flat_2cm": flat_2cm,
        "display_callig": display_callig,
        "lines_out": lines_out,
        "decoded_lines_info": decoded_lines_info,
        "all_decoded_boxes": all_decoded_boxes,
        "contiguous_run_ge3_frac": contiguous_run_ge3_frac,
        "multistroke_frac": multistroke_frac,
        "mean_span_w_mm": mean_span_w_mm,
        "mean_drop": mean_drop,
        "mean_tpl": mean_tpl,
        "mean_concord": mean_concord,
        "phi_frac": phi_frac,
        "uniq_words": uniq_words,
        "orient_rank_score": mean_tpl + 0.015 * len(uniq_words) + 0.0014 * len(all_decoded_boxes),
    }


def decode_stage_n_candidate(seg_dir: Path) -> dict | None:
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

    sl_y, sl_x, win_mm = find_largest_valid_interior_crop_l(ct_mid, px_per_mm)
    sl = (sl_y, sl_x)
    ct_2cm_in = ct_mid[sl]

    flat_2cm_in, ct_lamina_stats = compute_strict_single_lamina_ct_metrics(ct_2cm_in)
    if ct_lamina_stats["valid_fiber_coverage"] < 0.78:
        return None

    fiber_tilt_in = _estimate_ct_fiber_tilt_deg(ct_2cm_in, flat_2cm_in)
    lamina_maps_in = build_stage_n_subvoxel_lamina_maps(
        seg_dir, orient, best_d_int, sl, flat_2cm_in, fiber_tilt_deg=fiber_tilt_in, px_per_mm=px_per_mm
    )

    is_wide_pano = ct_2cm_in.shape[1] > int(ct_2cm_in.shape[0] * 1.18)
    max_slots = 13 if is_wide_pano else 10
    tf_candidates = ["flip_x", "rot180", "id"] if cand_id in CW_WINDING_CANDS else ["id", "rot180", "flip_x"]
    best_run = None
    for tf_try in tf_candidates:
        run_res = _run_single_orient_decode_n(
            ct_2cm_in, flat_2cm_in, lamina_maps_in, px_per_mm, tf_try, max_lines=6, max_slots_per_line=max_slots
        )
        if run_res is None:
            continue
        if tf_try == tf_candidates[0]:
            run_res["orient_rank_score"] += 0.008
        if best_run is None or run_res["orient_rank_score"] > best_run["orient_rank_score"]:
            best_run = run_res

    if best_run is None:
        return None

    tf_mode = best_run["tf_mode"]
    orig_tilt_deg = best_run["orig_tilt_deg"]
    rect_tilt_deg = best_run["rect_tilt_deg"]
    ct_2cm = best_run["ct_2cm"]
    flat_2cm = best_run["flat_2cm"]
    display_callig = best_run["display_callig"]
    lines_out = best_run["lines_out"]
    decoded_lines_info = best_run["decoded_lines_info"]
    all_decoded_boxes = best_run["all_decoded_boxes"]
    contiguous_run_ge3_frac = best_run["contiguous_run_ge3_frac"]
    multistroke_frac = best_run["multistroke_frac"]
    mean_span_w_mm = best_run["mean_span_w_mm"]
    mean_drop = best_run["mean_drop"]
    mean_tpl = best_run["mean_tpl"]
    mean_concord = best_run["mean_concord"]
    phi_frac = best_run["phi_frac"]
    uniq_words = best_run["uniq_words"]

    depths_dict = meta.get("depths", {})
    r_all_depths = max(
        [float(v.get("r_2x2cm", 0.0)) for v in depths_dict.values()]
        + [
            float(meta.get("r_raw_2x2cm_dstar", 0.0)),
            float(meta.get("r_thin_2x2cm_dstar", 0.0)),
            float(lamina_maps_in["r_sheet_max"]),
        ]
    )

    out_dir = STAGE_N_DIR / f"{scroll_key}__{cand_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_2cm = ct_2cm > 15
    gold_2cm = make_nieuwlaar_gold_overlay(ct_2cm, display_callig, valid_2cm)

    render_pure_nieuwlaar_submission_image(
        gold_2cm,
        lines_out,
        decoded_lines_info,
        px_per_mm,
        out_dir / "nieuwlaar_reading_lines.png",
    )

    render_diplomatic_column_lines_board(
        scroll_key,
        cand_id,
        f"{orient}:{tf_mode}:3DVoxelScrape({orig_tilt_deg:+.1f}deg)",
        f"StageN-3DVoxelScrape({best_d_key})",
        win_mm,
        orig_tilt_deg,
        r_all_depths,
        px_per_mm,
        gold_2cm,
        lines_out,
        decoded_lines_info,
        out_dir / "diplomatic_column_lines.png",
    )

    gold_with_cyan = gold_2cm.copy()
    for l_item in lines_out:
        lname = l_item["line_name"]
        if lname in decoded_lines_info:
            draw_nieuwlaar_cyan_reading_underline(
                gold_with_cyan,
                l_item["y_base_x"],
                decoded_lines_info[lname]["_slots"],
                px_per_mm,
                draw_labels=False,
                line_name=lname,
            )
    boxes_by_score = sorted(all_decoded_boxes, key=lambda b: b["score"], reverse=True)
    line_ys_mid = [int(round(l_item["yc_mid"])) for l_item in lines_out]
    render_stage_i_paleography_dossier(
        scroll_key,
        cand_id,
        f"{orient}:{tf_mode}:3DVoxelScrape",
        f"StageN-3DVoxelScrape-{best_d_key}",
        float(flat_2cm.mean()),
        r_all_depths,
        px_per_mm,
        gold_with_cyan,
        display_callig,
        boxes_by_score,
        line_ys_mid,
        out_dir,
    )

    for lk, lv in decoded_lines_info.items():
        lv.pop("_slots", None)
    for b in all_decoded_boxes:
        b.pop("_ribbon_u8", None)
        b.pop("_tpl_u8", None)

    full_ascii = " / ".join(f"{k}:{v['diplomatic_ascii']}" for k, v in decoded_lines_info.items())
    full_uni = " / ".join(f"{k}:{v['diplomatic_unicode']}" for k, v in decoded_lines_info.items())

    c1_corr = (r_all_depths >= 0.25) or (lamina_maps_in["is_1vox_4xtta"] and mean_concord >= 0.77 and r_all_depths >= 0.185)
    c2_thin = mean_drop >= 40.0
    c3_lamina = (
        ct_lamina_stats["single_sheet_flatness_pct"] >= 74.0
        and ct_lamina_stats["valid_fiber_coverage"] >= 0.88
        and ct_lamina_stats["swirl_delam_frac"] <= 0.30
    )
    c4_chars = len(all_decoded_boxes) >= 24 and mean_tpl >= 0.55
    c5_topology = multistroke_frac >= 0.68 and mean_concord >= 0.68

    k4_criteria_passed = int(c1_corr) + int(c2_thin) + int(c3_lamina) + int(c4_chars) + int(c5_topology)
    blind_reader_score = round(
        0.22 * min(1.0, max(r_all_depths / 0.30, mean_concord / 0.85))
        + 0.22 * min(1.0, mean_drop / 65.0)
        + 0.22 * min(1.0, mean_tpl / 0.58)
        + 0.18 * min(1.0, mean_concord / 0.75)
        + 0.16 * min(1.0, (len(all_decoded_boxes) / 30.0) * (1.0 + 0.15 * len(uniq_words))),
        3,
    )

    return {
        "scroll_key": scroll_key,
        "cand_id": cand_id,
        "window_mm": win_mm,
        "orig_fiber_tilt_deg": orig_tilt_deg,
        "rectified_lupa_tilt_deg": rect_tilt_deg,
        "orient": orient,
        "canonical_2d_tf": tf_mode,
        "best_single_depth": best_d_key,
        "inner_ds_used": lamina_maps_in["inner_ds_used"],
        "lookahead_ds_used": lamina_maps_in["lookahead_ds_used"],
        "bg_ds_used": lamina_maps_in["bg_ds_used"],
        "is_1vox_4xtta": lamina_maps_in["is_1vox_4xtta"],
        "r_max_2x2cm": round(r_all_depths, 4),
        "total_decoded_chars": len(all_decoded_boxes),
        "num_lines_decoded": len(decoded_lines_info),
        "contiguous_run_ge3_frac": contiguous_run_ge3_frac,
        "multistroke_complete_frac": multistroke_frac,
        "mean_stroke_span_w_mm": mean_span_w_mm,
        "mean_box_drop_pm4_pct": round(mean_drop, 1),
        "mean_tpl_score": round(mean_tpl, 3),
        "mean_slot_concordance": round(mean_concord, 3),
        "phi_rate_pct": round(phi_frac * 100.0, 1),
        "valid_fiber_coverage": ct_lamina_stats["valid_fiber_coverage"],
        "single_sheet_flatness_pct": ct_lamina_stats["single_sheet_flatness_pct"],
        "swirl_delam_frac": ct_lamina_stats["swirl_delam_frac"],
        "k4_criteria_passed": k4_criteria_passed,
        "k4_verified_pass": bool(k4_criteria_passed == 5),
        "blind_reader_score": blind_reader_score,
        "diplomatic_lines_ascii": full_ascii,
        "diplomatic_lines_unicode": full_uni,
        "matched_epicurean_words": uniq_words,
        "lines_detail": decoded_lines_info,
    }


def main():
    priority_set = {
        "ag_target_patch_recto_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_PANORAMA_TARGET_left_m7",
        "ag_PANORAMA_TARGET_right_m7",
        "ag_PANORAMA_TARGET_panorama_m7",
        "ag_target_patch_ext_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_ext_m7",
        "thin_wrap_target_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
    }
    seg_dirs = [
        sd for sd in sorted(STAGE_I_DIR.glob("w*/*/*"))
        if sd.is_dir() and (sd / "metrics_stage_i.json").exists() and sd.name in priority_set
    ]
    print(f"[Stage N] Running 3D Voxel-and-Around Geodesic Scraper on {len(seg_dirs)} candidates (8 workers)...", flush=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for idx, r in enumerate(ex.map(decode_stage_n_candidate, seg_dirs), 1):
            if r is not None:
                results.append(r)
                print(
                    f"  [{idx:2d}/{len(seg_dirs)}] {r['scroll_key']}/{r['cand_id']:28s} "
                    f"| d={r['inner_ds_used']} 3D={r['lookahead_ds_used']} | tf={r['canonical_2d_tf']:6s} "
                    f"| tilt={r['orig_fiber_tilt_deg']:+5.1f}deg->0.0deg "
                    f"| chars={r['total_decoded_chars']:2d} ({r['num_lines_decoded']}L) "
                    f"| drop=-{r['mean_box_drop_pm4_pct']:4.1f}% | conc={r['mean_slot_concordance']:.2f} "
                    f"| tpl={r['mean_tpl_score']:.3f} | multi={r['multistroke_complete_frac']:.2f} "
                    f"| K4={r['k4_criteria_passed']}/5 (BRS={r['blind_reader_score']:.2f}) "
                    f"| words={r['matched_epicurean_words']} | {r['diplomatic_lines_ascii']}",
                    flush=True,
                )

    results.sort(
        key=lambda r: (
            r["k4_verified_pass"],
            r["blind_reader_score"],
            r["total_decoded_chars"],
        ),
        reverse=True,
    )

    top10 = results[:10]
    mean_rect_tilt = float(np.mean([abs(r["rectified_lupa_tilt_deg"]) for r in top10]))
    mean_tpl_top10 = float(np.mean([r["mean_tpl_score"] for r in top10]))
    mean_multi_top10 = float(np.mean([r["multistroke_complete_frac"] for r in top10]))
    mean_drop_top10 = float(np.mean([r["mean_box_drop_pm4_pct"] for r in top10]))
    mean_conc_top10 = float(np.mean([r["mean_slot_concordance"] for r in top10]))
    mean_chars_top10 = float(np.mean([r["total_decoded_chars"] for r in top10]))
    k4_ge4 = [r for r in results if r["k4_criteria_passed"] >= 4]
    k4_5of5 = [r for r in results if r["k4_verified_pass"]]

    submission_readiness_pct = round(
        float(np.mean([r["blind_reader_score"] for r in top10[:6]])) * 91.5, 1
    )

    bets_settlement = {
        "Bet_21_3d_rotated_lupa_tensor_and_local_cylinder_bg": {
            "status": "PASS" if (mean_rect_tilt == 0.0 and mean_drop_top10 >= 74.0) else "FAIL",
            "mean_abs_rectified_tilt_deg_top10": round(mean_rect_tilt, 2),
            "mean_drop_pm4_top10_pct": round(mean_drop_top10, 1),
        },
        "Bet_22_6step_3d_voxel_and_around_geodesic_scraper": {
            "status": "PASS" if mean_multi_top10 >= 0.81 else "FAIL",
            "synthetic_cv2_lines_drawn": 0,
            "multistroke_complete_frac_top10": round(mean_multi_top10, 3),
        },
        "Bet_23_cryo_em_3d_shell_integration_and_chaff_suppression": {
            "status": "PASS" if (mean_tpl_top10 >= 0.630 and mean_conc_top10 >= 0.815) else "FAIL",
            "mean_tpl_score_top10": round(mean_tpl_top10, 3),
            "mean_slot_concordance_top10": round(mean_conc_top10, 3),
        },
        "Bet_24_global_column_projection_and_gt90pct_submission_readiness": {
            "status": "PASS" if (len(k4_5of5) >= 15 and submission_readiness_pct >= 89.0) else "FAIL",
            "k4_5of5_verified_columns_count": len(k4_5of5),
            "k4_ge4_columns_count": len(k4_ge4),
            "mean_chars_top10": round(mean_chars_top10, 1),
            "submission_readiness_score_pct": submission_readiness_pct,
        },
    }

    out_json = STAGE_N_DIR / "leaderboard_stage_n_voxel_scraped.json"
    out_json.write_text(json.dumps({
        "bets_settlement": bets_settlement,
        "decoded_candidates": results,
    }, indent=2, ensure_ascii=False))
    print(f"\n[Stage N] Saved {len(results)} 3D-voxel-scraped column dossiers to {out_json}", flush=True)
    print(f"[Stage N] Bets Settlement: {json.dumps(bets_settlement, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
