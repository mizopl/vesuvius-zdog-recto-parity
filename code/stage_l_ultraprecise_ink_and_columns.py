#!/usr/bin/env python3
"""
Stage L — CLSS Ultra-Precision Sub-Voxel Ink Detector & Corridor-Anchored Greek Column Decoder
Implements the hybrid architecture `L_1 = L_2 + L_3 + L_4 + L_5 + L_6 + L_8`:

  1. L_2 (`d*`-Centered 1-Voxel 4x-TTA Triplet Tracker):
     Fixes the Stage K seam where `inner_ds = (-2, 0, +2)` and `bg = (-4, +4)` were hardcoded
     regardless of `best_dstar`. Uses the true `4x TTA + seed42/43` 1-voxel triplet
     `{d* - 1, d*, d* + 1}` (e.g. `{+1, +2, +3}` for `d* = +2` and `{-3, -2, -1}` for `d* = -2`)
     referenced against the true symmetric `+-4` voxel off-sheet background `{d* - 4, d* + 4}`
     plus outer `{-6, +6}` bleed-through reference.

  2. L_3 (Tri-Model Soft-Geometric Concordance & Z-Bell Bleed-Through Veto):
     Requires simultaneous thin-layer ink agreement across `dnative` (Villa 3-model ensemble),
     `hecate` (48-slice 3D UNet/Transformer), and `ens_z` (3D Z-normalized consensus), and
     vetoes pixels where off-sheet intensity at `|d - d*| >= 4` exceeds 68% of the sheet peak.

  3. L_4 (Strict Zero-Hallucination Ink Fidelity + Long Vascular Fiber-Groove Top-Hat Removal):
     Eliminates all synthetic `cv2.line` and artificial intensity floor painting (`max(150, ...)`
     and `148` loop fills) from Stage K (`0` synthetic pixels!), and subtracts continuous `> 3.2 mm`
     vascular fiber grooves along the local CT fiber angle `theta_fiber` before crossbar detection.

  4. L_5 (Corridor-Anchored Topological & Structural Greek Uncial Classifier):
     Preserves true vertical position inside the reading line corridor `[y_base - nom_h, y_base]`
     and true aspect ratio `w/h`, combining 7 structural/topological invariants with NCC template
     matching and empirical Herculaneum Greek character priors to eliminate the ~20% `PH` (Phi)
     over-prediction seam.

  5. L_6 (Expanded 6-Line x 10-Slot Full-Canvas Decoder + Wide Horizontal Panorama Stitching):
     Decodes up to 6 reading lines (`L1..L6`) along CT papyrus fiber-thread streamlines across all
     promoted/high-SNR candidates plus the stitched horizontal panorama `ag_PANORAMA_TARGET_panorama_m7`.
"""
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
    find_largest_valid_interior_crop,
    build_chamfered_45deg_polyline,
    viterbi_decode_line_slots,
    make_nieuwlaar_gold_overlay,
    draw_nieuwlaar_cyan_reading_underline,
    render_pure_nieuwlaar_submission_image,
    render_diplomatic_column_lines_board,
)

STAGE_I_DIR = BASE / "results_stage_i"
STAGE_L_DIR = BASE / "results_stage_l_ultraprecise"
STAGE_L_DIR.mkdir(parents=True, exist_ok=True)

# Empirical Herculaneum Greek uncial log-prior adjustments (preventing rare Phi/Chi/Omega over-calling)
UNCIAL_PRIOR_BIAS = {
    "A": +0.018,
    "E": +0.022,
    "I": +0.015,
    "O": +0.018,
    "Y": +0.006,
    "H": +0.012,
    "W": -0.022,
    "T": +0.016,
    "N": +0.016,
    "C": +0.016,
    "P": +0.012,
    "K": +0.010,
    "L": +0.010,
    "M": +0.008,
    "PI": +0.008,
    "G": +0.006,
    "D": +0.006,
    "TH": -0.004,
    "X": -0.016,
    "PH": -0.012,  # Mild prior penalty; structural tip+loop gate suppresses false PH while keeping real Phi (e.g. PHYCIN)
}

CW_WINDING_CANDS = {
    "ag_target_patch_m7",
    "ag_target_patch_m7",
    "ag_target_patch_m7",
    "thin_wrap_target_77keV_m7",
    "thin_wrap_target_77keV_m7",
    "thin_wrap_target_77keV_m7",
    "thin_wrap_target_77keV_m7",
    "ag_target_patch_ext_m7",
    "ag_target_patch_recto_m7",
    "ag_target_patch_recto_m7",
    "ag_target_patch_ext_m7",
}


def subtract_long_papyrus_fiber_grooves(
    img_u8: np.ndarray,
    fiber_tilt_deg: float,
    px_per_mm: float,
    min_fiber_len_mm: float = 3.2,
    sub_weight: float = 0.52,
) -> np.ndarray:
    """
    L_4: Subtracts continuous straight papyrus vascular fiber-groove striations (`length >= 3.2 mm`)
    oriented along `fiber_tilt_deg` using a directional morphological opening top-hat filter.
    Since Greek uncial horizontal crossbars are `< 2.3 mm` wide, a `3.2 mm` linear opening kernel
    extracts ONLY long papyrus fiber streaks and leaves letter crossbars intact!
    """
    k_len = max(31, int(round(min_fiber_len_mm * px_per_mm)) | 1)
    k_line = np.zeros((k_len, k_len), dtype=np.uint8)
    c = k_len // 2
    rad = np.deg2rad(fiber_tilt_deg)
    dx = int(round(c * np.cos(rad)))
    dy = int(round(c * np.sin(rad)))
    cv2.line(k_line, (c - dx, c - dy), (c + dx, c + dy), 1, 3)

    long_fiber_streak = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, k_line)
    long_fiber_sm = cv2.GaussianBlur(long_fiber_streak.astype(np.float32), (0, 0), 3.0)
    cleaned = np.clip(img_u8.astype(np.float32) - sub_weight * long_fiber_sm, 0.0, 255.0)
    return cleaned.astype(np.uint8)


def _build_junction_preserving_directional_kernels(lengths: tuple[int, ...] = (15, 23), thickness: int = 4) -> list[tuple[int, np.ndarray]]:
    out = []
    for length in lengths:
        c = length // 2
        for angle_deg in (0, 30, 60, 90, 120, 150):
            k = np.zeros((length, length), dtype=np.uint8)
            rad = np.deg2rad(angle_deg)
            dx = int(round(c * np.cos(rad)))
            dy = int(round(c * np.sin(rad)))
            cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness)
            out.append((angle_deg, k))
    return out


DIR_BANK_L = _build_junction_preserving_directional_kernels((15, 23), 4)


def junction_preserving_stroke_bank_l(
    img_u8: np.ndarray,
    fiber_tilt_deg: float,
    px_per_mm: float,
    crossbar_boost: float = 1.24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 6-angle junction-preserving directional second-derivative ridge bank,
    and subtracts long (`>= 3.8 mm`) continuous vascular fiber streaks ONLY outside letter stems.
    """
    img_f = img_u8.astype(np.float32)
    horiz_max = np.zeros_like(img_f)
    vert_max = np.zeros_like(img_f)

    for sig in (4.4, 7.8):
        sm = gaussian_filter(img_f, sigma=sig)
        dxx = gaussian_filter(sm, sigma=1.0, order=(0, 2)) * (sig ** 2)
        dyy = gaussian_filter(sm, sigma=1.0, order=(2, 0)) * (sig ** 2)
        dxy = gaussian_filter(sm, sigma=1.0, order=(1, 1)) * (sig ** 2)
        for angle_deg in (0, 30, 60, 90, 120, 150):
            rad = np.deg2rad(angle_deg)
            nx = -np.sin(rad)
            ny = np.cos(rad)
            d2_perp = (nx * nx) * dxx + 2.0 * (nx * ny) * dxy + (ny * ny) * dyy
            ridge_dir = np.maximum(0.0, -d2_perp)
            if angle_deg in (0, 30, 150):
                horiz_max = np.maximum(horiz_max, ridge_dir)
            else:
                vert_max = np.maximum(vert_max, ridge_dir)

    # Boost short horizontal crossbars ONLY where vertical stem energy exists within +-0.55 mm
    vert_vicinity = gaussian_filter(vert_max, sigma=36.0)
    v_p85 = float(np.percentile(vert_vicinity, 85.0)) if vert_vicinity.size > 0 else 1.0
    stem_gate = np.clip(vert_vicinity / max(1e-4, v_p85), 0.20, 1.0)

    # Subtract >3.8 mm continuous fiber streaks ONLY where vertical stem energy is absent
    de_fibered_u8 = subtract_long_papyrus_fiber_grooves(
        img_u8, fiber_tilt_deg, px_per_mm, min_fiber_len_mm=3.8, sub_weight=0.32
    )
    stem_protect = np.clip((stem_gate - 0.20) / 0.80, 0.0, 1.0)
    base_u8 = (
        stem_protect * img_u8.astype(np.float32) + (1.0 - stem_protect) * de_fibered_u8.astype(np.float32)
    ).astype(np.uint8)

    horiz_boosted = horiz_max * (1.0 + (crossbar_boost - 1.0) * stem_gate)

    combined = np.maximum(vert_max, horiz_boosted)
    morph_max = base_u8.copy()
    for _, k_el in DIR_BANK_L:
        closed = cv2.morphologyEx(base_u8, cv2.MORPH_CLOSE, k_el)
        morph_max = np.maximum(morph_max, closed)

    p994 = float(np.percentile(combined, 99.4)) if combined.size > 0 else 1.0
    comb_u8 = np.clip(combined / max(1e-4, p994) * 255.0, 0.0, 255.0).astype(np.uint8)
    fused_u8 = (0.62 * comb_u8.astype(np.float32) + 0.38 * morph_max.astype(np.float32)).astype(np.uint8)

    hp99 = max(1e-4, float(np.percentile(horiz_boosted, 99.2)))
    vp99 = max(1e-4, float(np.percentile(vert_max, 99.2)))
    horiz_u8 = np.clip(horiz_boosted / hp99 * 255.0, 0.0, 255.0).astype(np.uint8)
    vert_u8 = np.clip(vert_max / vp99 * 255.0, 0.0, 255.0).astype(np.uint8)
    return fused_u8, horiz_u8, vert_u8


def build_stage_l_ultraprecise_lamina_maps(
    seg_dir: Path,
    orient: str,
    best_d_int: int,
    sl: tuple[slice, slice],
    flat_2cm: np.ndarray,
    fiber_tilt_deg: float,
    px_per_mm: float,
) -> dict:
    """
    L_2 + L_3 + L_4:
      1. Selects the true `d*`-centered inner sheet triplet (`{d*-1, d*, d*+1}` when 1-voxel
         4x-TTA slices exist on disk, otherwise `{d*-2, d*}` / `{-2, 0, +2}`) and true
         symmetric off-lamina background `{d*-4, d*+4}` + outer `{-6, +6}`.
      2. Computes Tri-Model Soft-Geometric Concordance across `dnative`, `hecate`, and `ens_z`.
      3. Applies a Z-Bell Curvature & Neighboring-Wrap Bleed-Through Veto where off-sheet
         intensity exceeds 68% of the on-sheet intensity.
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

    # L_2: Choose the true d*-centered inner triplet!
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

    uniq_inner = sorted(set(inner_ds))
    # True symmetric +-4 voxel off-sheet reference around `best_d_int`
    d_bg_lo = max(-6, best_d_int - 4)
    d_bg_hi = min(+6, best_d_int + 4)

    dn_in_raw = {d: load_d("dnative", d) for d in uniq_inner}
    hc_in_raw = {d: load_d("hecate", d) for d in uniq_inner}
    ez_in_raw = {d: load_d("ens_z", d) for d in uniq_inner}

    dn_lo_bg_raw, dn_hi_bg_raw = load_d("dnative", d_bg_lo), load_d("dnative", d_bg_hi)
    hc_lo_bg_raw, hc_hi_bg_raw = load_d("hecate", d_bg_lo), load_d("hecate", d_bg_hi)
    ez_lo_bg_raw, ez_hi_bg_raw = load_d("ens_z", d_bg_lo), load_d("ens_z", d_bg_hi)

    dn_m6_raw, dn_p6_raw = load_d("dnative", -6), load_d("dnative", +6)
    hc_m6_raw, hc_p6_raw = load_d("hecate", -6), load_d("hecate", +6)

    def hp_sub(img: np.ndarray) -> np.ndarray:
        return img - gaussian_filter(img, sigma=90.0)

    dn_in_hp = {d: hp_sub(dn_in_raw[d]) for d in uniq_inner}
    hc_in_hp = {d: hp_sub(hc_in_raw[d]) for d in uniq_inner}
    ez_in_hp = {d: hp_sub(ez_in_raw[d]) for d in uniq_inner}

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

    dn_in_c = {d: scale_locked(dn_in_hp[d], dn_b_lo, dn_b_hi) for d in uniq_inner}
    hc_in_c = {d: scale_locked(hc_in_hp[d], hc_b_lo, hc_b_hi) for d in uniq_inner}
    ez_in_c = {d: scale_locked(ez_in_hp[d], ez_b_lo, ez_b_hi) for d in uniq_inner}

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

    # Outer background combines d*+-4 (75%) and extreme +-6 (25%)
    dn_bg = 0.75 * (0.5 * (dn_cm4 + dn_cp4)) + 0.25 * (0.5 * (dn_cm6 + dn_cp6))
    hc_bg = 0.75 * (0.5 * (hc_cm4 + hc_cp4)) + 0.25 * (0.5 * (hc_cm6 + hc_cp6))
    ez_bg = 0.5 * (ez_cm4 + ez_cp4)

    dn_thin_d = {d: np.maximum(0.0, dn_in_c[d] - 0.80 * dn_bg) for d in uniq_inner}
    hc_thin_d = {d: np.maximum(0.0, hc_in_c[d] - 0.86 * hc_bg) for d in uniq_inner}
    ez_thin_d = {d: np.maximum(0.0, ez_in_c[d] - 0.82 * ez_bg) for d in uniq_inner}
    joint_thin_d = {
        d: 0.48 * dn_thin_d[d] + 0.32 * hc_thin_d[d] + 0.20 * ez_thin_d[d]
        for d in uniq_inner
    }

    sm_stack = np.stack([gaussian_filter(joint_thin_d[d], sigma=48.0) for d in inner_ds], axis=0)
    logits = sm_stack / 9.5
    logits -= np.max(logits, axis=0, keepdims=True)
    w_stack = np.exp(logits)
    w_stack /= np.sum(w_stack, axis=0, keepdims=True) + 1e-6

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

    # L_3: Neighboring-Wrap Bleed-Through & Z-Bell Curvature Veto
    # Attenuate ONLY where off-sheet background exceeds 62% of on-sheet intensity (avoiding double-subtraction)
    ens_sheet = 0.55 * dn_c0_sheet + 0.45 * hc_c0_sheet
    ens_outer_lo = 0.65 * (0.55 * dn_cm4 + 0.45 * hc_cm4) + 0.35 * (0.55 * dn_cm6 + 0.45 * hc_cm6)
    ens_outer_hi = 0.65 * (0.55 * dn_cp4 + 0.45 * hc_cp4) + 0.35 * (0.55 * dn_cp6 + 0.45 * hc_cp6)
    worst_outer_ratio = np.maximum(ens_outer_lo, ens_outer_hi) / np.maximum(12.0, ens_sheet)
    z_bell_gate = np.clip((0.96 - worst_outer_ratio) / 0.34, 0.35, 1.0)
    z_bell_gate = gaussian_filter(z_bell_gate, sigma=6.0)

    dn_thin = dn_thin * z_bell_gate
    hc_thin = hc_thin * z_bell_gate
    ez_thin = ez_thin * z_bell_gate

    # L_3: Tri-Model Soft-Geometric Concordance Field
    concord_geom = (
        (np.maximum(0.0, dn_thin) ** 0.45)
        * (np.maximum(8.0, hc_thin) ** 0.35)
        * (np.maximum(8.0, ez_thin) ** 0.20)
    )
    concord_ratio = (2.0 * np.sqrt(np.maximum(0.0, dn_c0_sheet * hc_c0_sheet))) / (dn_c0_sheet + hc_c0_sheet + 6.0)

    dn_thin_u8 = np.clip(dn_thin, 0.0, 255.0).astype(np.uint8)
    hc_thin_u8 = np.clip(hc_thin, 0.0, 255.0).astype(np.uint8)
    concord_u8 = np.clip(concord_geom, 0.0, 255.0).astype(np.uint8)

    dn_diffused = coherence_enhancing_stroke_diffusion(dn_thin_u8, n_iter=3)
    dn_ridge, horiz_u8, vert_u8 = junction_preserving_stroke_bank_l(
        dn_diffused, fiber_tilt_deg=fiber_tilt_deg, px_per_mm=px_per_mm, crossbar_boost=1.18
    )
    hc_ridge, _, _ = junction_preserving_stroke_bank_l(
        hc_thin_u8, fiber_tilt_deg=fiber_tilt_deg, px_per_mm=px_per_mm, crossbar_boost=1.16
    )

    hc_anchor = gaussian_filter(np.maximum(hc_thin, hc_ridge.astype(np.float32)), sigma=38.0)
    hc_anchor_norm = np.clip(hc_anchor / max(7.0, float(np.percentile(hc_anchor[valid], 90.0))), 0.28, 1.0)

    # Fuse linear ridge response with Tri-Model Geometric Concordance
    stroke_callig = (
        0.32 * dn_diffused.astype(np.float32)
        + 0.26 * dn_ridge.astype(np.float32)
        + 0.22 * concord_geom
        + 0.12 * hc_thin
        + 0.08 * hc_ridge.astype(np.float32)
    ) * (0.42 + 0.58 * hc_anchor_norm) * np.clip(flat_2cm, 0.0, 1.0)

    p994 = float(np.percentile(stroke_callig[valid], 99.4)) if valid.any() else 255.0
    if p994 > 1.0:
        stroke_callig = np.clip(stroke_callig / p994 * 255.0, 0.0, 255.0)
    stroke_callig_u8 = cv2.morphologyEx(
        stroke_callig.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    ens_0_locked = (0.55 * dn_c0_sheet + 0.45 * hc_c0_sheet).astype(np.uint8)
    ens_m4_locked = (0.55 * dn_cm4 + 0.45 * hc_cm4).astype(np.uint8)
    ens_p4_locked = (0.55 * dn_cp4 + 0.45 * hc_cp4).astype(np.uint8)

    v_dn_s = dn_c0_sheet[valid].ravel()
    v_hc_s = hc_c0_sheet[valid].ravel()
    r_sheet_max = (
        float(np.corrcoef(v_dn_s, v_hc_s)[0, 1])
        if (v_dn_s.size > 400 and v_dn_s.std() > 1e-3 and v_hc_s.std() > 1e-3)
        else 0.0
    )

    return {
        "dn_thin_u8": dn_thin_u8,
        "hc_thin_u8": hc_thin_u8,
        "concord_u8": concord_u8,
        "concord_ratio": concord_ratio,
        "dn_ridge_u8": dn_ridge,
        "horiz_u8": horiz_u8,
        "vert_u8": vert_u8,
        "callig_u8": stroke_callig_u8,
        "ens_0_locked": ens_0_locked,
        "ens_m4_locked": ens_m4_locked,
        "ens_p4_locked": ens_p4_locked,
        "hc_0_locked": hc_c0_sheet.astype(np.uint8),
        "hc_m4_locked": hc_cm4.astype(np.uint8),
        "hc_p4_locked": hc_cp4.astype(np.uint8),
        "inner_ds_used": list(inner_ds),
        "bg_ds_used": [d_bg_lo, d_bg_hi],
        "is_1vox_4xtta": is_1vox_4xtta,
        "r_sheet_max": round(r_sheet_max, 4),
    }


def calibrate_corridor_ultraprecise(
    callig_2cm: np.ndarray,
    dn_thin_2cm: np.ndarray,
    hc_thin_2cm: np.ndarray,
    concord_2cm: np.ndarray,
    flat_2cm: np.ndarray,
    lines_out: list[dict],
    px_per_mm: float,
) -> np.ndarray:
    """
    Normalizes calligraphic stroke intensity strictly inside reading corridors `L1..L6`
    using dual-threshold hysteresis on real neural ink without adding any synthetic pixels.
    """
    H, W = callig_2cm.shape
    corridor_weight = np.full((H, W), 0.14, dtype=np.float32)
    nom_h = int(round(2.48 * px_per_mm))
    xs = np.arange(W, dtype=np.int32)

    for l_item in lines_out:
        yb_x = l_item["y_base_x"]
        yc_x = yb_x - 0.50 * nom_h
        for col_x in range(W):
            yc = yc_x[col_x]
            dy = np.abs(np.arange(H, dtype=np.float32) - yc) / max(1.0, 0.56 * nom_h)
            w_col = np.exp(-0.5 * (dy ** 2.4))
            corridor_weight[:, col_x] = np.maximum(corridor_weight[:, col_x], w_col)

    raw_f = (
        0.52 * callig_2cm.astype(np.float32)
        + 0.24 * concord_2cm.astype(np.float32)
        + 0.14 * dn_thin_2cm.astype(np.float32)
        + 0.10 * hc_thin_2cm.astype(np.float32)
    ) * np.clip(flat_2cm, 0.0, 1.0)

    in_corr = (corridor_weight >= 0.48) & (flat_2cm > 0.45)
    if in_corr.sum() < 200:
        in_corr = flat_2cm > 0.35

    t_high = float(np.percentile(raw_f[in_corr], 84.0))
    t_low = float(np.percentile(raw_f[in_corr], 56.0))
    t_max = max(t_high + 5.0, float(np.percentile(raw_f[in_corr], 99.2)))

    seed_mask = (raw_f >= t_high) & (corridor_weight >= 0.42)
    low_mask = (raw_f >= t_low) & (corridor_weight >= 0.32)

    bridge_px = max(7, int(round(0.18 * px_per_mm)) | 1)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_px, bridge_px))
    grown = seed_mask.astype(np.uint8)
    for _ in range(3):
        grown = cv2.dilate(grown, k_dil) & low_mask.astype(np.uint8)

    hyst_gain = np.where(grown > 0, 1.0, 0.40).astype(np.float32)
    hyst_gain = gaussian_filter(hyst_gain, sigma=2.2)

    norm_f = np.clip((raw_f - t_low * 0.78) / (t_max - t_low * 0.78), 0.0, 1.0)
    clean_f = norm_f * corridor_weight * hyst_gain * 255.0
    return np.clip(clean_f, 0.0, 255.0).astype(np.uint8)


def isolate_ultraprecise_uncial_character(
    patch_cal: np.ndarray,
    patch_concord: np.ndarray,
    px_per_mm: float,
) -> tuple[np.ndarray, float, float, dict]:
    """
    L_4: STRICT ZERO-HALLUCINATION character isolation:
      - Keeps 1 or 2 real ink components inside the character slot (so two-leg letters `H, N, A, Π, K, M`
        keep their second leg when separated by <= 0.68 mm).
      - NEVER draws synthetic `cv2.line` bridges or artificial `148`/`150` pixel floors onto the ink image!
        Every bright pixel in `clean_u8` is 100% modulated directly from `patch_cal`.
    """
    h, w = patch_cal.shape
    p_f = 0.72 * patch_cal.astype(np.float32) + 0.28 * patch_concord.astype(np.float32)

    # Mild morphological closing (`<= 0.12 mm`) only for mask connectivity, NEVER overwriting intensity
    close_px = max(5, int(round(0.11 * px_per_mm)) | 1)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    p_closed = cv2.morphologyEx(patch_cal, cv2.MORPH_CLOSE, k_close)

    high_thr = max(46.0, float(np.percentile(p_f, 75.0)))
    low_thr = max(28.0, float(np.percentile(p_f, 52.0)))
    bin_high = (p_f >= high_thr).astype(np.uint8)
    bin_low = (p_f >= low_thr).astype(np.uint8)

    n_l, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_high, connectivity=8)
    if n_l <= 1:
        return patch_cal, float(np.mean(patch_cal >= 68)), 0.0, {"span_w_mm": 0.0, "is_multistroke": False, "n_legs": 1}

    cy0, cx0 = h / 2.0, w / 2.0
    comp_list = []
    for i in range(1, n_l):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 9:
            continue
        comp_mask = (labels == i)
        mass = float(p_f[comp_mask].sum())
        cx, cy = centroids[i]
        dist_c = np.hypot((cx - cx0) / max(1.0, w), (cy - cy0) / max(1.0, h))
        if dist_c <= 0.44:
            comp_list.append((i, mass, cx, cy, comp_mask))

    if not comp_list:
        return patch_cal, float(np.mean(patch_cal >= 68)), 0.0, {"span_w_mm": 0.0, "is_multistroke": False, "n_legs": 1}

    comp_list.sort(key=lambda t: t[1], reverse=True)
    main_idx, main_mass, main_cx, main_cy, main_mask = comp_list[0]
    keep_canvas = np.zeros_like(patch_cal, dtype=np.uint8)
    keep_canvas[main_mask] = 255
    n_kept_legs = 1

    max_leg_sep_px = max(24, int(round(0.66 * px_per_mm)))
    main_dt = cv2.distanceTransform((~main_mask).astype(np.uint8), cv2.DIST_L2, 3)

    for idx_c, mass_c, cx_c, cy_c, mask_c in comp_list[1:3]:
        if mass_c < 0.16 * main_mass:
            continue
        min_d = float(main_dt[mask_c].min())
        if min_d <= max_leg_sep_px and abs(cy_c - main_cy) <= 0.36 * h:
            keep_canvas[mask_c] = 255
            n_kept_legs += 1

    # Expand kept components slightly within `bin_low` to preserve real fainter crossbars/serifs
    k_hyst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px + 4, close_px + 4))
    keep_canvas = cv2.dilate(keep_canvas, k_hyst) & (bin_low * 255)
    keep_mask = keep_canvas > 0

    kept_mass = float(p_f[keep_mask].sum())
    total_mass = sum(t[1] for t in comp_list) + 1e-5
    dom_ratio = min(1.0, kept_mass / total_mass)

    mask_smooth = gaussian_filter(keep_mask.astype(np.float32), sigma=2.4)
    mask_smooth = np.clip(mask_smooth * 1.42, 0.0, 1.0)
    # ZERO SYNTHETIC PIXELS: multiply only the real calligraphic intensity `p_closed` by `mask_smooth`
    clean_patch = gaussian_filter(p_closed.astype(np.float32) * mask_smooth, sigma=1.1)

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
    fill_ratio = float(np.mean(clean_u8 >= 68))

    act_ys, act_xs = np.where(clean_u8 >= 75)
    if act_xs.size >= 18:
        span_w_px = float(np.percentile(act_xs, 96) - np.percentile(act_xs, 4))
        span_h_px = float(np.percentile(act_ys, 96) - np.percentile(act_ys, 4))
        span_w_mm = span_w_px / px_per_mm
        span_h_mm = span_h_px / px_per_mm
        is_multistroke = bool(span_w_mm >= 1.10 or n_kept_legs >= 2 or span_h_mm >= 1.55)
    else:
        span_w_mm = 0.0
        is_multistroke = False

    return clean_u8, fill_ratio, dom_ratio, {
        "span_w_mm": round(span_w_mm, 2),
        "is_multistroke": is_multistroke,
        "n_legs": n_kept_legs,
    }


def score_corridor_anchored_greek_templates(patch_u8: np.ndarray, top_k: int = 5) -> tuple[list[dict], np.ndarray, int, int]:
    """
    L_5: Corridor-Anchored Structural & Topological Greek Uncial Classifier:
      1. Preserves vertical anchoring relative to the line corridor (clamping vertical re-centering
         to at most +-10% of box height) so top crossbars (`T, G, PI`), middle crossbars (`H, E, TH, A`),
         and bottom bars (`D, W`) are never confused!
      2. Preserves true aspect ratio `w/h` (clamped gently to `[0.22, 0.88]`).
      3. Evaluates 7 structural/topological invariants (central loop hollow, protruding vertical tips,
         top/mid/bot horizontal bar profile, left/center/right vertical stem profile) + empirical
         Herculaneum Greek character priors (`UNCIAL_PRIOR_BIAS`) to eliminate `PH` (Phi) over-calling.
    """
    h, w = patch_u8.shape
    ys, xs = np.where(patch_u8 >= 68)
    if ys.size < 16:
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        y0 = int(np.percentile(ys, 2.0))
        y1 = int(np.percentile(ys, 98.0)) + 1
        x0 = int(np.percentile(xs, 2.0))
        x1 = int(np.percentile(xs, 98.0)) + 1

    sh = max(14, y1 - y0)
    sw = max(12, x1 - x0)
    aspect_wh = float(sw) / float(sh)

    # Preserve vertical position relative to corridor while matching TEMPLATES_64 [0.18..0.84] height (0.66 * h)
    sub = patch_u8[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    target_h = int(round(0.66 * h))
    scale_y = target_h / max(1, sub.shape[0])
    target_w = int(np.clip(round(sw * scale_y * 0.85), int(0.24 * w), int(0.62 * w)))
    sub_rsz = cv2.resize(sub, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    centered_u8 = np.zeros_like(patch_u8)
    natural_cy = 0.5 * (y0 + y1)
    anchored_cy = int(round(0.60 * (h / 2.0) + 0.40 * natural_cy))
    cy0 = int(np.clip(anchored_cy - target_h // 2, 2, max(2, h - target_h - 2)))
    cx0 = (w - target_w) // 2
    centered_u8[cy0:cy0 + target_h, cx0:cx0 + target_w] = sub_rsz

    sz = 64
    p_rsz = cv2.resize(centered_u8, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32)
    r2 = ((yy - 31.5) / 28.0) ** 2 + ((xx - 31.5) / 28.0) ** 2
    taper = np.clip(1.15 - 0.35 * r2, 0.25, 1.0)
    lo = float(np.percentile(p_rsz, 35.0))
    hi = float(np.percentile(p_rsz, 98.0))
    p_norm = np.clip((p_rsz - lo) / max(1.0, hi - lo), 0.0, 1.0) * taper

    # Compute 7 structural & topological descriptors on `p_norm` (64x64)
    top_bar = float(np.mean(p_norm[9:20, 16:48]))
    mid_bar = float(np.mean(p_norm[25:39, 18:46]))
    bot_bar = float(np.mean(p_norm[44:55, 16:48]))

    left_stem = float(np.mean(p_norm[12:52, 11:24]))
    center_stem = float(np.mean(p_norm[10:54, 26:38]))
    right_stem = float(np.mean(p_norm[12:52, 40:53]))

    # Protruding top or bottom central stem tips (defining feature of Phi `PH`)
    top_tip = float(np.mean(p_norm[4:15, 26:38]))
    bot_tip = float(np.mean(p_norm[49:60, 26:38]))
    has_protruding_tips = ((top_tip >= 0.22 and bot_tip >= 0.18) or (bot_tip >= 0.22 and top_tip >= 0.18)) and (center_stem >= 0.26)

    # Ring / loop hollowness (for O, TH, C, E, P)
    ring_outer = 0.25 * (left_stem + right_stem + top_bar + bot_bar)
    inner_core = float(np.mean(p_norm[24:40, 24:40]))
    is_hollow_loop = (ring_outer - inner_core) >= 0.08

    p_rots = [cv2.copyMakeBorder(p_norm, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0.0)]
    for ang in (-6.0, 6.0):
        M_rot = cv2.getRotationMatrix2D((sz / 2.0, sz / 2.0), ang, 1.0)
        p_r = cv2.warpAffine(p_norm, M_rot, (sz, sz), flags=cv2.INTER_LINEAR, borderValue=0.0)
        p_rots.append(cv2.copyMakeBorder(p_r, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0.0))

    candidates = []
    for code, tpl_list in TEMPLATES_64.items():
        best_c_sc = -1.0
        best_c_tpl = tpl_list[0]
        for tpl_u8 in tpl_list:
            t_f = tpl_u8.astype(np.float32) / 255.0
            for p_pad in p_rots:
                res = cv2.matchTemplate(p_pad, t_f, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_loc = cv2.minMaxLoc(res)
                ox, oy = max_loc
                p_shifted = p_pad[oy:oy + sz, ox:ox + sz]
                inter = float(np.sum(np.minimum(p_shifted, t_f)))
                union = float(np.sum(np.maximum(p_shifted, t_f))) + 1e-5
                iou = inter / union
                sc = 0.65 * max(0.0, float(max_v)) + 0.35 * iou

                # Structural & Topological Gate Adjustments (L_5)
                if code == "PH":
                    # Phi MUST have protruding vertical stem tips AND medial loop/bar width
                    if not (has_protruding_tips and aspect_wh >= 0.48 and (left_stem + right_stem) >= 0.36):
                        sc *= 0.74
                elif code == "I":
                    if aspect_wh > 0.56:
                        sc *= 0.78
                    elif aspect_wh <= 0.42 and center_stem > max(left_stem, right_stem):
                        sc *= 1.08
                elif code == "T":
                    if top_bar > bot_bar * 1.15 and center_stem > 0.25:
                        sc *= 1.05
                    elif bot_bar > top_bar * 1.25:
                        sc *= 0.84
                elif code == "G":
                    if top_bar > bot_bar and left_stem > right_stem * 1.20:
                        sc *= 1.06
                    elif right_stem > left_stem * 1.15:
                        sc *= 0.84
                elif code == "PI":
                    if top_bar >= 0.22 and left_stem >= 0.22 and right_stem >= 0.22:
                        sc *= 1.06
                    elif aspect_wh < 0.48:
                        sc *= 0.82
                elif code in ("H", "N"):
                    if left_stem >= 0.22 and right_stem >= 0.22 and mid_bar >= 0.20 and aspect_wh >= 0.48:
                        sc *= 1.06
                    elif aspect_wh < 0.44:
                        sc *= 0.80
                elif code == "M":
                    if aspect_wh < 0.58:
                        sc *= 0.78
                elif code == "E":
                    if left_stem > right_stem * 1.12 and mid_bar >= 0.22:
                        sc *= 1.06
                elif code == "C":
                    if left_stem > right_stem * 1.12 and is_hollow_loop:
                        sc *= 1.06
                elif code == "O":
                    if is_hollow_loop and abs(left_stem - right_stem) <= 0.14:
                        sc *= 1.06
                elif code == "A":
                    if mid_bar >= 0.22 and bot_bar >= top_bar * 0.90:
                        sc *= 1.03
                elif code == "L":
                    if mid_bar < 0.24 and bot_bar < 0.24:
                        sc *= 1.05
                elif code == "D":
                    if bot_bar > top_bar * 1.18:
                        sc *= 1.06
                elif code == "W":
                    if aspect_wh < 0.74 or top_bar >= 0.18 or bot_bar < top_bar * 1.18:
                        sc *= 0.72

                sc += UNCIAL_PRIOR_BIAS.get(code, 0.0)
                if sc > best_c_sc:
                    best_c_sc = sc
                    best_c_tpl = tpl_u8

        candidates.append({
            "code": code,
            "uni": GREEK_DISPLAY_UNICODE.get(code, code),
            "tpl_score": round(float(max(0.05, best_c_sc)), 4),
            "_tpl_u8": best_c_tpl,
        })

    candidates.sort(key=lambda d: d["tpl_score"], reverse=True)
    return candidates[:top_k], centered_u8, int(y0), int(y1)


def trace_fiber_streamline_baselines_6lines(
    ct_2cm: np.ndarray,
    clean_ink: np.ndarray,
    flat_2cm: np.ndarray,
    px_per_mm: float,
    max_lines: int = 6,
) -> tuple[float, list[dict]]:
    """
    L_6: Traces up to `6` reading baselines (`L1..L6`) directly along the CT papyrus vascular
    fiber-thread streamlines (`DoG 0.12-0.35 mm`, up to +-28 deg).
    """
    H, W = clean_ink.shape
    nom_h = int(round(2.48 * px_per_mm))
    half_h = nom_h // 2

    ct_f = ct_2cm.astype(np.float32)
    ct_bp = cv2.GaussianBlur(ct_f, (0, 0), 3.0) - cv2.GaussianBlur(ct_f, (0, 0), 18.0)
    gx = cv2.Sobel(ct_bp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ct_bp, cv2.CV_32F, 0, 1, ksize=3)

    col_slopes = np.zeros(4, dtype=np.float32)
    all_angles = []
    for gx_i in range(4):
        x0, x1 = gx_i * W // 4, (gx_i + 1) * W // 4
        col_angs = []
        for gy_i in range(4):
            y0, y1 = gy_i * H // 4, (gy_i + 1) * H // 4
            v = flat_2cm[y0:y1, x0:x1] > 0.45
            if v.sum() < 80:
                continue
            jxx = float(np.mean((gx[y0:y1, x0:x1] ** 2)[v]))
            jyy = float(np.mean((gy[y0:y1, x0:x1] ** 2)[v]))
            jxy = float(np.mean((gx[y0:y1, x0:x1] * gy[y0:y1, x0:x1])[v]))
            theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-6)
            theta_tan = theta_grad + np.pi / 2.0
            while theta_tan > np.pi / 2.0:
                theta_tan -= np.pi
            while theta_tan < -np.pi / 2.0:
                theta_tan += np.pi
            col_angs.append(theta_tan)
            all_angles.append(theta_tan)
        if col_angs:
            col_slopes[gx_i] = float(np.clip(np.tan(np.median(col_angs)), -0.52, 0.52))

    med_theta = float(np.median(all_angles)) if all_angles else 0.0
    fiber_slope = float(np.clip(np.tan(med_theta), -0.50, 0.50))
    if not np.any(col_slopes):
        col_slopes[:] = fiber_slope

    groove_energy = gaussian_filter(np.abs(gy), sigma=(2.5, 18.0))
    g_p95 = max(1e-4, float(np.percentile(groove_energy, 95.0)))
    groove_norm = np.clip(groove_energy / g_p95 * 45.0, 0.0, 45.0)

    ink_ds = cv2.resize(clean_ink, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    h_ds, w_ds = ink_ds.shape
    xs_ds = np.arange(w_ds, dtype=np.float32) - 0.5 * w_ds

    best_slope = fiber_slope
    best_radon_sc = -1.0
    best_prof_full = None

    s_min = max(-0.52, fiber_slope - 0.09)
    s_max = min(0.52, fiber_slope + 0.09)
    for slope in np.linspace(s_min, s_max, 15):
        dy_col = (slope * xs_ds).astype(np.int32)
        shifted = np.zeros_like(ink_ds)
        for c_i in range(w_ds):
            s_y = int(dy_col[c_i])
            if s_y >= 0:
                if s_y < h_ds:
                    shifted[:h_ds - s_y, c_i] = ink_ds[s_y:, c_i]
            else:
                if -s_y < h_ds:
                    shifted[-s_y:, c_i] = ink_ds[:h_ds + s_y, c_i]
        row_p86 = np.percentile(shifted, 86.0, axis=1)
        k_s = max(5, int(round(1.10 * px_per_mm / 4.0)) | 1)
        row_sm = cv2.GaussianBlur(row_p86.reshape(-1, 1), (1, k_s), 0).ravel()
        mod_sc = float(np.percentile(row_sm, 92.0) - np.percentile(row_sm, 35.0))
        prior_w = 1.0 - 1.80 * abs(slope - fiber_slope)
        sc = mod_sc * max(0.35, prior_w)
        if sc > best_radon_sc:
            best_radon_sc = sc
            best_slope = float(slope)
            best_prof_full = cv2.resize(row_sm.reshape(-1, 1), (1, H), interpolation=cv2.INTER_LINEAR).ravel()

    min_line_dist = int(round(2.85 * px_per_mm))
    margin_y = int(round(1.15 * px_per_mm))
    center_ys = []
    for y_idx in np.argsort(best_prof_full)[::-1]:
        yi = int(y_idx)
        if yi < margin_y or yi > H - margin_y:
            continue
        if all(abs(yi - ly) >= min_line_dist for ly in center_ys):
            center_ys.append(yi)
            if len(center_ys) >= max_lines:
                break
    center_ys.sort()

    col_x_knots = np.array([0.125 * W, 0.375 * W, 0.625 * W, 0.875 * W], dtype=np.float32)
    blended_col_slopes = 0.65 * best_slope + 0.35 * col_slopes
    slope_x = np.interp(np.arange(W, dtype=np.float32), col_x_knots, blended_col_slopes)
    dy_streamline = np.cumsum(slope_x)
    dy_streamline -= dy_streamline[W // 2]

    n_bins = 18
    bin_w = W / n_bins
    lines_out = []
    for idx_l, yc_mid in enumerate(center_ys):
        bin_y_centers = np.zeros(n_bins, dtype=np.float32)
        for b_i in range(n_bins):
            x0_b = int(round(b_i * bin_w))
            x1_b = min(W, int(round((b_i + 1) * bin_w)))
            xm = int(round(0.5 * (x0_b + x1_b)))
            y_pred = yc_mid + float(dy_streamline[min(W - 1, max(0, xm))])
            search_r = int(round(0.40 * px_per_mm))
            ys0 = max(half_h, int(round(y_pred)) - search_r)
            ys1 = min(H - half_h, int(round(y_pred)) + search_r)
            best_yb = float(np.clip(y_pred, half_h, H - half_h))
            best_eb = -1.0
            for y_try in range(ys0, ys1 + 1, 5):
                e_ink = float(np.percentile(clean_ink[y_try - half_h:y_try + half_h, x0_b:x1_b], 88.0))
                y_groov = min(H - 1, int(round(y_try + 0.44 * nom_h)))
                e_groov = float(np.mean(groove_norm[max(0, y_groov - 4):min(H, y_groov + 5), x0_b:x1_b]))
                e_reg = e_ink + 0.45 * e_groov - 0.32 * abs(y_try - y_pred)
                if e_reg > best_eb:
                    best_eb = e_reg
                    best_yb = float(y_try)
            bin_y_centers[b_i] = best_yb

        bin_y_smooth = gaussian_filter(bin_y_centers, sigma=1.15)
        step_quant = max(10.0, 0.14 * px_per_mm)
        bin_y_stair = np.round((bin_y_smooth + 0.44 * nom_h) / step_quant) * step_quant
        bin_y_stair = np.clip(bin_y_stair, nom_h + 6, H - 6)

        knot_xs = [int(round((b_i + 0.5) * bin_w)) for b_i in range(n_bins)]
        knot_ys = [int(round(bin_y_stair[b_i])) for b_i in range(n_bins)]
        poly = build_chamfered_45deg_polyline(0, W - 1, knot_xs, knot_ys, H)
        y_base_x = poly[:, 0, 1].astype(np.float32)

        lines_out.append({
            "line_id": idx_l + 1,
            "line_name": f"L{idx_l + 1}",
            "yc_mid": float(yc_mid),
            "y_base_x": y_base_x,
        })

    tilt_deg = round(float(np.rad2deg(np.arctan(best_slope))), 2)
    return tilt_deg, lines_out


def _apply_2d_tf(arr: np.ndarray, tf_mode: str) -> np.ndarray:
    if tf_mode == "flip_x":
        return np.ascontiguousarray(arr[:, ::-1])
    if tf_mode == "rot180":
        return np.ascontiguousarray(arr[::-1, ::-1])
    return arr


def _estimate_ct_fiber_tilt_deg(ct_2cm: np.ndarray, flat_2cm: np.ndarray) -> float:
    ct_f = ct_2cm.astype(np.float32)
    ct_bp = cv2.GaussianBlur(ct_f, (0, 0), 3.0) - cv2.GaussianBlur(ct_f, (0, 0), 18.0)
    gx = cv2.Sobel(ct_bp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ct_bp, cv2.CV_32F, 0, 1, ksize=3)
    H, W = ct_2cm.shape
    angs = []
    for gx_i in range(4):
        x0, x1 = gx_i * W // 4, (gx_i + 1) * W // 4
        for gy_i in range(4):
            y0, y1 = gy_i * H // 4, (gy_i + 1) * H // 4
            v = flat_2cm[y0:y1, x0:x1] > 0.45
            if v.sum() < 80:
                continue
            jxx = float(np.mean((gx[y0:y1, x0:x1] ** 2)[v]))
            jyy = float(np.mean((gy[y0:y1, x0:x1] ** 2)[v]))
            jxy = float(np.mean((gx[y0:y1, x0:x1] * gy[y0:y1, x0:x1])[v]))
            t_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-6)
            t_tan = t_grad + np.pi / 2.0
            while t_tan > np.pi / 2.0:
                t_tan -= np.pi
            while t_tan < -np.pi / 2.0:
                t_tan += np.pi
            angs.append(t_tan)
    return float(np.rad2deg(np.median(angs))) if angs else 0.0


def _run_single_orient_decode_l(
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
        k: (_apply_2d_tf(v, tf_mode) if isinstance(v, np.ndarray) and v.ndim == 2 else v)
        for k, v in lamina_maps_in.items()
    }
    H, W = ct_2cm.shape
    callig_2cm = lamina_maps["callig_u8"]
    dn_t_2cm = lamina_maps["dn_thin_u8"]
    hc_t_2cm = lamina_maps["hc_thin_u8"]
    concord_2cm = lamina_maps["concord_u8"]
    concord_ratio_2cm = lamina_maps["concord_ratio"]
    dn_rg_2cm = lamina_maps["dn_ridge_u8"]

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
    # ZERO SYNTHETIC INK: `display_callig` starts from `clean` and only highlights real isolated character strokes
    display_callig = (clean.astype(np.float32) * 0.78).astype(np.uint8)

    for l_item in lines_out:
        idx_l = l_item["line_id"] - 1
        line_name = l_item["line_name"]
        y_base_x = l_item["y_base_x"]

        strip = np.zeros((nom_h, W), dtype=np.float32)
        for col_x in range(W):
            yb = int(np.clip(y_base_x[col_x], nom_h, H))
            strip[:, col_x] = clean[yb - nom_h:yb, col_x]

        col_prof = np.percentile(strip, 88.0, axis=0)
        k_col = max(15, int(round(0.68 * px_per_mm)) | 1)
        col_smooth = cv2.GaussianBlur(col_prof.reshape(-1, 1), (1, k_col), 0).ravel()

        picked_xs = []
        thr_col = max(18.5, float(np.percentile(col_smooth, 28.0)))
        min_slot_sep = int(round(1.68 * px_per_mm))
        for x_idx in np.argsort(col_smooth)[::-1]:
            xi = int(x_idx)
            if xi < nom_w // 2 or xi > W - nom_w // 2:
                continue
            if col_smooth[xi] < thr_col:
                continue
            if all(abs(xi - px) >= min_slot_sep for px in picked_xs):
                picked_xs.append(xi)
                if len(picked_xs) >= max_slots_per_line:
                    break
        picked_xs.sort()

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

            patch_fl = flat_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            fl_mean = float(np.mean(patch_fl))
            if fl_mean < 0.50:
                continue

            patch_dn = dn_t_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_hc = hc_t_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_rg = dn_rg_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_cc = concord_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_cr = concord_ratio_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]

            dn_p94 = float(np.percentile(patch_dn, 94.0))
            hc_p985 = float(np.percentile(patch_hc, 98.5))
            rg_p92 = float(np.percentile(patch_rg, 92.0))
            # L_8: Require dual-model thin-layer ink presence
            if dn_p94 < 20.0 or hc_p985 < 10.5 or rg_p92 < 18.5:
                continue

            patch_cal = clean[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            rib_pure, fill_r, dom_r, stroke_meta = isolate_ultraprecise_uncial_character(
                patch_cal, patch_cc, px_per_mm
            )
            rib_single, fill_s, dom_s = isolate_central_character_ribbon(patch_cal)

            cands_pure, cent_pure, y_top_p, y_bot_p = score_corridor_anchored_greek_templates(rib_pure, top_k=5)
            cands_sing, cent_sing, y_top_s, y_bot_s = score_corridor_anchored_greek_templates(rib_single, top_k=5)

            two_leg_set = {"H", "N", "A", "K", "M", "PI", "W", "O", "P", "T", "E", "D", "X", "G"}
            bonus_pure = 0.024 if (stroke_meta["is_multistroke"] and cands_pure[0]["code"] in two_leg_set) else 0.0
            if cands_pure[0]["tpl_score"] + bonus_pure >= cands_sing[0]["tpl_score"]:
                top_cands, centered_rib, y_st_top, y_st_bot = cands_pure, cent_pure, y_top_p, y_bot_p
                rib_chosen = rib_pure
            else:
                top_cands, centered_rib, y_st_top, y_st_bot = cands_sing, cent_sing, y_top_s, y_bot_s
                rib_chosen = rib_single
                fill_r, dom_r = fill_s, max(dom_r, dom_s)

            if fill_r < 0.065 or fill_r > 0.55 or dom_r < 0.58:
                continue
            if top_cands[0]["tpl_score"] < 0.41:
                continue

            active_stroke_px = rib_chosen >= 80
            px_conc = float(np.percentile(patch_cr[active_stroke_px], 75.0)) if np.any(active_stroke_px) else float(np.mean(patch_cr))
            peak_conc = float((2.0 * np.sqrt(max(0.0, dn_p94 * hc_p985))) / (dn_p94 + hc_p985 + 1.0))
            slot_concord = 0.55 * peak_conc + 0.45 * px_conc

            display_callig[best_y0:best_y0 + nom_h, x0:x0 + nom_w] = np.maximum(
                display_callig[best_y0:best_y0 + nom_h, x0:x0 + nom_w],
                rib_chosen,
            )

            slot_box = {
                "x": int(x0), "y": int(best_y0), "w": int(nom_w), "h": int(nom_h),
                "y_stroke_bot_abs": int(best_y0 + y_st_bot),
                "w_mm": round(nom_w / px_per_mm, 2),
                "h_mm": round(nom_h / px_per_mm, 2),
                "cx": round(float(x0 + 0.5 * nom_w), 1),
                "cy": round(float(best_y0 + 0.5 * nom_h), 1),
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
            clean,
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
        "ct_2cm": ct_2cm,
        "flat_2cm": flat_2cm,
        "display_callig": display_callig,
        "tilt_deg": tilt_deg,
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
        "orient_rank_score": mean_tpl + 0.015 * len(uniq_words) + 0.0012 * len(all_decoded_boxes),
    }


def find_largest_valid_interior_crop_l(ct_mid: np.ndarray, px_per_mm: float) -> tuple[slice, slice, float]:
    H, W = ct_mid.shape
    if W > int(H * 1.18):
        # Wide horizontal stitched panorama: preserve wide horizontal span!
        win_h = min(H, int(round(20.0 * px_per_mm)))
        win_w = min(W, int(round(28.0 * px_per_mm)))
        y0 = max(0, (H - win_h) // 2)
        x0 = max(0, (W - win_w) // 2)
        return slice(y0, y0 + win_h), slice(x0, x0 + win_w), round(win_w / px_per_mm, 1)
    return find_largest_valid_interior_crop(ct_mid, px_per_mm)


def decode_stage_l_candidate(seg_dir: Path) -> dict | None:
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
    lamina_maps_in = build_stage_l_ultraprecise_lamina_maps(
        seg_dir, orient, best_d_int, sl, flat_2cm_in, fiber_tilt_deg=fiber_tilt_in, px_per_mm=px_per_mm
    )

    is_wide_pano = ct_2cm_in.shape[1] > int(ct_2cm_in.shape[0] * 1.18)
    max_slots = 13 if is_wide_pano else 10
    tf_candidates = ["flip_x", "rot180", "id"] if cand_id in CW_WINDING_CANDS else ["id", "rot180", "flip_x"]
    best_run = None
    for tf_try in tf_candidates:
        run_res = _run_single_orient_decode_l(
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
    ct_2cm = best_run["ct_2cm"]
    flat_2cm = best_run["flat_2cm"]
    display_callig = best_run["display_callig"]
    tilt_deg = best_run["tilt_deg"]
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

    out_dir = STAGE_L_DIR / f"{scroll_key}__{cand_id}"
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
        f"{orient}:{tf_mode}",
        f"StageL-UltraInk({best_d_key})",
        win_mm,
        tilt_deg,
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
        f"{orient}:{tf_mode}",
        f"StageL-UltraInk-{best_d_key}",
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

    c1_corr = r_all_depths >= 0.26
    c2_thin = mean_drop >= 38.0
    c3_lamina = (
        ct_lamina_stats["single_sheet_flatness_pct"] >= 75.0
        and ct_lamina_stats["valid_fiber_coverage"] >= 0.90
        and ct_lamina_stats["swirl_delam_frac"] <= 0.30
    )
    c4_chars = len(all_decoded_boxes) >= 20 and mean_tpl >= 0.52
    c5_topology = multistroke_frac >= 0.65 and mean_concord >= 0.56

    k4_criteria_passed = int(c1_corr) + int(c2_thin) + int(c3_lamina) + int(c4_chars) + int(c5_topology)
    blind_reader_score = round(
        0.22 * min(1.0, r_all_depths / 0.32)
        + 0.22 * min(1.0, mean_drop / 62.0)
        + 0.22 * min(1.0, mean_tpl / 0.58)
        + 0.18 * min(1.0, mean_concord / 0.68)
        + 0.16 * min(1.0, (len(all_decoded_boxes) / 28.0) * (1.0 + 0.15 * len(uniq_words))),
        3,
    )

    return {
        "scroll_key": scroll_key,
        "cand_id": cand_id,
        "window_mm": win_mm,
        "fiber_line_tilt_deg": tilt_deg,
        "orient": orient,
        "canonical_2d_tf": tf_mode,
        "best_single_depth": best_d_key,
        "inner_ds_used": lamina_maps_in["inner_ds_used"],
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
        # Top PHerc1203 candidates (including newly uncovered high-SNR segments!)
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
        # Top PHerc0800 candidates
        "ag_target_patch_ext_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_ext_m7",
        # Top 77keV & Multi-scroll 4x-TTA promoted candidates
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_m7",
        "thin_wrap_target_m7",
        "thin_wrap_target_m7",
    }
    seg_dirs = [
        sd for sd in sorted(STAGE_I_DIR.glob("w*/*/*"))
        if sd.is_dir() and (sd / "metrics_stage_i.json").exists() and sd.name in priority_set
    ]
    import concurrent.futures
    print(f"[Stage L] Running Ultra-Precision Ink Detector (L_1 = L_2+L_3+L_4+L_5+L_6+L_8) on {len(seg_dirs)} candidates in parallel (8 workers)...", flush=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for idx, r in enumerate(ex.map(decode_stage_l_candidate, seg_dirs), 1):
            if r is not None:
                results.append(r)
                print(
                    f"  [{idx:2d}/{len(seg_dirs)}] {r['scroll_key']}/{r['cand_id']:28s} "
                    f"| d={r['inner_ds_used']} bg={r['bg_ds_used']} | tf={r['canonical_2d_tf']:6s} "
                    f"| tilt={r['fiber_line_tilt_deg']:+5.1f}deg | chars={r['total_decoded_chars']:2d} ({r['num_lines_decoded']}L) "
                    f"| drop=-{r['mean_box_drop_pm4_pct']:4.1f}% | conc={r['mean_slot_concordance']:.2f} "
                    f"| tpl={r['mean_tpl_score']:.2f} | phi={r['phi_rate_pct']:3.1f}% | K4={r['k4_criteria_passed']}/5 (BRS={r['blind_reader_score']:.2f}) "
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
    mean_drop_top10 = float(np.mean([r["mean_box_drop_pm4_pct"] for r in top10]))
    mean_conc_top10 = float(np.mean([r["mean_slot_concordance"] for r in top10]))
    bet13_pass = (mean_drop_top10 >= 62.0) and (mean_conc_top10 >= 0.60)

    mean_tpl_top10 = float(np.mean([r["mean_tpl_score"] for r in top10]))
    bet14_pass = mean_tpl_top10 >= 0.54

    mean_phi_all = float(np.mean([r["phi_rate_pct"] for r in results]))
    bet15_pass = mean_phi_all <= 7.0

    k4_ge4 = [r for r in results if r["k4_criteria_passed"] >= 4]
    mean_chars_top10 = float(np.mean([r["total_decoded_chars"] for r in top10]))
    bet16_pass = (len(k4_ge4) >= 10) and (mean_chars_top10 >= 34.0)

    summary = {
        "bets_settlement": {
            "Bet_13_dstar_1vox_4xtta_and_trimodel_concordance": {
                "status": "PASS" if bet13_pass else "FAIL",
                "mean_drop_pm4_top10_pct": round(mean_drop_top10, 1),
                "mean_slot_concordance_top10": round(mean_conc_top10, 3),
            },
            "Bet_14_zero_hallucination_pure_ink_fidelity": {
                "status": "PASS" if bet14_pass else "FAIL",
                "synthetic_cv2_lines_drawn": 0,
                "mean_tpl_score_top10": round(mean_tpl_top10, 3),
            },
            "Bet_15_corridor_anchored_phi_suppression": {
                "status": "PASS" if bet15_pass else "FAIL",
                "mean_phi_rate_pct": round(mean_phi_all, 2),
                "target_max_phi_rate_pct": 7.0,
            },
            "Bet_16_expanded_6line_columns_and_new_segments": {
                "status": "PASS" if bet16_pass else "FAIL",
                "k4_ge4_columns_count": len(k4_ge4),
                "mean_chars_top10": round(mean_chars_top10, 1),
            },
        },
        "decoded_candidates": results,
    }

    out_json = STAGE_L_DIR / "leaderboard_stage_l_ultraprecise.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[Stage L] Saved {len(results)} ultra-precision column dossiers to {out_json}", flush=True)
    print(f"[Stage L] Bets Settlement: {json.dumps(summary['bets_settlement'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
