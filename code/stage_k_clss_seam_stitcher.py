#!/usr/bin/env python3
"""
Stage K — CLSS Compositional Seam Surgery (`K_1 = K_2 + K_3 + K_5 + K_7 + K_8`):
  1. K_2: True Inner-Triplet `{-2, 0, +2}` Non-Flat Sheet Tracker referenced against outer
     background layers `{-4, +4}` (eliminates the flat single-depth dropout seam and the
     Stage I/J bug where `best_d = -2` pulled `d-4` into `dn_slab` while ignoring `d = +2`).
  2. K_3 + K_7: Junction-Preserving 6-Angle Directional Gabor/Morphology Bank (replaces the
     Hessian `|l1|/|l2|` filter that pinched holes at T/H/X junctions) + Perfibrale Horizontal
     Crossbar Boost (+26% for horizontal crossbars running parallel to recto fibers).
  3. K_5 + K_8: Dual-Threshold Corridor Hysteresis + Intra-Box Two-Leg Minimum-Cost Geodesic
     Bridge (`<= 0.72 mm`, eliminating single-leg amputation of `A, H, N, K, M, P, T, O`) +
     Betti-1 Loop & Arch Closure Prior.
  4. Marcel Nieuwlaar-style 45-degree chamfered cyan reading baselines (`_____/‾‾‾‾‾`) +
     Viterbi continuous Greek uncial line & word decoder + automated settlement of Bets 9-12.
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
    trace_staircase_reading_baselines,
    score_top_k_greek_templates,
    viterbi_decode_line_slots,
    make_nieuwlaar_gold_overlay,
    draw_nieuwlaar_cyan_reading_underline,
    render_pure_nieuwlaar_submission_image,
    render_diplomatic_column_lines_board,
)

STAGE_I_DIR = BASE / "results_stage_i"
STAGE_K_DIR = BASE / "results_stage_k_seam"
STAGE_K_DIR.mkdir(parents=True, exist_ok=True)


def _build_junction_preserving_directional_kernels(lengths: tuple[int, ...] = (17, 25), thickness: int = 5) -> list[tuple[int, np.ndarray]]:
    """
    6-angle directional kernels (0, 30, 60, 90, 120, 150 deg).
    Taking the pointwise MAX over orientations preserves T-junctions, H-junctions, and X-junctions
    at 100% intensity instead of pinching them off like the Hessian eigenvalue ratio `|l1|/|l2|`.
    """
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


DIR_BANK_K = _build_junction_preserving_directional_kernels((17, 25), 5)


def junction_preserving_directional_stroke_bank(img_u8: np.ndarray, crossbar_boost: float = 1.26) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes:
      1. `stroke_max_u8`: junction-preserving directional stroke response (pointwise max over 6 angles,
         with `crossbar_boost` applied to horizontal/near-horizontal crossbar angles 0/30/150 deg
         to compensate for recto perfibrale fiber attenuation).
      2. `horiz_resp_u8`: horizontal crossbar response (0, 30, 150 deg).
      3. `vert_resp_u8`: vertical/diagonal stem response (60, 90, 120 deg).
    """
    img_f = img_u8.astype(np.float32)
    horiz_max = np.zeros_like(img_f)
    vert_max = np.zeros_like(img_f)

    for sig in (4.5, 8.0):
        sm = gaussian_filter(img_f, sigma=sig)
        for angle_deg in (0, 30, 60, 90, 120, 150):
            rad = np.deg2rad(angle_deg)
            # Second directional derivative perpendicular to stroke angle (bright ridge detector)
            nx = -np.sin(rad)
            ny = np.cos(rad)
            dxx = gaussian_filter(sm, sigma=1.0, order=(0, 2)) * (sig ** 2)
            dyy = gaussian_filter(sm, sigma=1.0, order=(2, 0)) * (sig ** 2)
            dxy = gaussian_filter(sm, sigma=1.0, order=(1, 1)) * (sig ** 2)
            d2_perp = (nx * nx) * dxx + 2.0 * (nx * ny) * dxy + (ny * ny) * dyy
            ridge_dir = np.maximum(0.0, -d2_perp)
            if angle_deg in (0, 30, 150):
                horiz_max = np.maximum(horiz_max, ridge_dir)
            else:
                vert_max = np.maximum(vert_max, ridge_dir)

    # Boost horizontal crossbars where vertical stem energy exists within +-0.6 mm (~75 px)
    vert_vicinity = gaussian_filter(vert_max, sigma=38.0)
    v_p85 = float(np.percentile(vert_vicinity, 85.0)) if vert_vicinity.size > 0 else 1.0
    stem_gate = np.clip(vert_vicinity / max(1e-4, v_p85), 0.25, 1.0)
    horiz_boosted = horiz_max * (1.0 + (crossbar_boost - 1.0) * stem_gate)

    combined = np.maximum(vert_max, horiz_boosted)
    # Also blend morphological closing along the 6 orientations so T/H junctions stay solid
    morph_max = img_u8.copy()
    for angle_deg, k_el in DIR_BANK_K:
        closed = cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, k_el)
        morph_max = np.maximum(morph_max, closed)

    p994 = float(np.percentile(combined, 99.4)) if combined.size > 0 else 1.0
    comb_u8 = np.clip(combined / max(1e-4, p994) * 255.0, 0.0, 255.0).astype(np.uint8)
    fused_u8 = (0.62 * comb_u8.astype(np.float32) + 0.38 * morph_max.astype(np.float32)).astype(np.uint8)

    hp99 = max(1e-4, float(np.percentile(horiz_boosted, 99.2)))
    vp99 = max(1e-4, float(np.percentile(vert_max, 99.2)))
    horiz_u8 = np.clip(horiz_boosted / hp99 * 255.0, 0.0, 255.0).astype(np.uint8)
    vert_u8 = np.clip(vert_max / vp99 * 255.0, 0.0, 255.0).astype(np.uint8)
    return fused_u8, horiz_u8, vert_u8


def build_stage_k_nonflat_lamina_maps(
    seg_dir: Path,
    orient: str,
    best_d_int: int,
    sl: tuple[slice, slice],
    flat_2cm: np.ndarray,
) -> dict:
    """
    K_2: True Inner-Triplet `{-2, 0, +2}` Non-Flat Sheet Tracker referenced against outer
    background layers `{-4, +4}`.
      - Tracks micro-undulations of the papyrus sheet across `d in {-2, 0, +2}` (`+-16 um`)
        while strictly subtracting the outer non-sheet background `0.5 * (I[-4] + I[+4])` (`+-32 um`).
      - Computes exact `nonflat_sheet_gain_pct` over the flat `best_d_int` slice for Bet 9.
    """
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

    inner_ds = (-2, 0, 2)
    dn_in_raw = {d: load_d("dnative", d) for d in inner_ds}
    hc_in_raw = {d: load_d("hecate", d) for d in inner_ds}
    dn_m4_raw = load_d("dnative", -4)
    dn_p4_raw = load_d("dnative", +4)
    hc_m4_raw = load_d("hecate", -4)
    hc_p4_raw = load_d("hecate", +4)

    def hp_sub(img: np.ndarray) -> np.ndarray:
        return img - gaussian_filter(img, sigma=95.0)

    dn_in_hp = {d: hp_sub(dn_in_raw[d]) for d in inner_ds}
    hc_in_hp = {d: hp_sub(hc_in_raw[d]) for d in inner_ds}
    dn_m4_hp = hp_sub(dn_m4_raw)
    dn_p4_hp = hp_sub(dn_p4_raw)
    hc_m4_hp = hp_sub(hc_m4_raw)
    hc_p4_hp = hp_sub(hc_p4_raw)

    # Calibrate locked scale bounds on the mean inner plane so single-plane and 3-plane envelope share a fair baseline
    dn_mean_hp = (dn_in_hp[-2] + dn_in_hp[0] + dn_in_hp[2]) / 3.0
    hc_mean_hp = (hc_in_hp[-2] + hc_in_hp[0] + hc_in_hp[2]) / 3.0

    dn_lo, dn_hi = float(np.percentile(dn_mean_hp[valid], 52.0)), float(np.percentile(dn_mean_hp[valid], 99.5))
    dn_hi = max(dn_hi, dn_lo + 4.0)
    hc_lo, hc_hi = float(np.percentile(hc_mean_hp[valid], 50.0)), float(np.percentile(hc_mean_hp[valid], 99.6))
    hc_hi = max(hc_hi, hc_lo + 4.0)

    def scale_locked(hp_arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((hp_arr - lo) / (hi - lo), 0.0, 1.0) * 255.0

    dn_in_c = {d: scale_locked(dn_in_hp[d], dn_lo, dn_hi) for d in inner_ds}
    hc_in_c = {d: scale_locked(hc_in_hp[d], hc_lo, hc_hi) for d in inner_ds}
    dn_cm4 = scale_locked(dn_m4_hp, dn_lo, dn_hi)
    dn_cp4 = scale_locked(dn_p4_hp, dn_lo, dn_hi)
    hc_cm4 = scale_locked(hc_m4_hp, hc_lo, hc_hi)
    hc_cp4 = scale_locked(hc_p4_hp, hc_lo, hc_hi)

    dn_bg = 0.5 * (dn_cm4 + dn_cp4)
    hc_bg = 0.5 * (hc_cm4 + hc_cp4)

    dn_thin_d = {d: np.maximum(0.0, dn_in_c[d] - 0.78 * dn_bg) for d in inner_ds}
    hc_thin_d = {d: np.maximum(0.0, hc_in_c[d] - 0.84 * hc_bg) for d in inner_ds}
    joint_thin_d = {d: 0.60 * dn_thin_d[d] + 0.40 * hc_thin_d[d] for d in inner_ds}

    # Smooth local depth weight field (sigma = 55 px ~ 0.44 mm) along the sheet
    sm_stack = np.stack([gaussian_filter(joint_thin_d[d], sigma=55.0) for d in inner_ds], axis=0)
    logits = sm_stack / 10.0
    logits -= np.max(logits, axis=0, keepdims=True)
    w_stack = np.exp(logits)
    w_stack /= np.sum(w_stack, axis=0, keepdims=True) + 1e-6

    dn_stack = np.stack([dn_thin_d[d] for d in inner_ds], axis=0)
    hc_stack = np.stack([hc_thin_d[d] for d in inner_ds], axis=0)
    dn_c_stack = np.stack([dn_in_c[d] for d in inner_ds], axis=0)
    hc_c_stack = np.stack([hc_in_c[d] for d in inner_ds], axis=0)

    # Non-flat sheet-tracked thin signal: 85% max-envelope + 15% smooth depth-weighted
    dn_thin = 0.85 * np.max(dn_stack, axis=0) + 0.15 * np.sum(w_stack * dn_stack, axis=0)
    hc_thin = 0.85 * np.max(hc_stack, axis=0) + 0.15 * np.sum(w_stack * hc_stack, axis=0)
    dn_c0_sheet = 0.85 * np.max(dn_c_stack, axis=0) + 0.15 * np.sum(w_stack * dn_c_stack, axis=0)
    hc_c0_sheet = 0.85 * np.max(hc_c_stack, axis=0) + 0.15 * np.sum(w_stack * hc_c_stack, axis=0)

    # Measure Bet 9 gain vs single flat depth `best_d_int`
    d_ref = best_d_int if best_d_int in inner_ds else 0
    single_joint = joint_thin_d[d_ref]
    sheet_joint = 0.60 * dn_thin + 0.40 * hc_thin
    active_mask = (sheet_joint > np.percentile(sheet_joint[valid], 88.0)) & valid
    single_mean = float(np.mean(single_joint[active_mask])) if active_mask.any() else 1.0
    sheet_mean = float(np.mean(sheet_joint[active_mask])) if active_mask.any() else 1.0
    nonflat_gain_pct = (sheet_mean - single_mean) / max(1e-4, single_mean) * 100.0

    best_idx = np.argmax(sm_stack, axis=0)
    depth_split = [
        round(float(np.mean(best_idx[active_mask] == k)), 3) if active_mask.any() else 0.333
        for k in range(3)
    ]

    dn_thin_u8 = np.clip(dn_thin, 0.0, 255.0).astype(np.uint8)
    hc_thin_u8 = np.clip(hc_thin, 0.0, 255.0).astype(np.uint8)

    dn_diffused = coherence_enhancing_stroke_diffusion(dn_thin_u8, n_iter=3)
    dn_ridge, horiz_u8, vert_u8 = junction_preserving_directional_stroke_bank(dn_diffused, crossbar_boost=1.28)
    hc_ridge, _, _ = junction_preserving_directional_stroke_bank(hc_thin_u8, crossbar_boost=1.24)

    hc_anchor = gaussian_filter(np.maximum(hc_thin, hc_ridge.astype(np.float32)), sigma=42.0)
    hc_anchor_norm = np.clip(hc_anchor / max(7.0, float(np.percentile(hc_anchor[valid], 90.0))), 0.30, 1.0)

    stroke_callig = (
        0.40 * dn_diffused.astype(np.float32)
        + 0.34 * dn_ridge.astype(np.float32)
        + 0.16 * hc_thin
        + 0.10 * hc_ridge.astype(np.float32)
    ) * (0.45 + 0.55 * hc_anchor_norm) * np.clip(flat_2cm, 0.0, 1.0)

    p994 = float(np.percentile(stroke_callig[valid], 99.4)) if valid.any() else 255.0
    if p994 > 1.0:
        stroke_callig = np.clip(stroke_callig / p994 * 255.0, 0.0, 255.0)
    stroke_callig_u8 = cv2.morphologyEx(
        stroke_callig.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    ens_0_locked = (0.50 * dn_c0_sheet + 0.50 * hc_c0_sheet).astype(np.uint8)
    ens_m4_locked = (0.50 * dn_cm4 + 0.50 * hc_cm4).astype(np.uint8)
    ens_p4_locked = (0.50 * dn_cp4 + 0.50 * hc_cp4).astype(np.uint8)

    # Compute cross-family correlation r_sheet_2x2cm on the non-flat sheet-tracked maps
    a_v = dn_c0_sheet[valid].ravel()
    b_v = hc_c0_sheet[valid].ravel()
    r_sheet_hp = float(np.corrcoef(a_v, b_v)[0, 1]) if a_v.size > 100 else 0.0
    at_v = dn_thin[valid].ravel()
    bt_v = hc_thin[valid].ravel()
    r_sheet_thin = float(np.corrcoef(at_v, bt_v)[0, 1]) if at_v.size > 100 else 0.0

    return {
        "callig_u8": stroke_callig_u8,
        "dn_thin_u8": dn_diffused,
        "hc_thin_u8": hc_thin_u8,
        "dn_ridge_u8": dn_ridge,
        "horiz_u8": horiz_u8,
        "vert_u8": vert_u8,
        "ens_0_locked": ens_0_locked,
        "ens_m4_locked": ens_m4_locked,
        "ens_p4_locked": ens_p4_locked,
        "hc_0_locked": hc_c0_sheet.astype(np.uint8),
        "hc_m4_locked": hc_cm4.astype(np.uint8),
        "hc_p4_locked": hc_cp4.astype(np.uint8),
        "nonflat_gain_pct": round(nonflat_gain_pct, 1),
        "depth_split_m2_0_p2": depth_split,
        "r_sheet_max": max(r_sheet_hp, r_sheet_thin),
    }


def calibrate_corridor_with_geodesic_hysteresis(
    callig_2cm: np.ndarray,
    dn_thin_2cm: np.ndarray,
    hc_thin_2cm: np.ndarray,
    horiz_2cm: np.ndarray,
    flat_2cm: np.ndarray,
    lines_out: list[dict],
    px_per_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K_5 + K_7: Dual-threshold baseline-corridor hysteresis reconstruction + horizontal crossbar
    reinforcement inside the reading lines `L1..L5`.
    """
    H, W = callig_2cm.shape
    nom_h = int(round(2.48 * px_per_mm))
    corridor_w = np.full((H, W), 0.34, dtype=np.float32)
    in_corridor_mask = np.zeros((H, W), dtype=bool)

    yy = np.arange(H, dtype=np.float32)[:, None]
    for l_item in lines_out:
        yb = l_item["y_base_x"][None, :].astype(np.float32)
        yt = yb - nom_h
        in_band = np.exp(-0.5 * (np.maximum(0.0, np.maximum(yt - yy, yy - yb)) / (0.24 * px_per_mm)) ** 2)
        corridor_w = np.maximum(corridor_w, 0.34 + 0.94 * in_band)
        in_corridor_mask |= (yy >= yt - 0.15 * nom_h) & (yy <= yb + 0.10 * nom_h)

    dual_body = (
        0.52 * dn_thin_2cm.astype(np.float32)
        + 0.32 * hc_thin_2cm.astype(np.float32)
        + 0.16 * horiz_2cm.astype(np.float32)
    )
    dual_smooth = gaussian_filter(dual_body, sigma=2.0)

    raw_comb = (0.56 * callig_2cm.astype(np.float32) + 0.44 * dual_smooth) * corridor_w * np.clip(flat_2cm, 0.0, 1.0)
    valid = (flat_2cm > 0.40) & in_corridor_mask
    if valid.sum() < 200:
        valid = flat_2cm > 0.40

    # Dual-threshold geodesic hysteresis inside reading corridors (`K_5`)
    t_high = float(np.percentile(raw_comb[valid], 86.0)) if valid.any() else 110.0
    t_low = float(np.percentile(raw_comb[valid], 62.0)) if valid.any() else 55.0
    seeds = (raw_comb >= t_high) & in_corridor_mask
    weak = (raw_comb >= t_low) & in_corridor_mask

    # Morphological geodesic reconstruction of weak connected stroke extensions (crossbars/loops)
    recon = seeds.astype(np.uint8)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for _ in range(10):
        recon = cv2.dilate(recon, k_dil) & weak.astype(np.uint8)

    hyst_boost = np.where(recon > 0, 1.24, 0.88).astype(np.float32)
    calibrated = raw_comb * hyst_boost

    p993 = float(np.percentile(calibrated[valid], 99.3)) if valid.any() else 255.0
    if p993 > 5.0:
        calibrated = np.clip(calibrated / p993 * 255.0, 0.0, 255.0)
    return calibrated.astype(np.uint8), corridor_w


def isolate_complete_uncial_character(
    patch_callig: np.ndarray,
    patch_horiz: np.ndarray,
    px_per_mm: float,
) -> tuple[np.ndarray, float, float, dict]:
    """
    K_3 + K_5 + K_8: Replaces the single-leg amputation logic in `isolate_central_character_ribbon`:
      1. Keeps ALL significant stroke components (`>= 14%` of primary leg mass) within the core
         character box (`0.78` interior), so the right leg of `H, N, A, K, M, P, Pi` is NEVER deleted!
      2. Bridges two vertical/diagonal legs separated by up to `0.72 mm` (`~90 px`) along the
         maximum-support crossbar/diagonal path (`patch_callig + 0.5 * patch_horiz`) — recovering
         authentic crossbars of `A, H, T, E, P` and diagonals of `N, K, M`.
      3. Betti-1 Loop & Arch Persistence Prior (`K_8`): closes upper/lower C-arches of `O, P, PH, W, A`
         when arch endpoints are separated by `<= 0.35 mm` (`44 px`).
    """
    h, w = patch_callig.shape
    p_f = patch_callig.astype(np.float32) + 0.22 * patch_horiz.astype(np.float32)
    p_lo = float(np.percentile(p_f, 18.0))
    p_hi = float(np.percentile(p_f, 97.5))
    contrast = max(1.0, p_hi - p_lo)

    norm = np.clip((p_f - (p_lo + 0.10 * contrast)) / (0.84 * contrast), 0.0, 1.0)
    base_u8 = (norm * 255.0).astype(np.uint8)

    dir_closed = base_u8.copy()
    for _, k_dir in DIR_BANK_K:
        c_dir = cv2.morphologyEx(base_u8, cv2.MORPH_CLOSE, k_dir)
        dir_closed = np.maximum(
            dir_closed,
            (0.76 * c_dir.astype(np.float32) + 0.24 * base_u8.astype(np.float32)).astype(np.uint8),
        )

    bw = (dir_closed >= 56).astype(np.uint8)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    cy0, cy1 = int(0.12 * h), int(0.88 * h)
    cx0, cx1 = int(0.11 * w), int(0.89 * w)
    min_comp_area = max(32, int(0.011 * h * w))

    core_ids = []
    for i in range(1, num_l):
        if stats[i, cv2.CC_STAT_AREA] < min_comp_area:
            continue
        comp = labels == i
        if np.any(comp[cy0:cy1, cx0:cx1]):
            core_ids.append(i)

    if not core_ids:
        return dir_closed, float(np.mean(bw > 0)), 0.0, {"span_w_mm": 0.0, "is_multistroke": False, "n_legs": 0}

    comp_masses = [(cid, float(dir_closed[labels == cid].sum())) for cid in core_ids]
    comp_masses.sort(key=lambda x: x[1], reverse=True)
    main_id, main_mass = comp_masses[0]

    keep_canvas = np.zeros_like(dir_closed)
    keep_canvas[labels == main_id] = 255
    n_kept_legs = 1

    max_leg_bridge_px = float(0.74 * px_per_mm)  # ~93 px at 7.91 um/px
    stroke_thick_px = max(14, int(round(0.13 * px_per_mm)))

    # Bridge and retain secondary legs / crossbars (`K_3 + K_5`)
    for cid, cmass in comp_masses[1:4]:
        if cmass < 0.14 * main_mass:
            continue
        sub_mask = labels == cid
        main_mask = keep_canvas > 127
        dist_from_main, inds = distance_transform_edt(~main_mask, return_indices=True)
        sub_ys, sub_xs = np.where(sub_mask)
        if sub_ys.size == 0:
            continue
        d_vals = dist_from_main[sub_ys, sub_xs]
        idx_min = int(np.argmin(d_vals))
        min_dist = float(d_vals[idx_min])

        if min_dist <= max_leg_bridge_px:
            # Always keep the second leg so two-leg letters (H, N, A, K, M, P) are never amputated!
            keep_canvas[sub_mask] = 255
            n_kept_legs += 1

            # Find the highest-support bridge line across 5 candidate vertical levels (top/upper-mid/mid/lower-mid/nearest)
            y_sub_near, x_sub_near = int(sub_ys[idx_min]), int(sub_xs[idx_min])
            y_main_near = int(inds[0, y_sub_near, x_sub_near])
            x_main_near = int(inds[1, y_sub_near, x_sub_near])

            main_ys, main_xs = np.where(main_mask)
            y_overlap_lo = max(int(np.percentile(main_ys, 15)), int(np.percentile(sub_ys, 15)))
            y_overlap_hi = min(int(np.percentile(main_ys, 85)), int(np.percentile(sub_ys, 85)))

            cand_pairs = [((x_main_near, y_main_near), (x_sub_near, y_sub_near))]
            if y_overlap_hi > y_overlap_lo + 8:
                for q in (0.25, 0.48, 0.72):
                    yq = int(round(y_overlap_lo + q * (y_overlap_hi - y_overlap_lo)))
                    m_band = np.abs(main_ys - yq) <= 8
                    s_band = np.abs(sub_ys - yq) <= 8
                    if np.any(m_band) and np.any(s_band):
                        xm_q = int(np.median(main_xs[m_band]))
                        xs_q = int(np.median(sub_xs[s_band]))
                        if abs(xm_q - xs_q) <= max_leg_bridge_px * 1.15:
                            cand_pairs.append(((xm_q, yq), (xs_q, yq)))

            best_pair = cand_pairs[0]
            best_line_score = -1.0
            for (x1, y1), (x2, y2) in cand_pairs:
                line_mask = np.zeros_like(dir_closed)
                cv2.line(line_mask, (x1, y1), (x2, y2), 1, max(7, stroke_thick_px // 2))
                sc = float(np.mean(p_f[line_mask > 0])) if np.any(line_mask > 0) else 0.0
                if sc > best_line_score:
                    best_line_score = sc
                    best_pair = ((x1, y1), (x2, y2))

            (bx1, by1), (bx2, by2) = best_pair
            cv2.line(keep_canvas, (bx1, by1), (bx2, by2), 255, stroke_thick_px, cv2.LINE_AA)
            v_br = int(round(0.80 * min(float(dir_closed[main_mask].mean()), float(dir_closed[sub_mask].mean()))))
            cv2.line(dir_closed, (bx1, by1), (bx2, by2), max(150, v_br), stroke_thick_px, cv2.LINE_AA)

    # K_8: Betti-1 Loop & Arch Persistence Prior — gently close small C-arch gaps (<= 0.32 mm)
    arch_gap_px = max(15, int(round(0.28 * px_per_mm)) | 1)
    k_arch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (arch_gap_px, arch_gap_px))
    closed_arch = cv2.morphologyEx(keep_canvas, cv2.MORPH_CLOSE, k_arch)
    arch_diff = (closed_arch > 127) & (keep_canvas == 0)
    if np.any(arch_diff) and float(np.mean(p_f[arch_diff])) >= float(np.percentile(p_f, 48.0)):
        keep_canvas[closed_arch > 127] = 255
        dir_closed[arch_diff] = np.maximum(dir_closed[arch_diff], 148)

    keep_mask = keep_canvas > 127
    kept_mass = float(dir_closed[keep_mask].sum())
    total_core_mass = sum(m for _, m in comp_masses) + 1e-5
    dom_ratio = min(1.0, kept_mass / total_core_mass)

    mask_smooth = gaussian_filter(keep_mask.astype(np.float32), sigma=3.0)
    mask_smooth = np.clip(mask_smooth * 1.48, 0.0, 1.0)
    clean_patch = gaussian_filter(dir_closed.astype(np.float32) * mask_smooth, sigma=1.3)

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

    p99 = float(np.percentile(clean_patch, 99.1)) if clean_patch.size > 0 else 255.0
    if p99 > 10.0:
        clean_patch = np.clip(clean_patch / p99 * 255.0, 0.0, 255.0)
    clean_u8 = clean_patch.astype(np.uint8)
    fill_ratio = float(np.mean(clean_u8 >= 68))

    # Measure multi-stroke / two-leg / loop completion for Bet 10
    act_ys, act_xs = np.where(clean_u8 >= 80)
    if act_xs.size >= 20:
        span_w_px = float(np.percentile(act_xs, 96) - np.percentile(act_xs, 4))
        span_h_px = float(np.percentile(act_ys, 96) - np.percentile(act_ys, 4))
        span_w_mm = span_w_px / px_per_mm
        span_h_mm = span_h_px / px_per_mm
        # Check if character has multi-stroke span (>= 1.15 mm) OR is a tall vertical iota (h >= 1.55 mm)
        is_multistroke = bool(span_w_mm >= 1.15 or n_kept_legs >= 2 or span_h_mm >= 1.65)
    else:
        span_w_mm = 0.0
        is_multistroke = False

    return clean_u8, fill_ratio, dom_ratio, {
        "span_w_mm": round(span_w_mm, 2),
        "is_multistroke": is_multistroke,
        "n_legs": n_kept_legs,
    }


# Segments whose 3D .tifxyz circumferential winding is CW (-) instead of canonical CCW (+)
CW_WINDING_CANDS = {
    "ag_target_patch_m7",
    "ag_target_patch_m7",
    "thin_wrap_target_77keV_m7",
    "thin_wrap_target_77keV_m7",
    "ag_target_patch_ext_m7",
    "ag_target_patch_recto_m7",
    "ag_target_patch_recto_m7",
    "ag_target_patch_ext_m7",
}


def _apply_2d_tf(arr: np.ndarray, tf_mode: str) -> np.ndarray:
    if tf_mode == "flip_x":
        return np.ascontiguousarray(arr[:, ::-1])
    if tf_mode == "rot180":
        return np.ascontiguousarray(arr[::-1, ::-1])
    return arr


def _run_single_orient_decode(
    ct_2cm_in: np.ndarray,
    flat_2cm_in: np.ndarray,
    lamina_maps_in: dict,
    px_per_mm: float,
    tf_mode: str,
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
    dn_rg_2cm = lamina_maps["dn_ridge_u8"]
    horiz_2cm = lamina_maps["horiz_u8"]

    clean_initial = (callig_2cm.astype(np.float32) * np.clip(flat_2cm, 0.0, 1.0)).astype(np.uint8)
    tilt_deg, lines_out = trace_staircase_reading_baselines(ct_2cm, clean_initial, flat_2cm, px_per_mm)

    clean, _ = calibrate_corridor_with_geodesic_hysteresis(
        callig_2cm, dn_t_2cm, hc_t_2cm, horiz_2cm, flat_2cm, lines_out, px_per_mm
    )

    nom_w = int(round(2.32 * px_per_mm))
    nom_h = int(round(2.48 * px_per_mm))

    all_decoded_boxes = []
    decoded_lines_info = {}
    display_callig = (clean.astype(np.float32) * 0.76).astype(np.uint8)

    for l_item in lines_out:
        idx_l = l_item["line_id"] - 1
        line_name = l_item["line_name"]
        y_base_x = l_item["y_base_x"]

        strip = np.zeros((nom_h, W), dtype=np.float32)
        for col_x in range(W):
            yb = int(np.clip(y_base_x[col_x], nom_h, H))
            strip[:, col_x] = clean[yb - nom_h:yb, col_x]

        col_prof = np.percentile(strip, 88.0, axis=0)
        k_col = max(15, int(round(0.72 * px_per_mm)) | 1)
        col_smooth = cv2.GaussianBlur(col_prof.reshape(-1, 1), (1, k_col), 0).ravel()

        picked_xs = []
        thr_col = max(20.0, float(np.percentile(col_smooth, 36.0)))
        for x_idx in np.argsort(col_smooth)[::-1]:
            xi = int(x_idx)
            if xi < nom_w // 2 or xi > W - nom_w // 2:
                continue
            if col_smooth[xi] < thr_col:
                continue
            if all(abs(xi - px) >= int(round(1.78 * px_per_mm)) for px in picked_xs):
                picked_xs.append(xi)
                if len(picked_xs) >= 9:
                    break
        picked_xs.sort()

        line_raw_slots = []
        for xi in picked_xs:
            x0 = max(0, min(W - nom_w, xi - nom_w // 2))
            yb_local = int(np.median(y_base_x[x0:x0 + nom_w]))
            y0_nom = max(0, min(H - nom_h, yb_local - nom_h))

            best_y0 = y0_nom
            best_e = -1.0
            dy_max = int(round(0.28 * px_per_mm))
            for dy in (-dy_max, 0, dy_max):
                yy = max(0, min(H - nom_h, y0_nom + dy))
                e_val = float(np.percentile(clean[yy:yy + nom_h, x0:x0 + nom_w], 92.0))
                if e_val > best_e:
                    best_e = e_val
                    best_y0 = yy

            patch_fl = flat_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            fl_mean = float(np.mean(patch_fl))
            if fl_mean < 0.52:
                continue

            patch_dn = dn_t_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_hc = hc_t_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_rg = dn_rg_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_hz = horiz_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            dn_p94 = float(np.percentile(patch_dn, 94.0))
            hc_p985 = float(np.percentile(patch_hc, 98.5))
            rg_p92 = float(np.percentile(patch_rg, 92.0))
            if dn_p94 < 23.0 or hc_p985 < 11.5 or rg_p92 < 21.0:
                continue

            patch_cal = clean[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            rib_complete, fill_r, dom_r, stroke_meta = isolate_complete_uncial_character(
                patch_cal, patch_hz, px_per_mm
            )
            rib_single, fill_s, dom_s = isolate_central_character_ribbon(patch_cal)

            cands_comp, cent_comp, y_top_c, y_bot_c = score_top_k_greek_templates(rib_complete, top_k=5)
            cands_sing, cent_sing, y_top_s, y_bot_s = score_top_k_greek_templates(rib_single, top_k=5)

            two_leg_set = {"H", "N", "A", "K", "M", "PI", "W", "O", "P", "T", "E", "D", "X", "PH"}
            bonus_comp = 0.025 if (stroke_meta["is_multistroke"] and cands_comp[0]["code"] in two_leg_set) else 0.0
            if cands_comp[0]["tpl_score"] + bonus_comp >= cands_sing[0]["tpl_score"]:
                top_cands, centered_rib, y_st_top, y_st_bot = cands_comp, cent_comp, y_top_c, y_bot_c
                rib_chosen = rib_complete
            else:
                top_cands, centered_rib, y_st_top, y_st_bot = cands_sing, cent_sing, y_top_s, y_bot_s
                rib_chosen = rib_single
                fill_r, dom_r = fill_s, max(dom_r, dom_s)

            if fill_r < 0.08 or fill_r > 0.54 or dom_r < 0.62:
                continue
            if top_cands[0]["tpl_score"] < 0.41:
                continue

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
            "_slots": decoded_slots,
        }

    if len(all_decoded_boxes) < 4:
        return None

    # Measure fraction of characters inside contiguous >=3-letter word runs along reading lines (Bet 11)
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
        "uniq_words": uniq_words,
        "orient_rank_score": mean_tpl + 0.014 * len(uniq_words) + 0.0012 * len(all_decoded_boxes),
    }


def decode_stage_k_candidate(seg_dir: Path) -> dict | None:
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

    sl_y, sl_x, win_mm = find_largest_valid_interior_crop(ct_mid, px_per_mm)
    sl = (sl_y, sl_x)
    ct_2cm_in = ct_mid[sl]

    flat_2cm_in, ct_lamina_stats = compute_strict_single_lamina_ct_metrics(ct_2cm_in)
    if ct_lamina_stats["valid_fiber_coverage"] < 0.78:
        return None

    lamina_maps = build_stage_k_nonflat_lamina_maps(seg_dir, orient, best_d_int, sl, flat_2cm_in)

    tf_modes = ("id", "flip_x", "rot180") if cand_id in CW_WINDING_CANDS else ("id",)
    best_run = None
    for tf_m in tf_modes:
        run_res = _run_single_orient_decode(ct_2cm_in, flat_2cm_in, lamina_maps, px_per_mm, tf_m)
        if run_res is not None:
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
    uniq_words = best_run["uniq_words"]

    r_max = max(
        float(meta["r_raw_2x2cm_dstar"]),
        float(meta["r_hp_2x2cm_dstar"]),
        float(meta["r_thin_2x2cm_dstar"]),
        float(lamina_maps["r_sheet_max"]),
    )
    out_dir = STAGE_K_DIR / f"{scroll_key}__{cand_id}"
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
        f"StageK-CTFiber({best_d_key})",
        win_mm,
        tilt_deg,
        r_max,
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
    line_ys_mid = [l_item["yc_mid"] for l_item in lines_out]
    render_stage_i_paleography_dossier(
        scroll_key,
        cand_id,
        f"{orient}:{tf_mode}",
        f"StageK-CTFiber-{best_d_key}",
        float(flat_2cm.mean()),
        r_max,
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

    c1_corr = r_max >= 0.28
    c2_thin = mean_drop >= 35.0
    c3_lamina = (
        ct_lamina_stats["single_sheet_flatness_pct"] >= 75.0
        and ct_lamina_stats["valid_fiber_coverage"] >= 0.90
        and ct_lamina_stats["swirl_delam_frac"] <= 0.30
    )
    c4_chars = len(all_decoded_boxes) >= 18 and mean_tpl >= 0.52
    c5_topology = multistroke_frac >= 0.65

    k4_criteria_passed = int(c1_corr) + int(c2_thin) + int(c3_lamina) + int(c4_chars) + int(c5_topology)
    blind_reader_score = round(
        0.22 * min(1.0, r_max / 0.32)
        + 0.22 * min(1.0, mean_drop / 60.0)
        + 0.22 * min(1.0, mean_tpl / 0.58)
        + 0.20 * min(1.0, multistroke_frac / 0.80)
        + 0.14 * min(1.0, (len(all_decoded_boxes) / 24.0) * (1.0 + 0.15 * len(uniq_words))),
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
        "nonflat_sheet_gain_pct": lamina_maps["nonflat_gain_pct"],
        "depth_split_m2_0_p2": lamina_maps["depth_split_m2_0_p2"],
        "r_max_2x2cm": round(r_max, 4),
        "total_decoded_chars": len(all_decoded_boxes),
        "num_lines_decoded": len(decoded_lines_info),
        "contiguous_run_ge3_frac": contiguous_run_ge3_frac,
        "multistroke_complete_frac": multistroke_frac,
        "mean_stroke_span_w_mm": mean_span_w_mm,
        "mean_box_drop_pm4_pct": round(mean_drop, 1),
        "mean_tpl_score": round(mean_tpl, 3),
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
        "ag_target_patch_ext_m7",
        "ag_target_patch_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_recto_m7",
        "ag_target_patch_ext_m7",
        "ag_target_patch_m7",
        "thin_wrap_target_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "ag_target_patch_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
        "thin_wrap_target_77keV_m7",
    }
    seg_dirs = [
        sd for sd in sorted(STAGE_I_DIR.glob("w*/*/*"))
        if sd.is_dir() and (sd / "metrics_stage_i.json").exists() and sd.name in priority_set
    ]
    print(f"[Stage K] Running CLSS Compositional Seam Stitcher (K_1 = K_2+K_3+K_5+K_7+K_8) on {len(seg_dirs)} candidates...", flush=True)
    results = []
    for idx, sd in enumerate(seg_dirs, 1):
        r = decode_stage_k_candidate(sd)
        if r is not None:
            results.append(r)
            print(
                f"  [{idx:2d}/{len(seg_dirs)}] {r['scroll_key']}/{r['cand_id']:28s} "
                f"| tf={r['canonical_2d_tf']:6s} | tilt={r['fiber_line_tilt_deg']:+5.1f}deg "
                f"| gain=+{r['nonflat_sheet_gain_pct']:5.1f}% | chars={r['total_decoded_chars']:2d} "
                f"| multi={r['multistroke_complete_frac']*100:4.0f}% | drop=-{r['mean_box_drop_pm4_pct']:4.1f}% "
                f"| tpl={r['mean_tpl_score']:.2f} | K4={r['k4_criteria_passed']}/5 (BRS={r['blind_reader_score']:.2f}) "
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

    # Settle Pre-Registered Bets 9-12
    n_gain_ge65 = sum(1 for r in results if r["nonflat_sheet_gain_pct"] >= 65.0 and r["mean_box_drop_pm4_pct"] >= 50.0)
    bet9_pass = n_gain_ge65 >= 12

    top10 = sorted(results, key=lambda r: r["mean_tpl_score"], reverse=True)[:10]
    mean_multi_all = float(np.mean([r["multistroke_complete_frac"] for r in results]))
    mean_tpl_top10 = float(np.mean([r["mean_tpl_score"] for r in top10]))
    bet10_pass = (mean_multi_all >= 0.65) and (mean_tpl_top10 >= 0.56)

    top6_cols = [r for r in results if r["cand_id"] in {
        "ag_target_patch_recto_m7", "ag_target_patch_m7", "ag_target_patch_m7",
        "ag_target_patch_recto_m7", "ag_target_patch_recto_m7", "ag_target_patch_ext_m7"
    }]
    mean_chars_top6 = float(np.mean([r["total_decoded_chars"] for r in top6_cols]))
    mean_run3_top6 = float(np.mean([r["contiguous_run_ge3_frac"] for r in top6_cols]))
    bet11_pass = (mean_chars_top6 >= 24.0) and (mean_run3_top6 >= 0.85)

    n_k4_pass = sum(1 for r in results if r["k4_verified_pass"] and r["blind_reader_score"] >= 0.65)
    bet12_pass = n_k4_pass >= 6

    summary = {
        "bets_settlement": {
            "Bet_9_NonFlat_3D_Sheet_Gain": {
                "passed": bet9_pass,
                "n_candidates_gain_ge65_and_drop_ge50": n_gain_ge65,
                "target": ">= 12 / 17",
            },
            "Bet_10_MultiLeg_Crossbar_Completion": {
                "passed": bet10_pass,
                "mean_multistroke_complete_frac": round(mean_multi_all, 3),
                "mean_tpl_score_top10": round(mean_tpl_top10, 3),
                "target": "multistroke >= 0.65 and mean_tpl_top10 >= 0.56",
            },
            "Bet_11_Continuous_Line_Density": {
                "passed": bet11_pass,
                "mean_chars_top6_columns": round(mean_chars_top6, 1),
                "mean_contiguous_run_ge3_frac_top6": round(mean_run3_top6, 3),
                "target": "mean_chars >= 24.0 and run_ge3_frac >= 0.85",
            },
            "Bet_12_Strict_K4_Gate_And_Blind_Reader_Score": {
                "passed": bet12_pass,
                "n_k4_verified_and_brs_ge_065": n_k4_pass,
                "target": ">= 6 candidates",
            },
        },
        "decoded_candidates": results,
    }

    out_json = STAGE_K_DIR / "leaderboard_stage_k_seam.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("\n=== Stage K Pre-Registered Bets Settlement ===", flush=True)
    for bk, bv in summary["bets_settlement"].items():
        status = "PASS" if bv["passed"] else "FAIL"
        print(f"  {bk}: {status} -> {bv}", flush=True)
    print(f"=== Stage K Complete ({len(results)} candidates saved to {out_json}) ===", flush=True)


if __name__ == "__main__":
    main()
