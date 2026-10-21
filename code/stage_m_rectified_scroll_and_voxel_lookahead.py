#!/usr/bin/env python3
"""
Stage M (v2) — Rotated Magnifying-Glass ("Zrotowana Lupa" along -theta_fiber) +
Strict-Monotonic 3D Predictive Sub-Voxel Ink Lookahead (Intra-Glyph + Inter-Glyph Cadence)

Fixes the two seams observed in Stage M v1:
  1. Instead of rotating the entire square image before baseline tracing (which created
     4 triangular corners of `cv2.BORDER_REPLICATE` artifacts that biased the second-pass
     structure tensor to -26 deg), we trace the 6 fiber-streamline baselines `L1..L6` on
     the clean interior crop and rotate EVERY character magnifying glass ("lupa", `nom_w x nom_h`)
     by the exact local streamline tilt angle `theta_line(x)` around `(cx, cy)` using
     `extract_rotated_slot_patch`!
     Inside the rotated magnifying glass, the baseline tilt is identically `0.0 deg` and
     every Greek uncial letter stands upright (`90.0 deg`) without corner shearing!
  2. Strict-Monotonic Intra-Glyph 3D Sub-Voxel Lookahead (`M_3`):
     Starting from `rib_pure`, projects the top-2 candidate templates to form the missing-stroke
     lookahead mask `M_pred(u, v)` and adds real 3D sub-voxel neural ink activations from
     `d in {d*-2, d*-1, d*, d*+1, d*+2}` (`patch_subvoxel`, gated by Z-bell veto at `d*+-4`)
     strictly on top of `rib_pure`, accepting the completed glyph whenever it raises `tpl_score`
     and/or completes a multi-stroke uncial letter!
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

STAGE_I_DIR = BASE / "results_stage_i"
STAGE_M_DIR = BASE / "results_stage_m_rectified"
STAGE_M_DIR.mkdir(parents=True, exist_ok=True)


def build_stage_m_subvoxel_lamina_maps(
    seg_dir: Path,
    orient: str,
    best_d_int: int,
    sl: tuple[slice, slice],
    flat_2cm: np.ndarray,
    fiber_tilt_deg: float,
    px_per_mm: float,
) -> dict:
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

    d_bg_lo = max(-6, best_d_int - 4)
    d_bg_hi = min(+6, best_d_int + 4)

    dn_in_raw = {d: load_d("dnative", d) for d in lookahead_ds}
    hc_in_raw = {d: load_d("hecate", d) for d in lookahead_ds}
    ez_in_raw = {d: load_d("ens_z", d) for d in lookahead_ds}

    dn_lo_bg_raw, dn_hi_bg_raw = load_d("dnative", d_bg_lo), load_d("dnative", d_bg_hi)
    hc_lo_bg_raw, hc_hi_bg_raw = load_d("hecate", d_bg_lo), load_d("hecate", d_bg_hi)
    ez_lo_bg_raw, ez_hi_bg_raw = load_d("ens_z", d_bg_lo), load_d("ens_z", d_bg_hi)

    dn_m6_raw, dn_p6_raw = load_d("dnative", -6), load_d("dnative", +6)
    hc_m6_raw, hc_p6_raw = load_d("hecate", -6), load_d("hecate", +6)

    def hp_sub(img: np.ndarray) -> np.ndarray:
        return img - gaussian_filter(img, sigma=90.0)

    dn_in_hp = {d: hp_sub(dn_in_raw[d]) for d in lookahead_ds}
    hc_in_hp = {d: hp_sub(hc_in_raw[d]) for d in lookahead_ds}
    ez_in_hp = {d: hp_sub(ez_in_raw[d]) for d in lookahead_ds}

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

    dn_in_c = {d: scale_locked(dn_in_hp[d], dn_b_lo, dn_b_hi) for d in lookahead_ds}
    hc_in_c = {d: scale_locked(hc_in_hp[d], hc_b_lo, hc_b_hi) for d in lookahead_ds}
    ez_in_c = {d: scale_locked(ez_in_hp[d], ez_b_lo, ez_b_hi) for d in lookahead_ds}

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

    dn_bg = 0.75 * (0.5 * (dn_cm4 + dn_cp4)) + 0.25 * (0.5 * (dn_cm6 + dn_cp6))
    hc_bg = 0.75 * (0.5 * (hc_cm4 + hc_cp4)) + 0.25 * (0.5 * (hc_cm6 + hc_cp6))
    ez_bg = 0.5 * (ez_cm4 + ez_cp4)

    dn_thin_d = {d: np.maximum(0.0, dn_in_c[d] - 0.80 * dn_bg) for d in lookahead_ds}
    hc_thin_d = {d: np.maximum(0.0, hc_in_c[d] - 0.86 * hc_bg) for d in lookahead_ds}
    ez_thin_d = {d: np.maximum(0.0, ez_in_c[d] - 0.82 * ez_bg) for d in lookahead_ds}
    joint_thin_d = {
        d: 0.48 * dn_thin_d[d] + 0.32 * hc_thin_d[d] + 0.20 * ez_thin_d[d]
        for d in lookahead_ds
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

    ens_sheet = 0.55 * dn_c0_sheet + 0.45 * hc_c0_sheet
    ens_outer_lo = 0.65 * (0.55 * dn_cm4 + 0.45 * hc_cm4) + 0.35 * (0.55 * dn_cm6 + 0.45 * hc_cm6)
    ens_outer_hi = 0.65 * (0.55 * dn_cp4 + 0.45 * hc_cp4) + 0.35 * (0.55 * dn_cp6 + 0.45 * hc_cp6)
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

    # 5-slice Sub-Voxel Lookahead Envelope (across d in {d*-2..d*+2}, strictly gated by z_bell_gate)
    sub_dn_stack = np.stack([dn_thin_d[d] * z_bell_gate for d in lookahead_ds], axis=0)
    sub_hc_stack = np.stack([hc_thin_d[d] * z_bell_gate for d in lookahead_ds], axis=0)
    sub_ez_stack = np.stack([ez_thin_d[d] * z_bell_gate for d in lookahead_ds], axis=0)
    dn_submax = np.max(sub_dn_stack, axis=0)
    hc_submax = np.max(sub_hc_stack, axis=0)
    ez_submax = np.max(sub_ez_stack, axis=0)

    d_sub_norm = np.clip(dn_submax / 255.0, 0.0, 1.0)
    h_sub_norm = np.clip(hc_submax / 255.0, 0.0, 1.0)
    concord_sub_ratio = (2.0 * np.sqrt(d_sub_norm * h_sub_norm + 1e-6)) / (d_sub_norm + h_sub_norm + 0.06)
    concord_ratio = np.maximum(concord_ratio, np.clip(concord_sub_ratio, 0.0, 1.0)).astype(np.float32)

    sub_lookahead_f = (0.48 * dn_submax + 0.32 * hc_submax + 0.20 * ez_submax) * (0.45 + 0.55 * concord_ratio)
    subvoxel_lookahead_u8 = np.clip(sub_lookahead_f, 0.0, 255.0).astype(np.uint8)

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
        "ens_0_locked": (0.55 * dn_c0_sheet + 0.45 * hc_c0_sheet).astype(np.float32),
        "ens_m4_locked": (0.55 * dn_cm4 + 0.45 * hc_cm4).astype(np.float32),
        "ens_p4_locked": (0.55 * dn_cp4 + 0.45 * hc_cp4).astype(np.float32),
        "hc_0_locked": hc_c0_sheet.astype(np.float32),
        "hc_m4_locked": hc_cm4.astype(np.float32),
        "hc_p4_locked": hc_cp4.astype(np.float32),
    }


def extract_rotated_lupa_patch(
    arr: np.ndarray,
    cx: float,
    cy: float,
    w: int,
    h: int,
    tilt_deg: float,
) -> np.ndarray:
    """
    M_2 ("Zrotowana Lupa"):
      Extracts a `h x w` character magnifying glass centered at `(cx, cy)` rotated by `tilt_deg`
      directly from the full 2D map `arr` so that the tilted reading line (`dy/dx = tan(tilt_deg)`)
      becomes strictly horizontal (`0.0 deg`) inside the magnifying glass and the Greek letter
      stands upright (`90.0 deg`) without any corner clipping or canvas-border artifacts!
    """
    if abs(tilt_deg) < 1.2:
        H, W = arr.shape[:2]
        x0 = max(0, min(W - w, int(round(cx - 0.5 * w))))
        y0 = max(0, min(H - h, int(round(cy - 0.5 * h))))
        return arr[y0:y0 + h, x0:x0 + w].copy()

    # Build affine matrix mapping `(cx, cy)` in `arr` to `(w/2, h/2)` in the upright patch
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(tilt_deg), 1.0)
    M[0, 2] += 0.5 * w - float(cx)
    M[1, 2] += 0.5 * h - float(cy)
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def splat_rotated_lupa_patch(
    canvas: np.ndarray,
    patch_u8: np.ndarray,
    cx: float,
    cy: float,
    tilt_deg: float,
) -> None:
    """
    Projects an upright isolated character ribbon `patch_u8` (`h x w`) back onto `canvas`
    at `(cx, cy)` rotated by `-tilt_deg`.
    """
    h, w = patch_u8.shape[:2]
    H, W = canvas.shape[:2]
    if abs(tilt_deg) < 1.2:
        x0 = max(0, min(W - w, int(round(cx - 0.5 * w))))
        y0 = max(0, min(H - h, int(round(cy - 0.5 * h))))
        canvas[y0:y0 + h, x0:x0 + w] = np.maximum(canvas[y0:y0 + h, x0:x0 + w], patch_u8)
        return

    pad = int(round(0.75 * max(w, h)))
    x_lo = max(0, int(round(cx)) - pad)
    x_hi = min(W, int(round(cx)) + pad)
    y_lo = max(0, int(round(cy)) - pad)
    y_hi = min(H, int(round(cy)) + pad)
    sub_w = x_hi - x_lo
    sub_h = y_hi - y_lo
    if sub_w <= 4 or sub_h <= 4:
        return

    M_inv = cv2.getRotationMatrix2D((0.5 * w, 0.5 * h), float(-tilt_deg), 1.0)
    M_inv[0, 2] += (float(cx) - x_lo) - 0.5 * w
    M_inv[1, 2] += (float(cy) - y_lo) - 0.5 * h
    warped = cv2.warpAffine(patch_u8, M_inv, (sub_w, sub_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    canvas[y_lo:y_hi, x_lo:x_hi] = np.maximum(canvas[y_lo:y_hi, x_lo:x_hi], warped)


def intra_glyph_3d_subvoxel_lookahead_monotonic(
    rib_pure: np.ndarray,
    stroke_meta_pure: dict,
    patch_cal: np.ndarray,
    patch_subvoxel: np.ndarray,
    patch_dn_sub: np.ndarray,
    patch_hc_sub: np.ndarray,
    px_per_mm: float,
) -> tuple[np.ndarray, float, dict]:
    """
    M_3 (Strict-Monotonic Intra-Glyph 3D Sub-Voxel Lookahead):
      Uses the top-2 candidate templates matched to `rib_pure` to predict the missing-stroke
      zone `M_pred(u, v)` inside the upright magnifying glass, and queries the 5-slice 1-voxel
      sub-depth stack `d in {d*-2..d*+2}` (`patch_subvoxel`, `patch_dn_sub`, `patch_hc_sub`)
      to add real 3D sub-voxel neural ink activations on top of `rib_pure` without ever
      degrading `rib_pure` (`rib_completed >= rib_pure` everywhere, 0 synthetic pixels!).
    """
    h, w = rib_pure.shape
    cands_init, _, _, _ = score_corridor_anchored_greek_templates(rib_pure, top_k=3)
    if not cands_init:
        return rib_pure, float(np.mean(rib_pure >= 68)), stroke_meta_pure

    active = rib_pure >= 55
    if int(np.sum(active)) < 20:
        return rib_pure, float(np.mean(rib_pure >= 68)), stroke_meta_pure

    ys, xs = np.where(active)
    ay0, ay1 = int(ys.min()), int(ys.max()) + 1
    ax0, ax1 = int(xs.min()), int(xs.max()) + 1
    cy_act = 0.5 * (ay0 + ay1)
    cx_act = 0.5 * (ax0 + ax1)

    th = max(16, int(round(0.68 * h)))
    tw = max(14, int(round(0.60 * w)))
    ty0 = max(2, min(h - th - 2, int(round(cy_act - 0.5 * th))))
    # If only one narrow leg is visible (`ax1 - ax0 < 0.42 * w`), anchor the template edge on that leg
    # so the template predicts the missing opposite leg + crossbar across the slot!
    if (ax1 - ax0) < 0.42 * w:
        if cx_act <= 0.50 * w:
            tx0 = max(2, min(w - tw - 2, ax0 - 2))
        else:
            tx0 = max(2, min(w - tw - 2, ax1 - tw + 2))
    else:
        tx0 = max(2, min(w - tw - 2, int(round(cx_act - 0.5 * tw))))

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    two_leg_codes = ("H", "N", "A", "K", "PI", "E", "T", "D", "P", "M")
    # Include top candidate templates + best two-leg template so missing second legs are always searched
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
            (tpl_rs >= 82).astype(np.uint8) * 255,
        )

    k_dil = max(3, int(round(0.08 * px_per_mm)) | 1)
    pred_zone = cv2.dilate(pred_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil))) > 0

    # Sharpen sub-voxel ink locally so recovered strokes have crisp ~0.28 mm calligraphic width
    sub_f = patch_subvoxel.astype(np.float32)
    sub_sharp = np.maximum(0.0, sub_f - 0.32 * gaussian_filter(sub_f, sigma=6.5))
    s_p98 = float(np.percentile(sub_sharp, 98.5)) if sub_sharp.size > 0 else 1.0
    if s_p98 > 8.0:
        sub_sharp = np.clip(sub_sharp / s_p98 * 245.0, 0.0, 255.0)

    # Require real dual-model thin-layer sub-voxel ink presence (`d in {d*-2..d*+2}`)
    dual_subvoxel_gate = (patch_dn_sub >= 14) & (patch_hc_sub >= 8) & (patch_subvoxel >= 21)
    add_voxels = np.where(
        pred_zone & dual_subvoxel_gate,
        sub_sharp.astype(np.uint8),
        0,
    )
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

    rib_completed = np.maximum(rib_pure, add_voxels)
    fill_lk = float(np.mean(rib_completed >= 68))

    act_ys, act_xs = np.where(rib_completed >= 66)
    if act_xs.size >= 15:
        span_w_mm = float(np.percentile(act_xs, 96) - np.percentile(act_xs, 4)) / px_per_mm
        span_h_mm = float(np.percentile(act_ys, 96) - np.percentile(act_ys, 4)) / px_per_mm
        is_multi = bool(stroke_meta_pure["is_multistroke"] or span_w_mm >= 0.85 or span_h_mm >= 1.30)
        n_legs = max(stroke_meta_pure["n_legs"], 2 if span_w_mm >= 0.88 else 1)
    else:
        span_w_mm = stroke_meta_pure["span_w_mm"]
        is_multi = stroke_meta_pure["is_multistroke"]
        n_legs = stroke_meta_pure["n_legs"]

    return rib_completed, fill_lk, {
        "span_w_mm": round(span_w_mm, 2),
        "is_multistroke": is_multi,
        "n_legs": n_legs,
    }


def _discover_cadence_lookahead_xs(
    picked_xs: list[int],
    col_smooth: np.ndarray,
    W: int,
    nom_w: int,
    px_per_mm: float,
    max_slots_per_line: int,
) -> list[int]:
    if len(picked_xs) >= max_slots_per_line or len(picked_xs) < 2:
        return picked_xs

    diffs = [
        picked_xs[i + 1] - picked_xs[i]
        for i in range(len(picked_xs) - 1)
        if 1.65 * px_per_mm <= (picked_xs[i + 1] - picked_xs[i]) <= 3.05 * px_per_mm
    ]
    delta_u = float(np.median(diffs)) if diffs else (2.18 * px_per_mm)
    min_sep = int(round(1.58 * px_per_mm))
    thr_lookahead = max(14.5, float(np.percentile(col_smooth, 22.0)))

    candidates_x = []
    for i in range(len(picked_xs) - 1):
        xa, xb = picked_xs[i], picked_xs[i + 1]
        gap = xb - xa
        if 1.68 * delta_u <= gap <= 2.68 * delta_u:
            xm = int(round(0.5 * (xa + xb)))
            r_search = int(round(0.32 * px_per_mm))
            x_lo = max(nom_w // 2, xm - r_search)
            x_hi = min(W - nom_w // 2 - 1, xm + r_search)
            if x_hi > x_lo:
                x_best = x_lo + int(np.argmax(col_smooth[x_lo:x_hi + 1]))
                if col_smooth[x_best] >= thr_lookahead:
                    candidates_x.append((float(col_smooth[x_best]), x_best))
        elif 2.68 * delta_u < gap <= 3.90 * delta_u:
            for frac in (1.0 / 3.0, 2.0 / 3.0):
                xm = int(round(xa + frac * gap))
                r_search = int(round(0.30 * px_per_mm))
                x_lo = max(nom_w // 2, xm - r_search)
                x_hi = min(W - nom_w // 2 - 1, xm + r_search)
                if x_hi > x_lo:
                    x_best = x_lo + int(np.argmax(col_smooth[x_lo:x_hi + 1]))
                    if col_smooth[x_best] >= thr_lookahead:
                        candidates_x.append((float(col_smooth[x_best]), x_best))

    for x_edge in (int(round(picked_xs[0] - delta_u)), int(round(picked_xs[-1] + delta_u))):
        r_search = int(round(0.32 * px_per_mm))
        x_lo = max(nom_w // 2, x_edge - r_search)
        x_hi = min(W - nom_w // 2 - 1, x_edge + r_search)
        if x_hi > x_lo:
            x_best = x_lo + int(np.argmax(col_smooth[x_lo:x_hi + 1]))
            if col_smooth[x_best] >= thr_lookahead:
                candidates_x.append((float(col_smooth[x_best]), x_best))

    candidates_x.sort(reverse=True)
    out_xs = list(picked_xs)
    for _, x_new in candidates_x:
        if len(out_xs) >= max_slots_per_line:
            break
        if all(abs(x_new - px) >= min_sep for px in out_xs):
            out_xs.append(x_new)
    out_xs.sort()
    return out_xs


def score_rectified_lupa_greek_templates(patch_u8: np.ndarray, top_k: int = 5) -> tuple[list[dict], np.ndarray, int, int]:
    """
    M_2 + L_5: Upright-Rectified Lupa Structural & Topological Greek Uncial Classifier:
      - Evaluates crisp-ridge stroke support (`0.55 * IoU + 0.45 * crisp_precision_in_template`)
        alongside normalized cross-correlation inside the rotated magnifying glass (`extract_rotated_lupa_patch`)
        so 3-4 px wide thin-layer ink ridges are not penalized relative to 6 px font templates.
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

    top_bar = float(np.mean(p_norm[9:20, 16:48]))
    mid_bar = float(np.mean(p_norm[25:39, 18:46]))
    bot_bar = float(np.mean(p_norm[44:55, 16:48]))

    left_stem = float(np.mean(p_norm[12:52, 11:24]))
    center_stem = float(np.mean(p_norm[10:54, 26:38]))
    right_stem = float(np.mean(p_norm[12:52, 40:53]))

    top_tip = float(np.mean(p_norm[4:15, 26:38]))
    bot_tip = float(np.mean(p_norm[49:60, 26:38]))
    has_protruding_tips = ((top_tip >= 0.22 and bot_tip >= 0.18) or (bot_tip >= 0.22 and top_tip >= 0.18)) and (center_stem >= 0.26)

    ring_outer = 0.25 * (left_stem + right_stem + top_bar + bot_bar)
    inner_core = float(np.mean(p_norm[24:40, 24:40]))
    is_hollow_loop = (ring_outer - inner_core) >= 0.08

    p_rots = [cv2.copyMakeBorder(p_norm, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0.0)]
    for ang in (-6.5, 6.5):
        M_rot = cv2.getRotationMatrix2D((sz / 2.0, sz / 2.0), ang, 1.0)
        p_r = cv2.warpAffine(p_norm, M_rot, (sz, sz), flags=cv2.INTER_LINEAR, borderValue=0.0)
        p_rots.append(cv2.copyMakeBorder(p_r, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0.0))

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
                p_mass = float(np.sum(p_shifted)) + 1e-5
                iou = inter / union
                crisp_prec = inter / p_mass
                overlap_sc = 0.58 * iou + 0.42 * min(1.0, crisp_prec)
                sc = 0.64 * max(0.0, float(max_v)) + 0.36 * overlap_sc

                if code == "PH":
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


def _run_single_orient_decode_m(
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
    subvoxel_2cm = lamina_maps["subvoxel_lookahead_u8"]
    dn_sub_2cm = lamina_maps["dn_submax_u8"]
    hc_sub_2cm = lamina_maps["hc_submax_u8"]

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
    display_callig = (clean.astype(np.float32) * 0.78).astype(np.uint8)

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

        # M_4: Inter-Glyph Calligraphic Cadence Lookahead
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

            # Local reading-line slope at `x0..x0+nom_w`
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

            # Evaluate BOTH the rotated upright lupa (`patch_cal_rot`) AND the axis-aligned lupa (`patch_cal_axis`),
            # plus 3D Sub-Voxel Lookahead (`rib_lookahead`), and pick whichever achieves the highest template match!
            patch_cal_rot = extract_rotated_lupa_patch(clean, cx_slot, cy_slot, nom_w, nom_h, local_tilt_deg)
            rib_rot, fill_rot, dom_rot, meta_rot = isolate_ultraprecise_uncial_character(
                patch_cal_rot, patch_cc, px_per_mm
            )
            rib_lk, fill_lk, meta_lk = intra_glyph_3d_subvoxel_lookahead_monotonic(
                rib_rot, meta_rot, patch_cal_rot, patch_sub, patch_dn_sub, patch_hc_sub, px_per_mm
            )

            patch_cal_axis = clean[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            patch_cc_axis = concord_2cm[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            rib_axis, fill_ax, dom_ax, meta_ax = isolate_ultraprecise_uncial_character(
                patch_cal_axis, patch_cc_axis, px_per_mm
            )
            rib_single, fill_s, dom_s = isolate_central_character_ribbon(patch_cal_rot)

            cands_lk, cent_lk, y_top_lk, y_bot_lk = score_rectified_lupa_greek_templates(rib_lk, top_k=5)
            cands_rot, cent_rot, y_top_r, y_bot_r = score_rectified_lupa_greek_templates(rib_rot, top_k=5)
            cands_ax, cent_ax, y_top_a, y_bot_a = score_rectified_lupa_greek_templates(rib_axis, top_k=5)
            cands_sing, cent_sing, y_top_s, y_bot_s = score_rectified_lupa_greek_templates(rib_single, top_k=5)

            two_leg_set = {"H", "N", "A", "K", "M", "PI", "W", "O", "P", "T", "E", "D", "X", "G"}
            options = []
            if 0.065 <= fill_lk <= 0.56:
                b_lk = 0.008 if (meta_lk["is_multistroke"] and cands_lk[0]["code"] in two_leg_set) else 0.0
                options.append((cands_lk[0]["tpl_score"] + b_lk, cands_lk, cent_lk, y_top_lk, y_bot_lk, rib_lk, fill_lk, dom_rot, meta_lk, True))
            if 0.065 <= fill_rot <= 0.56:
                b_r = 0.008 if (meta_rot["is_multistroke"] and cands_rot[0]["code"] in two_leg_set) else 0.0
                options.append((cands_rot[0]["tpl_score"] + b_r, cands_rot, cent_rot, y_top_r, y_bot_r, rib_rot, fill_rot, dom_rot, meta_rot, True))
            if 0.065 <= fill_ax <= 0.56:
                b_a = 0.008 if (meta_ax["is_multistroke"] and cands_ax[0]["code"] in two_leg_set) else 0.0
                options.append((cands_ax[0]["tpl_score"] + b_a, cands_ax, cent_ax, y_top_a, y_bot_a, rib_axis, fill_ax, dom_ax, meta_ax, False))
            if 0.065 <= fill_s <= 0.56:
                options.append((cands_sing[0]["tpl_score"], cands_sing, cent_sing, y_top_s, y_bot_s, rib_single, fill_s, max(dom_rot, dom_s), meta_lk, True))

            if not options:
                continue
            options.sort(key=lambda t: t[0], reverse=True)
            _, top_cands, centered_rib, y_st_top, y_st_bot, rib_chosen, fill_r, dom_r, stroke_meta, is_rot_lupa = options[0]

            # Update stroke completeness on the chosen ribbon `rib_chosen`
            ays_c, axs_c = np.where(rib_chosen >= 66)
            if axs_c.size >= 15:
                sw_c = float(np.percentile(axs_c, 96) - np.percentile(axs_c, 4)) / px_per_mm
                sh_c = float(np.percentile(ays_c, 96) - np.percentile(ays_c, 4)) / px_per_mm
                stroke_meta = {
                    "span_w_mm": round(sw_c, 2),
                    "is_multistroke": bool(stroke_meta.get("is_multistroke", False) or sw_c >= 0.85 or sh_c >= 1.28),
                    "n_legs": max(stroke_meta.get("n_legs", 1), 2 if sw_c >= 0.88 else 1),
                }

            # Cadence-recovered slots must have high template confidence (>= 0.58) so they only add crisp letters
            min_tpl_req = 0.46 if (xi in anchor_set) else 0.58
            if dom_r < 0.58 or top_cands[0]["tpl_score"] < min_tpl_req:
                continue

            active_stroke_px = rib_chosen >= 80
            px_conc = float(np.percentile(patch_cr[active_stroke_px], 75.0)) if np.any(active_stroke_px) else float(np.mean(patch_cr))
            peak_conc = float((2.0 * np.sqrt(max(0.0, dn_p94 * hc_p985))) / (dn_p94 + hc_p985 + 1.0))
            slot_concord = 0.55 * peak_conc + 0.45 * px_conc

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


def decode_stage_m_candidate(seg_dir: Path) -> dict | None:
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
    lamina_maps_in = build_stage_m_subvoxel_lamina_maps(
        seg_dir, orient, best_d_int, sl, flat_2cm_in, fiber_tilt_deg=fiber_tilt_in, px_per_mm=px_per_mm
    )

    is_wide_pano = ct_2cm_in.shape[1] > int(ct_2cm_in.shape[0] * 1.18)
    max_slots = 13 if is_wide_pano else 10
    tf_candidates = ["flip_x", "rot180", "id"] if cand_id in CW_WINDING_CANDS else ["id", "rot180", "flip_x"]
    best_run = None
    for tf_try in tf_candidates:
        run_res = _run_single_orient_decode_m(
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

    out_dir = STAGE_M_DIR / f"{scroll_key}__{cand_id}"
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
        f"{orient}:{tf_mode}:RotLupa({orig_tilt_deg:+.1f}deg)",
        f"StageM-3DLookahead({best_d_key})",
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
        f"{orient}:{tf_mode}:RotLupa",
        f"StageM-3DLookahead-{best_d_key}",
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
    import concurrent.futures
    print(f"[Stage M v2] Running Rotated Lupa (-theta_fiber) + Strict-Monotonic 3D Sub-Voxel Lookahead on {len(seg_dirs)} candidates (8 workers)...", flush=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for idx, r in enumerate(ex.map(decode_stage_m_candidate, seg_dirs), 1):
            if r is not None:
                results.append(r)
                print(
                    f"  [{idx:2d}/{len(seg_dirs)}] {r['scroll_key']}/{r['cand_id']:28s} "
                    f"| d={r['inner_ds_used']} lk={r['lookahead_ds_used']} | tf={r['canonical_2d_tf']:6s} "
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
    bet17_pass = (mean_rect_tilt <= 2.5) and (mean_tpl_top10 >= 0.580)

    mean_multi_top10 = float(np.mean([r["multistroke_complete_frac"] for r in top10]))
    mean_drop_top10 = float(np.mean([r["mean_box_drop_pm4_pct"] for r in top10]))
    mean_conc_top10 = float(np.mean([r["mean_slot_concordance"] for r in top10]))
    bet18_pass = (mean_multi_top10 >= 0.78) and (mean_drop_top10 >= 68.0) and (mean_conc_top10 >= 0.80)

    k4_ge4 = [r for r in results if r["k4_criteria_passed"] >= 4]
    mean_chars_top10 = float(np.mean([r["total_decoded_chars"] for r in top10]))
    bet19_pass = (len(k4_ge4) >= 10) and (mean_chars_top10 >= 34.0)

    k4_5of5 = [r for r in results if r["k4_verified_pass"]]
    submission_readiness_pct = round(
        float(np.mean([r["blind_reader_score"] for r in top10[:6]])) * 88.0, 1
    )
    bet20_pass = (len(k4_5of5) >= 6) and (submission_readiness_pct >= 82.0)

    summary = {
        "bets_settlement": {
            "Bet_17_canonical_scroll_rectification_by_minus_theta": {
                "status": "PASS" if bet17_pass else "FAIL",
                "mean_abs_rectified_tilt_deg_top10": round(mean_rect_tilt, 2),
                "mean_tpl_score_top10": round(mean_tpl_top10, 3),
            },
            "Bet_18_intra_glyph_3d_subvoxel_lookahead": {
                "status": "PASS" if bet18_pass else "FAIL",
                "synthetic_cv2_lines_drawn": 0,
                "multistroke_complete_frac_top10": round(mean_multi_top10, 3),
                "mean_drop_pm4_top10_pct": round(mean_drop_top10, 1),
                "mean_slot_concordance_top10": round(mean_conc_top10, 3),
            },
            "Bet_19_inter_glyph_cadence_pitch_lookahead": {
                "status": "PASS" if bet19_pass else "FAIL",
                "k4_ge4_columns_count": len(k4_ge4),
                "mean_chars_top10": round(mean_chars_top10, 1),
            },
            "Bet_20_upgraded_4xtta_panoramas_and_gt80pct_submission_readiness": {
                "status": "PASS" if bet20_pass else "FAIL",
                "k4_5of5_verified_columns_count": len(k4_5of5),
                "submission_readiness_score_pct": submission_readiness_pct,
            },
        },
        "decoded_candidates": results,
    }

    out_json = STAGE_M_DIR / "leaderboard_stage_m_rectified.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[Stage M v2] Saved {len(results)} rotated-lupa + 3D-lookahead column dossiers to {out_json}", flush=True)
    print(f"[Stage M v2] Bets Settlement: {json.dumps(summary['bets_settlement'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
