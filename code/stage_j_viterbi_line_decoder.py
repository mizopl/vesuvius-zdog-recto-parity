#!/usr/bin/env python3
"""
Stage J (v3) — Nieuwlaar-Style 45-Degree Chamfered Reading Baseline Tracker,
Hysteresis Baseline-Corridor Ink Calibration, Stroke-Centered Uncial Classifier,
and Viterbi Continuous Greek Line & Word Decoder.

Key improvements in v3 (from visual inspection of v2 vs `pherc0343-nieuwlaar.webp`):
  1. Eliminates the lower-half `W` (omega) template bias: centers each isolated character
     ribbon on its active ink bounding box (preserving aspect ratio) before matching against
     `TEMPLATES_64`, and aligns the true lower stroke edge (`y_stroke_bottom`) with the
     cyan reading baseline!
  2. True Nieuwlaar 45-Degree Chamfered Cyan Baseline (`_____/‾‾‾‾‾`):
     Instead of vertical 90-degree histogram steps across blank gaps, builds clean horizontal
     segments under each character/word group connected by smooth 45-degree diagonal chamfer ramps,
     drawn ONLY under contiguous letter runs (`>= 2` adjacent characters within `4.6 mm`).
  3. Hysteresis Baseline-Corridor Ink Reconstruction:
     Seeds thin-lamina (`d* +- 4 vox` gated) ink inside the reading corridor above the baseline
     and reconstructs connected stroke bodies (crossbars, arches, diagonals) while suppressing
     interlinear noise.
"""
import json
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
    build_calligraphic_thin_lamina_maps,
    isolate_central_character_ribbon,
    filter_boxes_by_frozen_stroke_z_attenuation,
    render_stage_i_paleography_dossier,
)

STAGE_I_DIR = BASE / "results_stage_i"
STAGE_J_DIR = BASE / "results_stage_j_lines"
STAGE_J_DIR.mkdir(parents=True, exist_ok=True)

GREEK_VOWELS = {"A", "E", "H", "I", "O", "Y", "W"}

EPICUREAN_LEXICON = [
    "KAI", "MEN", "GAR", "OYK", "OYD", "OYTE", "ALL", "ALLA", "APA", "ARA",
    "TON", "THN", "TOY", "TWN", "TOIC", "TAIC", "TI", "TIN", "TINA",
    "DIA", "KATA", "META", "PARA", "PERI", "PRO", "PROC", "CYN", "YPO", "EPI", "APO", "EIC",
    "AYT", "AYTO", "AYTON", "AYTHN", "AYTWN", "OYTOC", "EKEI", "ALLWN",
    "LOG", "LOGO", "LOGON", "LOGOY", "LOGWN",
    "PHYC", "PHYCI", "PHYCIN", "PHYCEW",
    "THE", "THEO", "THEON", "THEWN", "THEOIC",
    "HDON", "HDONH", "HDONHN", "HDONHC",
    "ALHTH", "ALHTHE", "DOX", "DOXA", "DOXAN",
    "POIH", "POIHT", "MOYC", "MOYCIK", "PHT", "PHTOP",
    "KAK", "KAKIA", "APET", "APETH", "CWPH", "COPH", "COPHO",
    "ATOM", "KEN", "KENON", "KOCM", "NOY", "NOYC",
    "AICTH", "PATH", "PATHO", "KINH", "KINHC", "XPON", "TOP",
    "EINAI", "ECTI", "ECTIN", "ECHEI", "LEGEI", "PHHCI", "GINET",
]

BIGRAM_BONUS: dict[tuple[str, str], float] = {}
TRIGRAM_BONUS: dict[tuple[str, str, str], float] = {}
for word in EPICUREAN_LEXICON:
    for i in range(len(word) - 1):
        bg = (word[i], word[i + 1])
        BIGRAM_BONUS[bg] = BIGRAM_BONUS.get(bg, 0.0) + 0.14
    for i in range(len(word) - 2):
        tg = (word[i], word[i + 1], word[i + 2])
        TRIGRAM_BONUS[tg] = TRIGRAM_BONUS.get(tg, 0.0) + 0.28

LEGAL_CLUSTERS = {
    ("C", "T"), ("N", "T"), ("M", "P"), ("N", "D"), ("N", "G"),
    ("P", "R"), ("T", "R"), ("K", "R"), ("G", "R"), ("D", "R"),
    ("P", "L"), ("K", "L"), ("G", "L"), ("M", "N"), ("K", "T"),
    ("P", "T"), ("X", "R"), ("X", "TH"), ("P", "H"), ("T", "H"),
}
DIPHTHONGS = {
    ("A", "I"), ("E", "I"), ("O", "I"), ("Y", "I"),
    ("A", "Y"), ("E", "Y"), ("O", "Y"), ("H", "Y"),
}


def center_ribbon_on_active_ink(patch_u8: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    """
    Finds the tight bounding box `[y_min:y_max, x_min:x_max]` of the active stroke (`>= 68`)
    inside `patch_u8`, returns `(centered_patch_u8, y_min, y_max, stroke_aspect_w_over_h)` so
    off-center strokes never trigger false bottom-heavy (`W`) template matches, and so the true
    bottom of the stroke (`y_max`) is known for anchoring the Nieuwlaar cyan baseline!
    """
    h, w = patch_u8.shape
    ys, xs = np.where(patch_u8 >= 68)
    if ys.size < 16:
        return patch_u8, 0, h, 1.0

    y0 = int(np.percentile(ys, 2.0))
    y1 = int(np.percentile(ys, 98.0)) + 1
    x0 = int(np.percentile(xs, 2.0))
    x1 = int(np.percentile(xs, 98.0)) + 1
    sh = max(12, y1 - y0)
    sw = max(12, x1 - x0)
    aspect_wh = float(sw) / float(sh)

    # Place [y0:y1, x0:x1] centered inside a square-ish canvas with 18% margin (matching TEMPLATES_64 [0.16..0.84])
    sub = patch_u8[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    target_h = int(round(0.66 * h))
    # Match TEMPLATES_64 horizontal span [0.20..0.80] (width = 0.60 * w for two-leg letters, 0.26 * w for iota)
    scale_y = target_h / max(1, sub.shape[0])
    target_w = int(np.clip(round(sub.shape[1] * scale_y * 0.84), int(0.26 * w), int(0.62 * w)))
    sub_rsz = cv2.resize(sub, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    centered = np.zeros_like(patch_u8)
    cy0 = (h - target_h) // 2
    cx0 = (w - target_w) // 2
    centered[cy0:cy0 + target_h, cx0:cx0 + target_w] = sub_rsz
    return centered, int(y0), int(y1), aspect_wh


def score_top_k_greek_templates(patch_u8: np.ndarray, top_k: int = 6) -> tuple[list[dict], np.ndarray, int, int]:
    """
    Centers `patch_u8` on its active stroke bounding box and evaluates all 20 Greek Uncial templates
    (`TEMPLATES_64`) with aspect-ratio & topological priors to eliminate `W` (omega) over-prediction.
    """
    centered_u8, y_stroke_top, y_stroke_bot, aspect_wh = center_ribbon_on_active_ink(patch_u8)

    sz = 64
    p_rsz = cv2.resize(centered_u8, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32)
    r2 = ((yy - 31.5) / 28.0) ** 2 + ((xx - 31.5) / 28.0) ** 2
    taper = np.clip(1.15 - 0.35 * r2, 0.25, 1.0)
    lo = float(np.percentile(p_rsz, 35.0))
    hi = float(np.percentile(p_rsz, 98.0))
    p_norm = np.clip((p_rsz - lo) / max(1.0, hi - lo), 0.0, 1.0) * taper
    p_rots = [cv2.copyMakeBorder(p_norm, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0.0)]
    for ang in (-7.0, 7.0):
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
                iou = inter / union
                sc = 0.65 * max(0.0, float(max_v)) + 0.35 * iou

                if code == "I" and aspect_wh > 0.62:
                    sc *= 0.82
                elif code == "W":
                    if aspect_wh < 0.72:
                        sc *= 0.76
                    else:
                        sc *= 0.92
                elif code in ("M", "PI", "H", "N") and aspect_wh < 0.45:
                    sc *= 0.82

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
    return candidates[:top_k], centered_u8, y_stroke_top, y_stroke_bot


def transition_score(c_prev2: str | None, c_prev1: str, c_curr: str) -> float:
    sc = 0.0
    if c_prev1 == c_curr:
        if c_curr in ("L", "M", "N", "C", "T", "P", "K"):
            sc -= 0.12
        else:
            sc -= 0.50

    v1 = c_prev1 in GREEK_VOWELS
    v2 = c_curr in GREEK_VOWELS
    if v1 != v2:
        sc += 0.15
    elif v1 and v2:
        if (c_prev1, c_curr) in DIPHTHONGS:
            sc += 0.20
        else:
            sc -= 0.08
    else:
        if (c_prev1, c_curr) in LEGAL_CLUSTERS:
            sc += 0.14
        else:
            sc -= 0.18

    sc += min(0.34, BIGRAM_BONUS.get((c_prev1, c_curr), 0.0))

    if c_prev2 is not None:
        v0 = c_prev2 in GREEK_VOWELS
        if (not v0) and (not v1) and (not v2):
            sc -= 0.42
        if v0 and v1 and v2:
            sc -= 0.32
        sc += min(0.48, TRIGRAM_BONUS.get((c_prev2, c_prev1, c_curr), 0.0))

    return sc


def viterbi_decode_line_slots(slots: list[dict]) -> tuple[list[dict], dict]:
    M = len(slots)
    if M == 0:
        return [], {"ascii": "", "unicode": "", "greedy_ascii": "", "matched_words": [], "viterbi_gain": 0.0}
    if M == 1:
        best_c = slots[0]["candidates"][0]
        s0 = dict(slots[0])
        s0.pop("candidates", None)
        s0["greek_ascii"] = best_c["code"]
        s0["greek_char"] = best_c["uni"]
        s0["greedy_ascii"] = best_c["code"]
        s0["tpl_score"] = round(best_c["tpl_score"], 3)
        s0["_tpl_u8"] = best_c["_tpl_u8"]
        return [s0], {
            "ascii": best_c["code"],
            "unicode": best_c["uni"].split()[0],
            "greedy_ascii": best_c["code"],
            "matched_words": [],
            "viterbi_gain": 0.0,
        }

    dp: list[dict[tuple[int, int], tuple[float, int]]] = [{} for _ in range(M)]
    for i0, c0 in enumerate(slots[0]["candidates"]):
        vis0 = 1.60 * c0["tpl_score"]
        for i1, c1 in enumerate(slots[1]["candidates"]):
            vis1 = 1.60 * c1["tpl_score"]
            tr = transition_score(None, c0["code"], c1["code"])
            dp[1][(i0, i1)] = (vis0 + vis1 + 0.24 * tr, -1)

    for j in range(2, M):
        for i_curr, c_curr in enumerate(slots[j]["candidates"]):
            vis_curr = 1.60 * c_curr["tpl_score"]
            for i_prev1, c_prev1 in enumerate(slots[j - 1]["candidates"]):
                best_val = -1e9
                best_i_prev2 = 0
                for i_prev2, c_prev2 in enumerate(slots[j - 2]["candidates"]):
                    prev_state = dp[j - 1].get((i_prev2, i_prev1))
                    if prev_state is None:
                        continue
                    tr = transition_score(c_prev2["code"], c_prev1["code"], c_curr["code"])
                    val = prev_state[0] + vis_curr + 0.24 * tr
                    if val > best_val:
                        best_val = val
                        best_i_prev2 = i_prev2
                dp[j][(i_prev1, i_curr)] = (best_val, best_i_prev2)

    best_final_pair = max(dp[M - 1].items(), key=lambda kv: kv[1][0])[0]
    best_total_score = dp[M - 1][best_final_pair][0]

    chosen_indices = [0] * M
    chosen_indices[M - 2], chosen_indices[M - 1] = best_final_pair
    for j in range(M - 1, 1, -1):
        i_prev2 = dp[j][(chosen_indices[j - 1], chosen_indices[j])][1]
        chosen_indices[j - 2] = i_prev2

    greedy_score = sum(1.60 * s["candidates"][0]["tpl_score"] for s in slots)
    decoded_slots = []
    for j, s in enumerate(slots):
        c_pick = s["candidates"][chosen_indices[j]]
        c_greedy = s["candidates"][0]
        s_out = dict(s)
        s_out.pop("candidates", None)
        s_out["greek_ascii"] = c_pick["code"]
        s_out["greek_char"] = c_pick["uni"]
        s_out["greedy_ascii"] = c_greedy["code"]
        s_out["tpl_score"] = round(c_pick["tpl_score"], 3)
        s_out["_tpl_u8"] = c_pick["_tpl_u8"]
        decoded_slots.append(s_out)

    ascii_seq = "".join(s["greek_ascii"] for s in decoded_slots)
    uni_seq = "".join(s["greek_char"].split()[0] for s in decoded_slots)
    matched_words = [w for w in EPICUREAN_LEXICON if len(w) >= 3 and w in ascii_seq]

    return decoded_slots, {
        "ascii": ascii_seq,
        "unicode": uni_seq,
        "greedy_ascii": "".join(s["greedy_ascii"] for s in decoded_slots),
        "matched_words": matched_words,
        "viterbi_gain": round(float(best_total_score - greedy_score), 3),
    }


def find_largest_valid_interior_crop(ct_mid: np.ndarray, px_per_mm: float) -> tuple[slice, slice, float]:
    H, W = ct_mid.shape
    for target_mm in (20.0, 16.5, 14.0, 12.0):
        win = min(H, W, int(round(target_mm * px_per_mm)))
        best_cov = -1.0
        best_yx = ((H - win) // 2, (W - win) // 2)
        step = max(16, (H - win) // 8) if H > win else 16
        ys = range(0, max(1, H - win + 1), step)
        xs = range(0, max(1, W - win + 1), step)
        for y0 in ys:
            for x0 in xs:
                sub = ct_mid[y0:y0 + win, x0:x0 + win]
                cov = float(np.mean((sub > 15) & (sub < 248)))
                if cov > best_cov:
                    best_cov = cov
                    best_yx = (y0, x0)
        if best_cov >= 0.91 or target_mm == 12.0:
            y0, x0 = best_yx
            return slice(y0, y0 + win), slice(x0, x0 + win), round(win / px_per_mm, 1)
    win = min(H, W)
    return slice(0, win), slice(0, win), round(win / px_per_mm, 1)


def build_chamfered_45deg_polyline(x0: int, x1: int, knot_xs: list[int], knot_ys: list[int], H: int) -> np.ndarray:
    """
    Constructs a Marcel Nieuwlaar-style 45-degree chamfered reading baseline polyline
    (`_____/‾‾‾‾‾\\_____`) from `[x0, x1]` through `(knot_xs, knot_ys)`:
      - Between consecutive knots `(xk0, yk0)` and `(xk1, yk1)`, stays flat at `yk0` up to the
        midpoint, transitions along a smooth 45-degree ramp (`|dx| = |dy|`), and stays flat at `yk1`.
    """
    if not knot_xs:
        return np.zeros((0, 1, 2), dtype=np.int32)

    xs = np.arange(x0, x1 + 1, dtype=np.int32)
    ys = np.full_like(xs, knot_ys[0], dtype=np.float32)

    for i in range(len(knot_xs) - 1):
        xa, ya = knot_xs[i], float(knot_ys[i])
        xb, yb = knot_xs[i + 1], float(knot_ys[i + 1])
        dy = yb - ya
        ramp_w = int(min( max(1, xb - xa), max(8, abs(dy) * 1.15) ))
        xm = (xa + xb) // 2
        rx0 = max(x0, xm - ramp_w // 2)
        rx1 = min(x1, rx0 + ramp_w)
        # Flat at ya before rx0, 45-deg ramp in [rx0, rx1], flat at yb after rx1
        m_before = (xs >= xa) & (xs <= rx0)
        ys[m_before] = ya
        m_ramp = (xs > rx0) & (xs <= rx1)
        if np.any(m_ramp) and rx1 > rx0:
            ys[m_ramp] = ya + dy * (xs[m_ramp] - rx0) / float(rx1 - rx0)
        m_after = (xs > rx1) & (xs <= xb)
        ys[m_after] = yb

    if knot_xs[-1] < x1:
        ys[xs >= knot_xs[-1]] = float(knot_ys[-1])

    ys_i = np.clip(np.round(ys), 4, H - 5).astype(np.int32)
    return np.stack([xs, ys_i], axis=1).reshape(-1, 1, 2)


def trace_staircase_reading_baselines(
    ct_2cm: np.ndarray,
    clean_ink: np.ndarray,
    flat_2cm: np.ndarray,
    px_per_mm: float,
) -> tuple[float, list[dict]]:
    """
    Traces reading baselines `y_base(x)` directly along the physical papyrus vascular fiber threads
    and inter-bundle grooves (`linie rozstępów / nici / wrzeciona papirusu`) in `ct_2cm`:
      1. Bandpass-filters `ct_2cm` at the vascular fiber-bundle scale (`0.12-0.35 mm`, DoG 3..18 px).
      2. Computes the true fiber-thread tangent angle `theta_tangent = theta_grad + pi/2 (mod pi)`
         across a 4x4 grid to capture both global fiber tilt (up to +-28 deg) and spindle curvature.
      3. Locks the Radon line-spacing search to a tight +-4.5 deg corridor around the true CT
         fiber-thread slope, and snaps each baseline bin onto the joint CT fiber-groove + ink peak.
    """
    H, W = clean_ink.shape
    nom_h = int(round(2.48 * px_per_mm))
    half_h = nom_h // 2

    ct_f = ct_2cm.astype(np.float32)
    ct_bp = cv2.GaussianBlur(ct_f, (0, 0), 3.0) - cv2.GaussianBlur(ct_f, (0, 0), 18.0)
    gx = cv2.Sobel(ct_bp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ct_bp, cv2.CV_32F, 0, 1, ksize=3)

    # Compute local fiber-thread tangent slope across 4x4 sub-windows + column-wise spindle profile
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

    # Horizontal CT fiber-groove / rib striation energy map (for snapping baselines onto fiber grooves)
    groove_energy = gaussian_filter(np.abs(gy), sigma=(2.5, 18.0))
    g_p95 = max(1e-4, float(np.percentile(groove_energy, 95.0)))
    groove_norm = np.clip(groove_energy / g_p95 * 45.0, 0.0, 45.0)

    ink_ds = cv2.resize(clean_ink, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    h_ds, w_ds = ink_ds.shape
    xs_ds = np.arange(w_ds, dtype=np.float32) - 0.5 * w_ds

    best_slope = fiber_slope
    best_radon_sc = -1.0
    best_prof_full = None

    # Search tightly around the true CT papyrus fiber-thread slope (+-0.09 ~ +-5 deg)
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
        k_s = max(5, int(round(1.15 * px_per_mm / 4.0)) | 1)
        row_sm = cv2.GaussianBlur(row_p86.reshape(-1, 1), (1, k_s), 0).ravel()
        mod_sc = float(np.percentile(row_sm, 92.0) - np.percentile(row_sm, 35.0))
        prior_w = 1.0 - 1.80 * abs(slope - fiber_slope)
        sc = mod_sc * max(0.35, prior_w)
        if sc > best_radon_sc:
            best_radon_sc = sc
            best_slope = float(slope)
            best_prof_full = cv2.resize(row_sm.reshape(-1, 1), (1, H), interpolation=cv2.INTER_LINEAR).ravel()

    min_line_dist = int(round(3.15 * px_per_mm))
    margin_y = int(round(1.40 * px_per_mm))
    center_ys = []
    for y_idx in np.argsort(best_prof_full)[::-1]:
        yi = int(y_idx)
        if yi < margin_y or yi > H - margin_y:
            continue
        if all(abs(yi - ly) >= min_line_dist for ly in center_ys):
            center_ys.append(yi)
            if len(center_ys) >= 5:
                break
    center_ys.sort()

    # Build smooth spindle streamline offset `dy_streamline(x)` from column-wise CT fiber slopes
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
            search_r = int(round(0.42 * px_per_mm))
            ys0 = max(half_h, int(round(y_pred)) - search_r)
            ys1 = min(H - half_h, int(round(y_pred)) + search_r)
            best_yb = float(np.clip(y_pred, half_h, H - half_h))
            best_eb = -1.0
            for y_try in range(ys0, ys1 + 1, 5):
                e_ink = float(np.percentile(clean_ink[y_try - half_h:y_try + half_h, x0_b:x1_b], 88.0))
                # Snap baseline (`y_try + 0.44 * nom_h`) onto actual CT papyrus fiber groove (`groove_norm`)
                y_groov = min(H - 1, int(round(y_try + 0.44 * nom_h)))
                e_groov = float(np.mean(groove_norm[max(0, y_groov - 4):min(H, y_groov + 5), x0_b:x1_b]))
                e_reg = e_ink + 0.45 * e_groov - 0.32 * abs(y_try - y_pred)
                if e_reg > best_eb:
                    best_eb = e_reg
                    best_yb = float(y_try)
            bin_y_centers[b_i] = best_yb

        bin_y_smooth = gaussian_filter(bin_y_centers, sigma=1.4)
        y_center_curve = np.interp(
            np.arange(W, dtype=np.float32),
            (np.arange(n_bins, dtype=np.float32) + 0.5) * bin_w,
            bin_y_smooth,
        )
        y_base_curve = np.clip(np.round(y_center_curve + 0.44 * nom_h), nom_h, H - 6).astype(np.int32)
        y_center_i = np.clip(y_base_curve - int(round(0.44 * nom_h)), half_h, H - half_h)

        lines_out.append({
            "line_id": idx_l + 1,
            "line_name": f"L{idx_l + 1}",
            "yc_mid": int(yc_mid),
            "y_center_x": y_center_i,
            "y_base_x": y_base_curve,
        })

    return round(float(np.rad2deg(np.arctan(best_slope))), 2), lines_out


def calibrate_ink_with_reading_baselines(
    callig_2cm: np.ndarray,
    dn_thin_2cm: np.ndarray,
    hc_thin_2cm: np.ndarray,
    flat_2cm: np.ndarray,
    lines_out: list[dict],
    px_per_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = callig_2cm.shape
    nom_h = int(round(2.48 * px_per_mm))
    corridor_w = np.full((H, W), 0.38, dtype=np.float32)

    yy = np.arange(H, dtype=np.float32)[:, None]
    for l_item in lines_out:
        yb = l_item["y_base_x"][None, :].astype(np.float32)
        yt = yb - nom_h
        in_band = np.exp(-0.5 * (np.maximum(0.0, np.maximum(yt - yy, yy - yb)) / (0.24 * px_per_mm)) ** 2)
        corridor_w = np.maximum(corridor_w, 0.38 + 0.90 * in_band)

    dual_body = 0.60 * dn_thin_2cm.astype(np.float32) + 0.40 * hc_thin_2cm.astype(np.float32)
    dual_smooth = gaussian_filter(dual_body, sigma=2.2)
    calibrated = (0.60 * callig_2cm.astype(np.float32) + 0.40 * dual_smooth) * corridor_w * np.clip(flat_2cm, 0.0, 1.0)

    valid = flat_2cm > 0.40
    p994 = float(np.percentile(calibrated[valid], 99.4)) if valid.any() else 255.0
    if p994 > 5.0:
        calibrated = np.clip(calibrated / p994 * 255.0, 0.0, 255.0)
    return calibrated.astype(np.uint8), corridor_w


def make_nieuwlaar_gold_overlay(ct_2cm: np.ndarray, ink_u8: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ct_f = ct_2cm.astype(np.float32)
    vals = ct_f[valid]
    if vals.size > 100:
        lo, hi = np.percentile(vals, [2.0, 98.5])
        ct_norm = np.clip((ct_f - lo) / max(1.0, hi - lo), 0.0, 1.0)
    else:
        ct_norm = ct_f / 255.0
    base = (ct_norm * 155.0).astype(np.uint8)
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    ink_f = gaussian_filter(ink_u8.astype(np.float32), sigma=1.8)
    ivals = ink_f[valid]
    if ivals.size > 100:
        p70 = float(np.percentile(ivals, 70.0))
        p996 = float(np.percentile(ivals, 99.6))
    else:
        p70, p996 = 58.0, 255.0
    alpha = np.clip((ink_f - p70) / max(1.0, p996 - p70), 0.0, 1.0) ** 1.08
    alpha[~valid] = 0.0

    gold = np.zeros_like(bgr, dtype=np.float32)
    gold[:, :, 0] = 78.0
    gold[:, :, 1] = 228.0
    gold[:, :, 2] = 255.0

    out = bgr.astype(np.float32) * (1.0 - 0.86 * alpha[:, :, None]) + gold * (0.86 * alpha[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_nieuwlaar_cyan_reading_underline(
    img_bgr: np.ndarray,
    y_base_x: np.ndarray,
    slots: list[dict],
    px_per_mm: float,
    draw_labels: bool = False,
    line_name: str = "",
):
    """
    Draws the authentic Marcel Nieuwlaar-style 45-degree chamfered Cyan Reading Underline
    (`_____/‾‾‾‾‾`) directly beneath contiguous letter groups (`gap <= 3.6 mm`), anchored at
    each character's actual lower stroke edge (`y_stroke_bot_abs`).
    """
    if not slots:
        return
    H, W, _ = img_bgr.shape
    slots_sorted = sorted(slots, key=lambda s: s["x"])
    max_bridge_px = int(round(3.6 * px_per_mm))
    ext_pad_px = int(round(0.32 * px_per_mm))

    # Group slots into contiguous letter clusters (only underline contiguous >=2 letter runs, or top singletons if draw_labels)
    groups: list[list[dict]] = [[slots_sorted[0]]]
    for s in slots_sorted[1:]:
        prev_s = groups[-1][-1]
        gap = s["x"] - (prev_s["x"] + prev_s["w"])
        if gap <= max_bridge_px:
            groups[-1].append(s)
        else:
            groups.append([s])

    cyan_color = (235, 222, 0)  # Nieuwlaar Cyan in BGR
    labeled_line = False
    for grp in groups:
        if len(grp) == 1 and len(slots_sorted) >= 3 and not draw_labels:
            # In pure Nieuwlaar view, underline contiguous runs of >=2 letters (plus short sub-baseline if strong)
            if grp[0].get("tpl_score", 0.0) < 0.44:
                continue
        x0_g = max(4, grp[0]["x"] - ext_pad_px)
        x1_g = min(W - 5, grp[-1]["x"] + grp[-1]["w"] + ext_pad_px)
        knot_xs = [int(s["x"] + 0.5 * s["w"]) for s in grp]
        knot_ys = []
        for s in grp:
            kx = min(W - 1, max(0, int(s["x"] + 0.5 * s["w"])))
            yb_f = float(y_base_x[kx])
            st_b = float(s.get("y_stroke_bot_abs", s["y"] + s["h"]))
            # Follow the fiber baseline `yb_f` while hugging the individual letter bottom within +-12 px
            dev = np.clip(st_b - yb_f, -12.0, 12.0)
            knot_ys.append(int(round(yb_f + 0.45 * dev)) + 3)
        # Prepend start and append end knots along the fiber baseline
        y_start = int(round(float(y_base_x[min(W - 1, max(0, x0_g))]))) + 3
        y_end = int(round(float(y_base_x[min(W - 1, max(0, x1_g))]))) + 3
        knot_xs = [x0_g] + knot_xs + [x1_g]
        knot_ys = [y_start] + knot_ys + [y_end]

        pts = build_chamfered_45deg_polyline(x0_g, x1_g, knot_xs, knot_ys, H)
        if pts.size > 0:
            cv2.polylines(img_bgr, [pts], isClosed=False, color=(18, 30, 36), thickness=5, lineType=cv2.LINE_AA)
            cv2.polylines(img_bgr, [pts], isClosed=False, color=cyan_color, thickness=3, lineType=cv2.LINE_AA)

        if draw_labels:
            if not labeled_line:
                y_l0 = int(np.clip(knot_ys[0] + 6, 22, H - 10))
                cv2.putText(img_bgr, line_name, (max(6, x0_g - 32), y_l0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA)
                labeled_line = True
            for s in grp:
                cx = int(s["x"] + 0.40 * s["w"])
                yb = int(np.clip(s.get("y_stroke_bot_abs", s["y"] + s["h"]) + 24, 24, H - 8))
                cv2.putText(img_bgr, s["greek_ascii"], (cx, yb),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (110, 255, 160), 2, cv2.LINE_AA)


def render_pure_nieuwlaar_submission_image(
    gold_2cm: np.ndarray,
    lines_out: list[dict],
    decoded_lines_info: dict,
    px_per_mm: float,
    out_path: Path,
):
    canvas = gold_2cm.copy()
    H, W, _ = canvas.shape
    for l_item in lines_out:
        lname = l_item["line_name"]
        if lname in decoded_lines_info:
            draw_nieuwlaar_cyan_reading_underline(
                canvas,
                l_item["y_base_x"],
                decoded_lines_info[lname]["_slots"],
                px_per_mm,
                draw_labels=False,
                line_name=lname,
            )

    px_1mm = int(round(1.0 * px_per_mm))
    bar_w = max(140, px_1mm * 2)
    cv2.rectangle(canvas, (16, H - 46), (16 + bar_w + 24, H - 12), (245, 245, 245), -1)
    x_b0 = 28 + (bar_w - px_1mm) // 2
    cv2.line(canvas, (x_b0, H - 36), (x_b0 + px_1mm, H - 36), (20, 20, 20), 3)
    cv2.line(canvas, (x_b0, H - 41), (x_b0, H - 31), (20, 20, 20), 2)
    cv2.line(canvas, (x_b0 + px_1mm, H - 41), (x_b0 + px_1mm, H - 31), (20, 20, 20), 2)
    cv2.putText(canvas, "1 mm", (x_b0 + max(0, (px_1mm - 42) // 2), H - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def render_diplomatic_column_lines_board(
    scroll: str,
    cand_id: str,
    orient: str,
    best_d_key: str,
    win_mm: float,
    tilt_deg: float,
    r_max: float,
    px_per_mm: float,
    gold_2cm: np.ndarray,
    lines_out: list[dict],
    decoded_lines_info: dict,
    out_path: Path,
):
    annot = gold_2cm.copy()
    wh, ww, _ = annot.shape
    step_2mm = int(round(2.0 * px_per_mm))

    for l_item in lines_out:
        lname = l_item["line_name"]
        if lname in decoded_lines_info:
            slots = decoded_lines_info[lname]["_slots"]
            draw_nieuwlaar_cyan_reading_underline(
                annot,
                l_item["y_base_x"],
                slots,
                px_per_mm,
                draw_labels=True,
                line_name=lname,
            )

    cv2.rectangle(annot, (ww - 24 - step_2mm, 24), (ww - 24, 24 + step_2mm), (0, 235, 255), 2)
    cv2.putText(
        annot, "2x2mm", (ww - 28 - step_2mm, 44 + step_2mm),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 235, 255), 1, cv2.LINE_AA,
    )

    line_keys = sorted(decoded_lines_info.keys())
    n_rows = max(5, len(line_keys))
    total_h = n_rows * 200
    left_panel = np.zeros((total_h, 960, 3), dtype=np.uint8)
    left_panel[:] = (14, 15, 18)
    map_h = min(960, total_h - 40)
    y_map0 = 40 + (total_h - 40 - map_h) // 2
    left_panel[y_map0:y_map0 + map_h, 0:960] = cv2.resize(annot, (960, map_h), interpolation=cv2.INTER_AREA)
    hdr = f"STAGE L (ULTRA-PRECISE INK + CT-FIBER): {scroll}/{cand_id} ({orient}, {best_d_key}, tilt={tilt_deg:+.1f}deg) | r={r_max:.4f}"
    cv2.putText(left_panel, hdr, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 248, 255), 1, cv2.LINE_AA)

    right_bands = []
    for row_idx in range(n_rows):
        band = np.zeros((200, 960, 3), dtype=np.uint8)
        band[:] = (20, 21, 26)
        if row_idx < len(line_keys):
            lk = line_keys[row_idx]
            linfo = decoded_lines_info[lk]
            slots = linfo["_slots"]
            words_str = ", ".join(linfo["matched_epicurean_words"]) if linfo["matched_epicurean_words"] else "continuous line"
            title = (
                f"{lk} (y={linfo['y_mm']:.1f}mm):  [{linfo['diplomatic_ascii']}]   "
                f"dz4=-{linfo['mean_drop_pm4_pct']:.0f}%  tpl={linfo['mean_tpl_score']:.2f}  |  Lexicon: {words_str}"
            )
            cv2.putText(band, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 235, 255), 1, cv2.LINE_AA)

            for s_idx, s in enumerate(slots[:8]):
                x_off = 10 + s_idx * 118
                rib = s["_ribbon_u8"]
                bh_b, bw_b = rib.shape
                pad = max(12, bh_b // 7)
                rib_pad = np.zeros((bh_b + 2 * pad, bw_b + 2 * pad), dtype=np.uint8)
                rib_pad[pad:pad + bh_b, pad:pad + bw_b] = rib

                r_tile = cv2.cvtColor(cv2.resize(rib_pad, (54, 134), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
                t_tile = cv2.cvtColor(cv2.resize(s["_tpl_u8"], (54, 134), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
                t_tile[:, :, 0] = (t_tile[:, :, 0].astype(np.float32) * 0.55).astype(np.uint8)

                band[46:180, x_off:x_off + 54] = r_tile
                band[46:180, x_off + 56:x_off + 110] = t_tile
                cv2.line(band, (x_off + 2, 178), (x_off + 52, 178), (235, 220, 0), 2)
                cv2.rectangle(band, (x_off, 46), (x_off + 110, 180), (65, 75, 90), 1)
                sub_lbl = f"{s['greek_ascii']} (-{s['stroke_drop_eff_pm4_pct']:.0f}%)"
                cv2.putText(band, sub_lbl, (x_off + 6, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 235, 250), 1, cv2.LINE_AA)

        cv2.rectangle(band, (0, 0), (959, 199), (55, 60, 72), 1)
        right_bands.append(band)

    right_panel = np.vstack(right_bands)
    board = np.hstack([left_panel, right_panel])
    cv2.imwrite(str(out_path), board)


def decode_continuous_lines_on_candidate(seg_dir: Path) -> dict | None:
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
    ct_2cm = ct_mid[sl]
    H, W = ct_2cm.shape

    flat_2cm, ct_lamina_stats = compute_strict_single_lamina_ct_metrics(ct_2cm)
    if ct_lamina_stats["valid_fiber_coverage"] < 0.78:
        return None

    lamina_maps = build_calligraphic_thin_lamina_maps(seg_dir, orient, best_d_int, sl, flat_2cm)
    callig_2cm = lamina_maps["callig_u8"]
    dn_t_2cm = lamina_maps["dn_thin_u8"]
    hc_t_2cm = lamina_maps["hc_thin_u8"]
    dn_rg_2cm = lamina_maps["dn_ridge_u8"]

    clean_initial = (callig_2cm.astype(np.float32) * np.clip(flat_2cm, 0.0, 1.0)).astype(np.uint8)
    tilt_deg, lines_out = trace_staircase_reading_baselines(ct_2cm, clean_initial, flat_2cm, px_per_mm)

    clean, _ = calibrate_ink_with_reading_baselines(
        callig_2cm, dn_t_2cm, hc_t_2cm, flat_2cm, lines_out, px_per_mm
    )

    nom_w = int(round(2.30 * px_per_mm))
    nom_h = int(round(2.48 * px_per_mm))

    all_decoded_boxes = []
    decoded_lines_info = {}
    display_callig = (clean.astype(np.float32) * 0.65).astype(np.uint8)

    for l_item in lines_out:
        idx_l = l_item["line_id"] - 1
        line_name = l_item["line_name"]
        y_base_x = l_item["y_base_x"]

        strip = np.zeros((nom_h, W), dtype=np.float32)
        for col_x in range(W):
            yb = int(np.clip(y_base_x[col_x], nom_h, H))
            strip[:, col_x] = clean[yb - nom_h:yb, col_x]

        col_prof = np.percentile(strip, 88.0, axis=0)
        k_col = max(15, int(round(0.75 * px_per_mm)) | 1)
        col_smooth = cv2.GaussianBlur(col_prof.reshape(-1, 1), (1, k_col), 0).ravel()

        picked_xs = []
        thr_col = max(22.0, float(np.percentile(col_smooth, 40.0)))
        for x_idx in np.argsort(col_smooth)[::-1]:
            xi = int(x_idx)
            if xi < nom_w // 2 or xi > W - nom_w // 2:
                continue
            if col_smooth[xi] < thr_col:
                continue
            if all(abs(xi - px) >= int(round(1.82 * px_per_mm)) for px in picked_xs):
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
            dn_p94 = float(np.percentile(patch_dn, 94.0))
            hc_p985 = float(np.percentile(patch_hc, 98.5))
            rg_p92 = float(np.percentile(patch_rg, 92.0))
            if dn_p94 < 25.0 or hc_p985 < 12.5 or rg_p92 < 23.0:
                continue

            patch_cal = clean[best_y0:best_y0 + nom_h, x0:x0 + nom_w]
            rib_raw, fill_r, dom_r = isolate_central_character_ribbon(patch_cal)
            if fill_r < 0.08 or fill_r > 0.50 or dom_r < 0.65:
                continue

            top_cands, centered_rib, y_st_top, y_st_bot = score_top_k_greek_templates(rib_raw, top_k=5)
            if top_cands[0]["tpl_score"] < 0.34:
                continue

            # Place `rib_raw` on the full map, but use `centered_rib` for the In-Focus card so strokes are centered!
            display_callig[best_y0:best_y0 + nom_h, x0:x0 + nom_w] = np.maximum(
                display_callig[best_y0:best_y0 + nom_h, x0:x0 + nom_w],
                rib_raw,
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
            min_drop_pm4_pct=30.0,
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

    r_max = max(float(meta["r_raw_2x2cm_dstar"]), float(meta["r_hp_2x2cm_dstar"]), float(meta["r_thin_2x2cm_dstar"]))
    out_dir = STAGE_J_DIR / f"{scroll_key}__{cand_id}"
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
        orient,
        best_d_key,
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
        orient,
        f"StageJ-v3-Nieuwlaar45-{best_d_key}",
        float(flat_2cm.mean()),
        r_max,
        px_per_mm,
        gold_with_cyan,
        display_callig,
        boxes_by_score,
        line_ys_mid,
        out_dir,
    )

    all_words = []
    for lk, lv in decoded_lines_info.items():
        all_words.extend(lv["matched_epicurean_words"])
        lv.pop("_slots", None)
    for b in all_decoded_boxes:
        b.pop("_ribbon_u8", None)
        b.pop("_tpl_u8", None)

    mean_drop = float(np.mean([b["stroke_drop_eff_pm4_pct"] for b in all_decoded_boxes]))
    mean_tpl = float(np.mean([b["tpl_score"] for b in all_decoded_boxes]))
    full_ascii = " / ".join(f"{k}:{v['diplomatic_ascii']}" for k, v in decoded_lines_info.items())
    full_uni = " / ".join(f"{k}:{v['diplomatic_unicode']}" for k, v in decoded_lines_info.items())

    return {
        "scroll_key": scroll_key,
        "cand_id": cand_id,
        "window_mm": win_mm,
        "fiber_line_tilt_deg": tilt_deg,
        "orient": orient,
        "best_single_depth": best_d_key,
        "r_max_2x2cm": round(r_max, 4),
        "total_decoded_chars": len(all_decoded_boxes),
        "num_lines_decoded": len(decoded_lines_info),
        "mean_box_drop_pm4_pct": round(mean_drop, 1),
        "mean_tpl_score": round(mean_tpl, 3),
        "valid_fiber_coverage": ct_lamina_stats["valid_fiber_coverage"],
        "single_sheet_flatness_pct": ct_lamina_stats["single_sheet_flatness_pct"],
        "swirl_delam_frac": ct_lamina_stats["swirl_delam_frac"],
        "diplomatic_lines_ascii": full_ascii,
        "diplomatic_lines_unicode": full_uni,
        "matched_epicurean_words": sorted(set(all_words)),
        "lines_detail": decoded_lines_info,
    }


def main():
    priority_Set = {
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
        if sd.is_dir() and (sd / "metrics_stage_i.json").exists() and sd.name in priority_Set
    ]
    print(f"[Stage J v3] Running Nieuwlaar 45-Deg Baseline + Stroke-Centered Viterbi Decoder on {len(seg_dirs)} candidates...", flush=True)
    results = []
    for idx, sd in enumerate(seg_dirs, 1):
        r = decode_continuous_lines_on_candidate(sd)
        if r is not None:
            results.append(r)
            print(
                f"  [{idx:2d}/{len(seg_dirs)}] {r['scroll_key']}/{r['cand_id']:28s} (tilt={r['fiber_line_tilt_deg']:+5.1f}deg) "
                f"| chars={r['total_decoded_chars']:2d} | drop_pm4=-{r['mean_box_drop_pm4_pct']:4.1f}% "
                f"| tpl={r['mean_tpl_score']:.2f} | words={r['matched_epicurean_words']} | {r['diplomatic_lines_ascii']}",
                flush=True,
            )

    results.sort(
        key=lambda r: (
            len(r["matched_epicurean_words"]) > 0,
            r["r_max_2x2cm"] * (r["mean_box_drop_pm4_pct"] / 100.0) * (r["single_sheet_flatness_pct"] / 100.0),
        ),
        reverse=True,
    )
    out_json = STAGE_J_DIR / "leaderboard_stage_j_lines.json"
    out_json.write_text(json.dumps({"decoded_candidates": results}, indent=2))
    print(f"=== Stage J v3 Complete ({len(results)} candidates saved to {out_json}) ===", flush=True)


if __name__ == "__main__":
    main()
