#!/usr/bin/env python3
"""
Stage F (Part 1) — Spatially-Smooth In-Focus Sheet Tracker, Multi-Window (Top-3)
2x2 cm Explorer, and Greek Uncial Paleographic Classifier & Dossier Generator.

Key capabilities:
1. Spatially-Smooth In-Focus Depth Field w_k(y, x) across d in {-8, -4, 0, +4, +8}:
   Tracks gentle 3D papyrus sheet undulations across the 2x2 cm window without merging
   adjacent wraps (uses a ~1.2 mm spatial coherence kernel weighted by cross-family r_2x2cm(d)).
2. Multi-Window (Top-3) 2x2 cm (4 cm^2) Search:
   Evaluates the top 3 distinct 2x2 cm windows per segment so a cleaner text column
   a few millimeters away is never missed.
3. Herculaneum Greek Uncial Paleographic Template Bank (20 letters: Ι, Τ, Γ, Π, Η, Ν, Μ,
   Λ, Α, Δ, Ο, Θ, Ϲ, Ε, Υ, Χ, Ρ, Ω, Κ, Φ):
   Scores every >=2.0x2.0 mm dual-model character candidate by stroke-topology match,
   dual-family (Villa x Hecate) agreement, single-sheet flatness, and horizontal line collinearity.
4. Generates `paleography_dossier.png` and `top3_windows_overview.png` for each segment.
"""
import json
from pathlib import Path
import cv2
import numpy as np


BASE = Path(__file__).resolve().parent.parent
STAGE_D_DIR = BASE / "results_stage_d"
STAGE_F_DIR = BASE / "results_stage_f"


def load_gray(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def highpass_norm(img: np.ndarray, valid: np.ndarray, sigma: float = 42.0,
                  p_low: float = 55.0, p_high: float = 99.6) -> np.ndarray:
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hp = np.maximum(0.0, f - bg * 0.88)
    vals = hp[valid]
    if vals.size < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo + 1e-4:
        return np.zeros_like(img, dtype=np.uint8)
    out = np.clip((hp - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = 0.0
    return (out * 255.0).astype(np.uint8)


def compute_single_sheet_flatness(ct: np.ndarray) -> np.ndarray:
    """
    Returns a float32 weight map in [0, 1] measuring how flat and intact the single
    papyrus sheet is at each pixel:
    - Penalizes dark inter-wrap air gaps (0 < ct < 45)
    - Penalizes extreme local CT gradient contours caused by oblique cuts across multiple wraps
    - Requires high-frequency papyrus fiber weave presence (rejects smooth interpolated hole-fills)
    """
    valid = (ct > 0).astype(np.float32)
    ct_f = ct.astype(np.float32)

    is_gap = ((ct > 0) & (ct < 45)).astype(np.float32)
    gap_frac = cv2.GaussianBlur(is_gap, (0, 0), sigmaX=35.0, sigmaY=35.0)
    no_gap_score = np.clip(1.0 - gap_frac * 4.5, 0.05, 1.0)

    gx = cv2.Sobel(ct_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ct_f, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    g_smooth = cv2.GaussianBlur(gmag, (0, 0), sigmaX=40.0, sigmaY=40.0)
    v_vals = g_smooth[ct > 45]
    if v_vals.size > 100:
        g_p80 = float(np.percentile(v_vals, 80.0))
        g_p98 = float(np.percentile(v_vals, 98.0))
        flat_grad_score = np.clip(1.0 - 0.75 * np.clip((g_smooth - g_p80) / max(1.0, g_p98 - g_p80), 0.0, 1.0), 0.2, 1.0)
    else:
        flat_grad_score = np.ones_like(ct_f)

    ct_blur3 = cv2.GaussianBlur(ct_f, (0, 0), sigmaX=2.5, sigmaY=2.5)
    hf_weave = cv2.GaussianBlur(np.abs(ct_f - ct_blur3), (0, 0), sigmaX=14.0, sigmaY=14.0)
    fiber_presence = np.clip((hf_weave - 2.2) / 3.8, 0.0, 1.0)

    return valid * no_gap_score * flat_grad_score * fiber_presence


def build_greek_uncial_templates(sz: int = 64) -> dict[str, list[np.ndarray]]:
    """
    Build 64x64 grayscale stroke templates for 20 Herculaneum Greek uncial characters
    (1st century BC papyrus bookhand: lunate sigma C, broad pi/eta/nu/mu, etc.)
    with 2 stroke-thickness / aspect variants per letter.
    Each template is zero-mean unit-norm float32 for fast normalized cross-correlation,
    plus we keep a uint8 visual prototype for the paleographic dossier.
    """
    raw_drawings: dict[str, list[np.ndarray]] = {}

    def make_canvas(variants_fn):
        arrs = []
        for thick, sx in [(6, 1.0), (8, 0.88)]:
            im = np.zeros((sz, sz), dtype=np.uint8)
            variants_fn(im, thick, sx)
            im_blur = cv2.GaussianBlur(im, (5, 5), sigmaX=1.4)
            arrs.append(im_blur)
        return arrs

    def pt(x_norm: float, y_norm: float, sx: float) -> tuple[int, int]:
        cx = 0.5 + (x_norm - 0.5) * sx
        return (int(round(cx * (sz - 1))), int(round(y_norm * (sz - 1))))

    # 1. Ι (Iota)
    raw_drawings["I"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.50, 0.16, sx), pt(0.50, 0.84, sx), 255, th, cv2.LINE_AA),
    ))
    # 2. Τ (Tau)
    raw_drawings["T"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.20, 0.20, sx), pt(0.80, 0.20, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.50, 0.20, sx), pt(0.50, 0.84, sx), 255, th, cv2.LINE_AA),
    ))
    # 3. Γ (Gamma)
    raw_drawings["G"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.26, 0.20, sx), pt(0.76, 0.20, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.26, 0.20, sx), pt(0.26, 0.84, sx), 255, th, cv2.LINE_AA),
    ))
    # 4. Π (Pi)
    raw_drawings["PI"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.20, 0.20, sx), pt(0.80, 0.20, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.24, 0.20, sx), pt(0.24, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.76, 0.20, sx), pt(0.76, 0.84, sx), 255, th, cv2.LINE_AA),
    ))
    # 5. Η (Eta)
    raw_drawings["H"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.22, 0.18, sx), pt(0.22, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.78, 0.18, sx), pt(0.78, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.22, 0.50, sx), pt(0.78, 0.50, sx), 255, th, cv2.LINE_AA),
    ))
    # 6. Ν (Nu)
    raw_drawings["N"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.22, 0.18, sx), pt(0.22, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.78, 0.18, sx), pt(0.78, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.22, 0.22, sx), pt(0.78, 0.80, sx), 255, th, cv2.LINE_AA),
    ))
    # 7. Μ (Mu)
    raw_drawings["M"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.18, 0.84, sx), pt(0.22, 0.20, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.22, 0.20, sx), pt(0.50, 0.62, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.50, 0.62, sx), pt(0.78, 0.20, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.78, 0.20, sx), pt(0.82, 0.84, sx), 255, th, cv2.LINE_AA),
    ))
    # 8. Λ (Lambda)
    raw_drawings["L"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.50, 0.18, sx), pt(0.20, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.50, 0.18, sx), pt(0.80, 0.84, sx), 255, th, cv2.LINE_AA),
    ))
    # 9. Α (Alpha)
    raw_drawings["A"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.50, 0.18, sx), pt(0.20, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.50, 0.18, sx), pt(0.80, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.28, 0.62, sx), pt(0.72, 0.56, sx), 255, max(4, th - 1), cv2.LINE_AA),
    ))
    # 10. Δ (Delta)
    raw_drawings["D"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.50, 0.18, sx), pt(0.20, 0.82, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.50, 0.18, sx), pt(0.80, 0.82, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.20, 0.82, sx), pt(0.80, 0.82, sx), 255, th, cv2.LINE_AA),
    ))
    # 11. Ο (Omicron)
    raw_drawings["O"] = make_canvas(lambda im, th, sx: (
        cv2.ellipse(im, (sz // 2, sz // 2), (int(round(0.28 * sx * sz)), int(round(0.31 * sz))), 0, 0, 360, 255, th, cv2.LINE_AA),
    ))
    # 12. Θ (Theta)
    raw_drawings["TH"] = make_canvas(lambda im, th, sx: (
        cv2.ellipse(im, (sz // 2, sz // 2), (int(round(0.28 * sx * sz)), int(round(0.31 * sz))), 0, 0, 360, 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.28, 0.50, sx), pt(0.72, 0.50, sx), 255, max(4, th - 1), cv2.LINE_AA),
    ))
    # 13. Ϲ (Lunate Sigma)
    raw_drawings["C"] = make_canvas(lambda im, th, sx: (
        cv2.ellipse(im, (int(sz * 0.54), sz // 2), (int(round(0.28 * sx * sz)), int(round(0.31 * sz))), 0, 55, 305, 255, th, cv2.LINE_AA),
    ))
    # 14. Ε (Lunate Epsilon)
    raw_drawings["E"] = make_canvas(lambda im, th, sx: (
        cv2.ellipse(im, (int(sz * 0.54), sz // 2), (int(round(0.28 * sx * sz)), int(round(0.31 * sz))), 0, 55, 305, 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.26, 0.50, sx), pt(0.68, 0.50, sx), 255, max(4, th - 1), cv2.LINE_AA),
    ))
    # 15. Υ (Upsilon)
    raw_drawings["Y"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.22, 0.20, sx), pt(0.50, 0.52, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.78, 0.20, sx), pt(0.50, 0.52, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.50, 0.52, sx), pt(0.50, 0.84, sx), 255, th, cv2.LINE_AA),
    ))
    # 16. Χ (Chi)
    raw_drawings["X"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.22, 0.20, sx), pt(0.78, 0.82, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.78, 0.20, sx), pt(0.22, 0.82, sx), 255, th, cv2.LINE_AA),
    ))
    # 17. Κ (Kappa)
    raw_drawings["K"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.24, 0.18, sx), pt(0.24, 0.84, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.74, 0.20, sx), pt(0.26, 0.52, sx), 255, th, cv2.LINE_AA),
        cv2.line(im, pt(0.28, 0.50, sx), pt(0.76, 0.82, sx), 255, th, cv2.LINE_AA),
    ))
    # 18. Ρ (Rho)
    raw_drawings["P"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.28, 0.18, sx), pt(0.28, 0.86, sx), 255, th, cv2.LINE_AA),
        cv2.ellipse(im, (int(sz * 0.46), int(sz * 0.36)), (int(round(0.22 * sx * sz)), int(round(0.18 * sz))), 0, -90, 90, 255, th, cv2.LINE_AA),
    ))
    # 19. Ω (Uncial Omega - w-shaped double bowl)
    raw_drawings["W"] = make_canvas(lambda im, th, sx: (
        cv2.ellipse(im, (int(sz * 0.35), int(sz * 0.52)), (int(round(0.16 * sx * sz)), int(round(0.24 * sz))), 0, 10, 175, 255, th, cv2.LINE_AA),
        cv2.ellipse(im, (int(sz * 0.65), int(sz * 0.52)), (int(round(0.16 * sx * sz)), int(round(0.24 * sz))), 0, 5, 170, 255, th, cv2.LINE_AA),
    ))
    # 20. Φ (Phi)
    raw_drawings["PH"] = make_canvas(lambda im, th, sx: (
        cv2.line(im, pt(0.50, 0.12, sx), pt(0.50, 0.88, sx), 255, th, cv2.LINE_AA),
        cv2.ellipse(im, (sz // 2, sz // 2), (int(round(0.26 * sx * sz)), int(round(0.20 * sz))), 0, 0, 360, 255, max(4, th - 1), cv2.LINE_AA),
    ))

    return raw_drawings


GREEK_DISPLAY_UNICODE = {
    "I": "Ι", "T": "Τ", "G": "Γ", "PI": "Π", "H": "Η",
    "N": "Ν", "M": "Μ", "L": "Λ", "A": "Α", "D": "Δ",
    "O": "Ο", "TH": "Θ", "C": "Ϲ", "E": "Ε", "Y": "Υ",
    "X": "Χ", "K": "Κ", "P": "Ρ", "W": "Ω", "PH": "Φ",
}

TEMPLATES_64 = build_greek_uncial_templates(64)


def classify_greek_letter_patch(patch_clean: np.ndarray) -> tuple[str, str, float, np.ndarray]:
    """
    Match a candidate character patch against the 20 Herculaneum Greek uncial templates.
    Returns (best_ascii_code, best_unicode_char, match_score, matched_template_u8).
    """
    sz = 64
    p_rsz = cv2.resize(patch_clean, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Suppress border noise with a gentle radial taper while keeping central character strokes
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32)
    r2 = ((yy - 31.5) / 28.0) ** 2 + ((xx - 31.5) / 28.0) ** 2
    taper = np.clip(1.15 - 0.35 * r2, 0.25, 1.0)
    lo = float(np.percentile(p_rsz, 35.0))
    hi = float(np.percentile(p_rsz, 98.0))
    p_norm = np.clip((p_rsz - lo) / max(1.0, hi - lo), 0.0, 1.0) * taper
    p_zm = p_norm - float(p_norm.mean())
    p_std = float(np.sqrt(np.sum(p_zm * p_zm))) + 1e-6

    best_code = "O"
    best_sc = -1.0
    best_tpl = TEMPLATES_64["O"][0]

    for code, tpl_list in TEMPLATES_64.items():
        for tpl_u8 in tpl_list:
            # Allow small +-4 px translation shifts via matchTemplate on padded patch
            p_pad = cv2.copyMakeBorder(p_norm, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0.0)
            res = cv2.matchTemplate(p_pad, tpl_u8.astype(np.float32) / 255.0, cv2.TM_CCOEFF_NORMED)
            ncc = float(res.max())
            # Also compute soft structural overlap (IoU-like) at best shift
            min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
            ox, oy = max_loc
            p_shifted = p_pad[oy:oy + sz, ox:ox + sz]
            t_f = tpl_u8.astype(np.float32) / 255.0
            inter = float(np.sum(np.minimum(p_shifted, t_f)))
            union = float(np.sum(np.maximum(p_shifted, t_f))) + 1e-5
            iou = inter / union
            sc = 0.65 * max(0.0, ncc) + 0.35 * iou
            if sc > best_sc:
                best_sc = sc
                best_code = code
                best_tpl = tpl_u8

    return best_code, GREEK_DISPLAY_UNICODE[best_code], round(float(best_sc), 3), best_tpl


def compute_spatially_tracked_infocus(meta: dict, seg_dir: Path, orient: str, valid: np.ndarray):
    """
    Computes BOTH:
    1. Global best single depth (best_k) and best contiguous pair (pair_label, infocus_2d)
    2. Spatially-Smooth In-Focus Sheet-Tracked Map (`spatial_infocus`), where at each pixel (y, x)
       we softly weight contiguous depth pairs based on local ~1.2 mm ink stroke contrast AND
       global cross-family correlation r_2x2cm(d).
    This tracks gentle 3D sheet drift across the 2x2 cm window without mixing out-of-focus wraps!
    """
    depths_meta = meta.get("depths", {})
    ordered_keys = ["d-8", "d-4", "d+0", "d+4", "d+8"]
    scores = {}
    maps = {}
    hp_maps = {}
    for k in ordered_keys:
        p = seg_dir / f"ens_z_{orient}_{k}.png"
        m = load_gray(p)
        if m is not None:
            maps[k] = m
            hp_maps[k] = highpass_norm(m, valid, sigma=42.0, p_low=55.0, p_high=99.6)
            dm = depths_meta.get(k, {})
            r_all = float(dm.get("r_villa_hecate", 0.0))
            r_2cm = float(dm.get("max_2x2cm_r_villa_hecate", r_all))
            if r_2cm < 0:
                r_2cm = r_all
            scores[k] = max(0.02, 0.55 * r_2cm + 0.45 * r_all)

    if not maps:
        md = load_gray(seg_dir / f"ens_z_{orient}_multidepth.png")
        md_hp = highpass_norm(md, valid, sigma=42.0, p_low=55.0, p_high=99.6)
        return "multidepth", "multidepth", md_hp, md_hp, md_hp

    best_k = max(scores.keys(), key=lambda k: scores[k])
    best_1d_hp = hp_maps[best_k]

    best_pair = (best_k,)
    best_pair_sc = -1e9
    for i in range(len(ordered_keys) - 1):
        k1, k2 = ordered_keys[i], ordered_keys[i + 1]
        if k1 in scores and k2 in scores:
            sc = scores[k1] + scores[k2]
            if sc > best_pair_sc:
                best_pair_sc = sc
                best_pair = (k1, k2)

    if len(best_pair) == 2:
        k1, k2 = best_pair
        w1, w2 = scores[k1], scores[k2]
        m1 = maps[k1].astype(np.float32)
        m2 = maps[k2].astype(np.float32)
        pair_blend = ((w1 * m1 + w2 * m2) / (w1 + w2)) * 0.45 + np.maximum(m1, m2) * 0.55
        infocus_2d_hp = highpass_norm(np.clip(pair_blend, 0, 255).astype(np.uint8), valid, sigma=42.0, p_low=55.0, p_high=99.6)
        pair_label = f"{k1}+{k2}"
    else:
        infocus_2d_hp = best_1d_hp.copy()
        pair_label = best_k

    # Spatially-Smooth Sheet Tracker across contiguous depth pairs:
    # Build 4 contiguous pair maps P_0=(-8,-4), P_1=(-4,0), P_2=(0,+4), P_3=(+4,+8)
    pair_imgs = []
    pair_energy = []
    for i in range(len(ordered_keys) - 1):
        k1, k2 = ordered_keys[i], ordered_keys[i + 1]
        if k1 in hp_maps and k2 in hp_maps:
            h1 = hp_maps[k1].astype(np.float32)
            h2 = hp_maps[k2].astype(np.float32)
            w1, w2 = scores[k1], scores[k2]
            p_im = 0.50 * ((w1 * h1 + w2 * h2) / (w1 + w2)) + 0.50 * np.maximum(h1, h2)
            # Local stroke sharpness energy smoothed at sigma=110 px (~1.0 mm)
            e_loc = cv2.GaussianBlur(p_im, (0, 0), sigmaX=110.0, sigmaY=110.0)
            prior_pair = (w1 + w2) ** 1.35
            pair_imgs.append(p_im)
            pair_energy.append(e_loc * prior_pair)

    if len(pair_imgs) >= 2:
        e_stack = np.stack(pair_energy, axis=0)
        # Softmax with strong selectivity (temperature) so at each (y, x) only the locally in-focus pair dominates
        e_max = np.max(e_stack, axis=0, keepdims=True)
        e_std = max(float(np.std(e_max)), 1.0)
        logits = (e_stack - e_max) / (0.18 * e_std)
        weights = np.exp(np.clip(logits, -20.0, 0.0))
        weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-6)
        spatial_blend = np.sum(weights * np.stack(pair_imgs, axis=0), axis=0)
        spatial_hp = highpass_norm(spatial_blend, valid, sigma=45.0, p_low=48.0, p_high=99.6)
    else:
        spatial_hp = infocus_2d_hp.copy()

    return best_k, pair_label, best_1d_hp, infocus_2d_hp, spatial_hp


def find_top_k_flat_2x2cm_windows(ink_hp: np.ndarray, flat_w: np.ndarray, valid: np.ndarray,
                                  win_px: int, top_k: int = 3):
    """Find the top K distinct (non-heavily-overlapping) 2x2 cm windows on a flat single-sheet surface."""
    H, W = ink_hp.shape
    wh = min(win_px, H)
    ww = min(win_px, W)

    weighted_ink = ink_hp.astype(np.float32) * (flat_w ** 2.0)
    ii_ink = cv2.integral(weighted_ink)
    ii_flat = cv2.integral(flat_w)
    ii_val = cv2.integral(valid.astype(np.uint8))

    step = max(16, win_px // 20)
    candidates = []

    for y0 in range(0, H - wh + 1, step):
        y1 = y0 + wh
        for x0 in range(0, W - ww + 1, step):
            x1 = x0 + ww
            v_cnt = int(ii_val[y1, x1] - ii_val[y0, x1] - ii_val[y1, x0] + ii_val[y0, x0])
            if v_cnt < 0.55 * wh * ww:
                continue
            f_sum = float(ii_flat[y1, x1] - ii_flat[y0, x1] - ii_flat[y1, x0] + ii_flat[y0, x0])
            mean_flat = f_sum / max(1, v_cnt)
            if mean_flat < 0.45:
                continue
            ink_sum = float(ii_ink[y1, x1] - ii_ink[y0, x1] - ii_ink[y1, x0] + ii_ink[y0, x0])
            sc = (ink_sum / max(1, v_cnt)) * (mean_flat ** 0.5)
            candidates.append((sc, y0, x0, wh, ww, mean_flat))

    if not candidates:
        return [(0, 0, wh, ww, 0.0, float(flat_w[:wh, :ww].mean()))]

    candidates.sort(key=lambda c: c[0], reverse=True)
    picked = []
    min_dist = 0.48 * win_px  # centers at least ~1.0 cm apart
    for (sc, y0, x0, wh_c, ww_c, mf) in candidates:
        if all(np.hypot(y0 - py, x0 - px) >= min_dist for (py, px, _, _, _, _) in picked):
            picked.append((y0, x0, wh_c, ww_c, sc, mf))
            if len(picked) >= top_k:
                break

    return picked


def detect_and_classify_letters(crop_ink: np.ndarray, crop_dn: np.ndarray, crop_hc: np.ndarray,
                                crop_flat: np.ndarray, vox_um: float):
    """
    Detect >=2.0x2.0 mm character boxes, assign horizontal line membership (L1..L5),
    and classify each box with our 20-letter Greek uncial template bank.
    """
    H, W = crop_ink.shape
    px_per_mm = 1000.0 / vox_um
    min_char_px = int(np.ceil(2.02 * px_per_mm))
    max_char_px = int(round(4.50 * px_per_mm))
    nom_char_w = int(round(2.35 * px_per_mm))
    nom_char_h = int(round(2.45 * px_per_mm))

    dual_geom = np.sqrt(crop_dn.astype(np.float32) * crop_hc.astype(np.float32))
    clean = ((0.55 * crop_ink.astype(np.float32) + 0.45 * dual_geom) * np.clip(crop_flat, 0.0, 1.0)).astype(np.uint8)
    v_vals = clean[crop_flat > 0.45]
    if v_vals.size < 100:
        return [], []

    row_prof = np.percentile(clean.astype(np.float32), 88.0, axis=1)
    k_1d = max(5, int(round(1.1 * px_per_mm)) | 1)
    row_smooth = cv2.GaussianBlur(row_prof.reshape(-1, 1), (1, k_1d), 0).ravel()
    min_line_dist = int(round(2.75 * px_per_mm))

    line_ys = []
    order = np.argsort(row_smooth)[::-1]
    for y_idx in order:
        if y_idx < min_char_px // 2 or y_idx > H - min_char_px // 2:
            continue
        if all(abs(int(y_idx) - ly) >= min_line_dist for ly in line_ys):
            line_ys.append(int(y_idx))
            if len(line_ys) >= 5:
                break
    line_ys.sort()

    thr = float(np.percentile(v_vals, 89.5))
    bw = (clean >= max(70.0, thr)).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_closed, connectivity=8)
    raw_rects = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w < int(round(1.0 * px_per_mm)) and h < int(round(1.0 * px_per_mm)):
            continue
        n_v = max(1, int(round(h / (2.85 * px_per_mm)))) if h > max_char_px else 1
        sub_h = h // n_v
        for iv in range(n_v):
            sy = y + iv * sub_h
            sh = sub_h if iv < n_v - 1 else (y + h - sy)
            if w > max_char_px:
                n_h = max(2, int(round(w / nom_char_w)))
                if n_h > 9:
                    continue
                sub_strip = clean[sy:sy+sh, x:x+w].astype(np.float32)
                v_prof = np.percentile(sub_strip, 85.0, axis=0) if sub_strip.size > 0 else np.zeros(w, dtype=np.float32)
                cuts = [0]
                step_w = w / n_h
                search_rad = max(8, int(0.32 * nom_char_w))
                for ic in range(1, n_h):
                    nom_c = int(round(ic * step_w))
                    c_lo = max(cuts[-1] + int(0.55 * min_char_px), nom_c - search_rad)
                    c_hi = min(w - int(0.55 * min_char_px), nom_c + search_rad)
                    cut_x = c_lo + int(np.argmin(v_prof[c_lo:c_hi])) if c_hi > c_lo else nom_c
                    cuts.append(cut_x)
                cuts.append(w)
                for ih in range(len(cuts) - 1):
                    raw_rects.append((x + cuts[ih], sy, cuts[ih + 1] - cuts[ih], sh))
            else:
                raw_rects.append((x, sy, w, sh))

    for ly in line_ys:
        y0 = max(0, ly - nom_char_h // 2)
        y1 = min(H, y0 + nom_char_h)
        y0 = max(0, y1 - nom_char_h)
        strip = clean[y0:y1, :].astype(np.float32)
        col_prof = np.percentile(strip, 90.0, axis=0)
        k_col = max(5, int(round(0.65 * px_per_mm)) | 1)
        col_smooth = cv2.GaussianBlur(col_prof.reshape(-1, 1), (1, k_col), 0).ravel()
        c_order = np.argsort(col_smooth)[::-1]
        picked_xs = []
        for x_idx in c_order:
            if x_idx < nom_char_w // 2 or x_idx > W - nom_char_w // 2:
                continue
            if col_smooth[x_idx] < max(52.0, thr * 0.62):
                break
            if all(abs(int(x_idx) - px) >= int(round(2.05 * px_per_mm)) for px in picked_xs):
                picked_xs.append(int(x_idx))
                raw_rects.append((int(x_idx) - nom_char_w // 2, y0, nom_char_w, nom_char_h))
                if len(picked_xs) >= 8:
                    break

    cand_boxes = []
    for (x, y, w, h) in raw_rects:
        if w < min_char_px:
            pad_x = (min_char_px - w + 1) // 2
            x = max(0, min(W - min_char_px, x - pad_x))
            w = min(min_char_px, W - x)
        if h < min_char_px:
            pad_y = (min_char_px - h + 1) // 2
            y = max(0, min(H - min_char_px, y - pad_y))
            h = min(min_char_px, H - y)
        if w > max_char_px or h > max_char_px or w < min_char_px or h < min_char_px:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 2.2:
            continue

        patch_clean = clean[y:y+h, x:x+w]
        patch_dn = crop_dn[y:y+h, x:x+w]
        patch_hc = crop_hc[y:y+h, x:x+w]
        patch_fl = crop_flat[y:y+h, x:x+w]
        if patch_clean.size == 0:
            continue

        fl_val = float(np.mean(patch_fl))
        dn_val = float(np.percentile(patch_dn, 94.0))
        hc_val = float(np.percentile(patch_hc, 94.0))
        ink_p94 = float(np.percentile(patch_clean, 94.0))
        ink_p30 = float(np.percentile(patch_clean, 30.0))
        local_contrast = ink_p94 - ink_p30
        if fl_val < 0.52 or dn_val < 68.0 or hc_val < 68.0 or local_contrast < 28.0:
            continue

        g_code, g_uni, tpl_sc, tpl_u8 = classify_greek_letter_patch(patch_clean)
        cx = x + 0.5 * w
        cy = y + 0.5 * h

        # Line membership bonus if centered on one of the 5 horizontal text lines
        line_id = 0
        line_bonus = 1.0
        for idx_l, ly in enumerate(line_ys):
            if abs(cy - ly) <= 1.15 * px_per_mm:
                line_id = idx_l + 1
                line_bonus = 1.08
                break

        score = (0.42 * dn_val + 0.42 * hc_val + 0.28 * local_contrast) * (fl_val ** 0.75) * (0.85 + 0.35 * tpl_sc) * line_bonus
        cand_boxes.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "w_mm": round(w / px_per_mm, 2),
            "h_mm": round(h / px_per_mm, 2),
            "cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "dn_p94": round(dn_val, 1),
            "hc_p94": round(hc_val, 1),
            "contrast": round(local_contrast, 1),
            "flatness": round(fl_val, 2),
            "greek_ascii": g_code,
            "greek_char": g_uni,
            "tpl_score": tpl_sc,
            "line_id": line_id,
            "score": round(score, 1),
            "_tpl_u8": tpl_u8,
        })

    cand_boxes.sort(key=lambda b: b["score"], reverse=True)
    boxes = []
    min_center_dist = 1.92 * px_per_mm
    for b in cand_boxes:
        if all(np.hypot(b["cx"] - kb["cx"], b["cy"] - kb["cy"]) >= min_center_dist for kb in boxes):
            boxes.append(b)
            if len(boxes) >= 24:
                break

    return boxes, line_ys


def make_calligraphic_gold(ct: np.ndarray, ink: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ct_f = ct.astype(np.float32)
    vals = ct_f[valid]
    if vals.size > 100:
        lo, hi = np.percentile(vals, [2.0, 98.5])
        ct_norm = np.clip((ct_f - lo) / max(1.0, hi - lo), 0.0, 1.0)
    else:
        ct_norm = ct_f / 255.0
    base = (ct_norm * 165.0).astype(np.uint8)
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    ink_f = ink.astype(np.float32)
    ivals = ink_f[valid]
    if ivals.size > 100:
        p72 = float(np.percentile(ivals, 72.0))
        p997 = float(np.percentile(ivals, 99.7))
    else:
        p72, p997 = 64.0, 255.0
    alpha = np.clip((ink_f - p72) / max(1.0, p997 - p72), 0.0, 1.0) ** 1.15
    alpha[~valid] = 0.0

    gold = np.zeros_like(bgr, dtype=np.float32)
    gold[:, :, 0] = 60.0
    gold[:, :, 1] = 215.0
    gold[:, :, 2] = 255.0

    out = bgr.astype(np.float32) * (1.0 - 0.84 * alpha[:, :, None]) + gold * (0.84 * alpha[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def render_paleography_dossier(scroll: str, seg: str, orient: str, pair_label: str,
                               mean_flat: float, r_2x2cm: float, px_per_mm: float,
                               p1_gold_clean: np.ndarray, crop_callig: np.ndarray,
                               boxes: list[dict], line_ys: list[int], out_dir: Path):
    """
    Build a high-resolution Paleographic Evidence Dossier (`paleography_dossier.png`):
    - Left panel (960x960): Annotated 2x2 cm (4 cm^2) In-Focus Gold window with L1..L5 lines
      and numbered Greek character boxes (#1: PI, #2: H, etc.) + 2x2mm reference scale.
    - Right panel (960x960): 4x3 grid of the Top 12 Greek Character Evidence Cards, where each
      card shows [CT+Gold Crop | Spatially-Tracked Ink Crop | Matched Greek Uncial Template].
    """
    wh, ww = crop_callig.shape
    annot = p1_gold_clean.copy()
    step_2mm = int(round(2.0 * px_per_mm))

    for i_l, ly in enumerate(line_ys):
        cv2.line(annot, (0, ly), (ww - 1, ly), (255, 200, 80), 1, cv2.LINE_AA)
        cv2.putText(annot, f"L{i_l+1}", (10, max(20, ly - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 210, 90), 2, cv2.LINE_AA)

    for idx_b, b in enumerate(boxes[:20]):
        bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
        color = (80, 255, 120) if idx_b < 12 else (120, 220, 255)
        cv2.rectangle(annot, (bx, by), (bx + bw_b, by + bh_b), color, 2)
        lbl = f"#{idx_b+1}:{b['greek_ascii']} ({b['w_mm']:.1f}x{b['h_mm']:.1f}mm)"
        cv2.putText(annot, lbl, (bx, max(18, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # 2x2 mm scale box
    cv2.rectangle(annot, (ww - 24 - step_2mm, 24), (ww - 24, 24 + step_2mm), (0, 235, 255), 2)
    cv2.putText(annot, "2x2mm", (ww - 28 - step_2mm, 44 + step_2mm),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 235, 255), 1, cv2.LINE_AA)

    hdr = f"{scroll}/{seg} ({orient}, {pair_label}) | flat={mean_flat:.2f} | r_2cm={r_2x2cm:.4f} | {len(boxes)} letters >=2x2mm"
    cv2.rectangle(annot, (8, 8), (min(ww - 8, 18 + len(hdr) * 9), 38), (15, 15, 15), -1)
    cv2.putText(annot, hdr, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 245, 255), 1, cv2.LINE_AA)

    left_panel = cv2.resize(annot, (960, 960), interpolation=cv2.INTER_AREA)

    # Build 4 rows x 3 cols of 320x240 evidence cards (total 960x960)
    card_w, card_h = 320, 240
    sub_sz = 100  # 3 sub-tiles of 100x172 inside 320x240 card
    cards = []
    for idx_b in range(12):
        card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
        card[:] = (18, 18, 22)
        if idx_b < len(boxes):
            b = boxes[idx_b]
            bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
            pad = int(round(0.30 * px_per_mm))
            x0_c = max(0, bx - pad)
            y0_c = max(0, by - pad)
            x1_c = min(ww, bx + bw_b + pad)
            y1_c = min(wh, by + bh_b + pad)

            g_crop = cv2.resize(p1_gold_clean[y0_c:y1_c, x0_c:x1_c], (sub_sz, 172), interpolation=cv2.INTER_CUBIC)
            k_crop = cv2.cvtColor(
                cv2.resize(crop_callig[y0_c:y1_c, x0_c:x1_c], (sub_sz, 172), interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_GRAY2BGR,
            )
            tpl_bgr = cv2.cvtColor(
                cv2.resize(b["_tpl_u8"], (sub_sz, 172), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_GRAY2BGR,
            )
            # Tint template slightly cyan/gold for visual clarity
            tpl_bgr[:, :, 0] = np.clip(tpl_bgr[:, :, 0] * 0.6, 0, 255).astype(np.uint8)
            tpl_bgr[:, :, 1] = np.clip(tpl_bgr[:, :, 1] * 1.0, 0, 255).astype(np.uint8)

            card[52:224, 5:5 + sub_sz] = g_crop
            card[52:224, 10 + sub_sz:10 + 2 * sub_sz] = k_crop
            card[52:224, 15 + 2 * sub_sz:15 + 3 * sub_sz] = tpl_bgr

            t1 = f"#{idx_b+1} [{b['greek_ascii']}] {b['w_mm']:.1f}x{b['h_mm']:.1f}mm L{b['line_id']}"
            t2 = f"V={b['dn_p94']:.0f} H={b['hc_p94']:.0f} fl={b['flatness']:.2f} tpl={b['tpl_score']:.2f}"
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


def process_segment(seg_dir: Path, out_root: Path) -> dict | None:
    meta_path = seg_dir / "metrics_stage_d.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    scroll = meta["scroll"]
    seg = meta["segment"]
    orient = meta["chosen_orient"]
    vox_um = float(meta.get("um", 7.91))
    win_2cm_px = int(round(20000.0 / vox_um))
    px_per_mm = 1000.0 / vox_um

    ct = load_gray(seg_dir / "ct_mid.png")
    dn_md = load_gray(seg_dir / f"dnative_{orient}_multidepth.png")
    hc_md = load_gray(seg_dir / f"hecate_{orient}_multidepth.png")
    if ct is None or dn_md is None or hc_md is None:
        return None

    valid = ct > 0
    flat_w = compute_single_sheet_flatness(ct)
    best_k, pair_label, best1_hp, infocus2_hp, spatial_hp = compute_spatially_tracked_infocus(meta, seg_dir, orient, valid)

    dn_hp = highpass_norm(dn_md, valid, sigma=42.0, p_low=52.0, p_high=99.6)
    hc_hp = highpass_norm(hc_md, valid, sigma=50.0, p_low=50.0, p_high=99.6)

    # Spatially-Tracked Calligraphic Blend: combines spatially-tracked in-focus depth map + Villa + Hecate
    callig_f = 0.52 * spatial_hp.astype(np.float32) + 0.25 * dn_hp.astype(np.float32) + 0.23 * hc_hp.astype(np.float32)
    callig_hp = highpass_norm(callig_f, valid, sigma=42.0, p_low=50.0, p_high=99.6)

    # Evaluate Top 3 distinct 2x2 cm windows and pick the one with the strongest paleographic letter yield
    top_windows = find_top_k_flat_2x2cm_windows(callig_hp, flat_w, valid, win_2cm_px, top_k=3)
    win_evals = []
    for w_idx, (y0, x0, wh, ww, win_sc, mean_flat) in enumerate(top_windows):
        sl = (slice(y0, y0 + wh), slice(x0, x0 + ww))
        boxes, line_ys = detect_and_classify_letters(callig_hp[sl], dn_hp[sl], hc_hp[sl], flat_w[sl], vox_um)
        # Rank window by: (has >=10 letters, number of letters, mean_flat * top10_score)
        top10_sc = float(np.mean([b["score"] for b in boxes[:10]])) if boxes else 0.0
        win_evals.append({
            "win_rank": w_idx + 1,
            "yxwh": (y0, x0, wh, ww),
            "win_sc": win_sc,
            "mean_flat": mean_flat,
            "boxes": boxes,
            "line_ys": line_ys,
            "top10_sc": top10_sc,
            "combined_rank_sc": (1000.0 if len(boxes) >= 10 else 0.0) + len(boxes) * 18.0 + top10_sc * mean_flat,
        })

    win_evals.sort(key=lambda w: w["combined_rank_sc"], reverse=True)
    best_win = win_evals[0]
    y0, x0, wh, ww = best_win["yxwh"]
    mean_flat = best_win["mean_flat"]
    boxes = best_win["boxes"]
    line_ys = best_win["line_ys"]

    out_dir = out_root / scroll / seg
    out_dir.mkdir(parents=True, exist_ok=True)

    sl = (slice(y0, y0 + wh), slice(x0, x0 + ww))
    p1_gold_clean = make_calligraphic_gold(ct[sl], callig_hp[sl], valid[sl])
    cv2.imwrite(str(out_dir / "clean_2x2cm_spatial_gold.png"), p1_gold_clean)
    cv2.imwrite(str(out_dir / "clean_2x2cm_spatial_gray.png"), callig_hp[sl])

    md = meta.get("multidepth", {})
    r_2x2cm = float(md.get("max_2x2cm_r_villa_hecate", 0.0))

    render_paleography_dossier(
        scroll, seg, orient, pair_label, mean_flat, r_2x2cm, px_per_mm,
        p1_gold_clean, callig_hp[sl], boxes, line_ys, out_dir
    )

    # Also save a side-by-side overview of the Top 3 2x2cm windows
    win_tiles = []
    for idx_w, we in enumerate(win_evals):
        wy0, wx0, wwh, www = we["yxwh"]
        wsl = (slice(wy0, wy0 + wwh), slice(wx0, wx0 + www))
        w_gold = make_calligraphic_gold(ct[wsl], callig_hp[wsl], valid[wsl])
        for wb in we["boxes"][:15]:
            cv2.rectangle(w_gold, (wb["x"], wb["y"]), (wb["x"] + wb["w"], wb["y"] + wb["h"]), (80, 255, 120), 2)
        wt = cv2.resize(w_gold, (540, 540), interpolation=cv2.INTER_AREA)
        lbl = f"Win #{idx_w+1} (y={wy0},x={wx0}) flat={we['mean_flat']:.2f} letters={len(we['boxes'])}"
        cv2.rectangle(wt, (6, 6), (534, 34), (15, 15, 15), -1)
        cv2.putText(wt, lbl, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 140), 1, cv2.LINE_AA)
        win_tiles.append(wt)
    if win_tiles:
        cv2.imwrite(str(out_dir / "top3_windows_overview.png"), np.hstack(win_tiles))

    # Strip internal numpy template before JSON serialization
    clean_boxes = []
    for b in boxes:
        cb = {k: v for k, v in b.items() if not k.startswith("_")}
        clean_boxes.append(cb)

    char_seq = " ".join([f"{b['greek_char']}({b['greek_ascii']})" for b in clean_boxes[:12]])
    summary = {
        "scroll": scroll,
        "segment": seg,
        "orient": orient,
        "vox_um": vox_um,
        "best_single_depth": best_k,
        "best_depth_pair": pair_label,
        "r_all": md.get("r_villa_hecate", 0.0),
        "r_2x2cm": r_2x2cm,
        "best_2x2cm_yxwh": [int(y0), int(x0), int(wh), int(ww)],
        "mean_sheet_flatness": round(float(mean_flat), 3),
        "num_dual_model_letter_candidates": len(clean_boxes),
        "top_12_greek_hypotheses": char_seq,
        "top_12_letter_boxes": clean_boxes[:12],
        "all_windows_evaluated": [
            {
                "yxwh": list(we["yxwh"]),
                "mean_flat": round(float(we["mean_flat"]), 3),
                "num_letters": len(we["boxes"]),
            }
            for we in win_evals
        ],
        "out_dir": str(out_dir),
    }
    (out_dir / "stage_f_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Stage F1] {scroll}/{seg} ({orient}): flat={mean_flat:.2f} letters={len(clean_boxes)} r_2x2cm={r_2x2cm:.4f} | top={char_seq[:48]}")
    return summary


def main():
    STAGE_F_DIR.mkdir(parents=True, exist_ok=True)
    seg_dirs = sorted([p.parent for p in STAGE_D_DIR.rglob("metrics_stage_d.json")])
    summaries = []
    for sd in seg_dirs:
        s = process_segment(sd, STAGE_F_DIR)
        if s is not None:
            summaries.append(s)
    summaries.sort(
        key=lambda x: (
            x["num_dual_model_letter_candidates"] >= 10 and x["mean_sheet_flatness"] >= 0.75,
            x["num_dual_model_letter_candidates"] >= 10,
            x["r_2x2cm"],
        ),
        reverse=True,
    )
    (STAGE_F_DIR / "leaderboard_stage_f.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(summaries)} segments to {STAGE_F_DIR / 'leaderboard_stage_f.json'}")


if __name__ == "__main__":
    main()
