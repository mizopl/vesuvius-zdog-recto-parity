#!/usr/bin/env python3
"""
Stage E v2: In-Focus Depth Selector, Single-Sheet Flatness Filter, and
Horizontal Text-Line / Greek Letter Candidate Extractor.

Addresses two critical physical effects observed in Stage E v1:
1. Out-of-focus depth bleed-through: Instead of pooling all 5 depths (-8..+8) equally,
   selects ONLY the top best single depth (d_best) and best contiguous 2-depth pair
   (d_a, d_b) based on cross-family correlation r_2x2cm(d).
2. Wrap-fold / layer-jump suppression: Computes a local single-sheet flatness mask from
   ct_mid.png (penalizing inter-wrap air gaps and high-contrast oblique wrap-slice contours)
   so the 2x2 cm (4 cm^2) prize window lands on a flat single papyrus sheet with orthogonal
   fiber weave.
3. Horizontal text-line & >=2x2 mm letter candidate detection: Locates periodic horizontal
   ink lines (~3.2-4.5 mm pitch) and extracts candidate Greek character boxes where both
   Villa and Hecate fire.
"""
import json
from pathlib import Path
import cv2
import numpy as np


BASE = Path(__file__).resolve().parent.parent
STAGE_D_DIR = BASE / "results_stage_d"
STAGE_E2_DIR = BASE / "results_stage_e2"


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
    - Penalizes dark inter-wrap air gaps (0 < ct < 42)
    - Penalizes extreme local CT gradient contours caused by oblique cuts across multiple wraps
    """
    valid = (ct > 0).astype(np.float32)
    ct_f = ct.astype(np.float32)

    # 1. Air-gap / crack fraction in a ~1.5 mm neighborhood
    is_gap = ((ct > 0) & (ct < 45)).astype(np.float32)
    gap_frac = cv2.GaussianBlur(is_gap, (0, 0), sigmaX=35.0, sigmaY=35.0)
    no_gap_score = np.clip(1.0 - gap_frac * 4.5, 0.05, 1.0)

    # 2. Local CT gradient energy (oblique layer-jump contours have huge gradient spikes)
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

    # 3. High-frequency papyrus fiber weave presence (rejects smooth interpolated hole-fills / resin voids)
    ct_blur3 = cv2.GaussianBlur(ct_f, (0, 0), sigmaX=2.5, sigmaY=2.5)
    hf_weave = cv2.GaussianBlur(np.abs(ct_f - ct_blur3), (0, 0), sigmaX=14.0, sigmaY=14.0)
    fiber_presence = np.clip((hf_weave - 2.2) / 3.8, 0.0, 1.0)

    return valid * no_gap_score * flat_grad_score * fiber_presence


def select_infocus_depths(meta: dict, seg_dir: Path, orient: str, valid: np.ndarray):
    """
    Selects the single best depth d_best and the best contiguous 2-depth pair (d_a, d_b)
    from [-8, -4, 0, +4, +8] based on cross-family correlation r_villa_hecate and max_2x2cm_r.
    """
    depths_meta = meta.get("depths", {})
    ordered_keys = ["d-8", "d-4", "d+0", "d+4", "d+8"]
    scores = {}
    maps = {}
    for k in ordered_keys:
        p = seg_dir / f"ens_z_{orient}_{k}.png"
        m = load_gray(p)
        if m is not None:
            maps[k] = m
            dm = depths_meta.get(k, {})
            r_all = float(dm.get("r_villa_hecate", 0.0))
            r_2cm = float(dm.get("max_2x2cm_r_villa_hecate", r_all))
            if r_2cm < 0:
                r_2cm = r_all
            scores[k] = 0.55 * r_2cm + 0.45 * r_all

    if not maps:
        md = load_gray(seg_dir / f"ens_z_{orient}_multidepth.png")
        return "multidepth", "multidepth", md, md

    best_k = max(scores.keys(), key=lambda k: scores[k])
    best_1d = maps[best_k]

    # Best contiguous pair
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
        w1 = max(0.05, scores[k1])
        w2 = max(0.05, scores[k2])
        m1 = maps[k1].astype(np.float32)
        m2 = maps[k2].astype(np.float32)
        pair_blend = ((w1 * m1 + w2 * m2) / (w1 + w2)) * 0.45 + np.maximum(m1, m2) * 0.55
        infocus_2d = np.clip(pair_blend, 0, 255).astype(np.uint8)
        pair_label = f"{k1}+{k2}"
    else:
        infocus_2d = best_1d.copy()
        pair_label = best_k

    return best_k, pair_label, best_1d, infocus_2d


def find_best_flat_2x2cm_window(ink_hp: np.ndarray, flat_w: np.ndarray, valid: np.ndarray, win_px: int):
    """Find the 2x2 cm window that maximizes ink signal on a flat single-sheet surface."""
    H, W = ink_hp.shape
    wh = min(win_px, H)
    ww = min(win_px, W)

    # Weight ink signal by flatness^2 so wrap folds are strongly avoided
    weighted_ink = ink_hp.astype(np.float32) * (flat_w ** 2.0)
    ii_ink = cv2.integral(weighted_ink)
    ii_flat = cv2.integral(flat_w)
    ii_val = cv2.integral(valid.astype(np.uint8))

    step = max(16, win_px // 20)
    best_sc = -1.0
    best_yx = (0, 0)
    best_mean_flat = 0.0

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
            if sc > best_sc:
                best_sc = sc
                best_yx = (y0, x0)
                best_mean_flat = mean_flat

    return best_yx[0], best_yx[1], wh, ww, best_sc, best_mean_flat


def detect_lines_and_letter_boxes(crop_ink: np.ndarray, crop_dn: np.ndarray, crop_hc: np.ndarray,
                                  crop_flat: np.ndarray, vox_um: float):
    """
    Inside a 2x2 cm window:
    1. Locate candidate character components of physical size strictly >= 2.0 mm x 2.0 mm (up to 4.5 mm)
       where both Villa (dnative) and Hecate show strong stroke signal (p94) and sheet flatness is high.
    2. Split horizontally merged multi-letter word blobs at vertical projection valleys (~2.2 mm pitch).
    3. Group character candidates into horizontal text lines (~2.8-4.8 mm pitch).
    """
    H, W = crop_ink.shape
    px_per_mm = 1000.0 / vox_um
    min_char_px = int(np.ceil(2.02 * px_per_mm))   # strictly >= 2.02 mm (~256 px)
    max_char_px = int(round(4.50 * px_per_mm))     # <= 4.50 mm (~570 px)
    nom_char_w = int(round(2.35 * px_per_mm))      # ~2.35 mm (~297 px)
    nom_char_h = int(round(2.45 * px_per_mm))      # ~2.45 mm (~310 px)

    # Clean ink map gated by flatness AND dual-model support
    dual_geom = np.sqrt(crop_dn.astype(np.float32) * crop_hc.astype(np.float32))
    clean = ((0.55 * crop_ink.astype(np.float32) + 0.45 * dual_geom) * np.clip(crop_flat, 0.0, 1.0)).astype(np.uint8)
    v_vals = clean[crop_flat > 0.45]
    if v_vals.size < 100:
        return [], []

    # Horizontal row profile to find text lines first (88th percentile across row so half-line columns aren't diluted)
    row_prof = np.percentile(clean.astype(np.float32), 88.0, axis=1)
    k_1d = max(5, int(round(1.1 * px_per_mm)) | 1)
    row_smooth = cv2.GaussianBlur(row_prof.reshape(-1, 1), (1, k_1d), 0).ravel()
    min_line_dist = int(round(2.75 * px_per_mm))  # >= 2.75 mm between text lines

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

    # Small vertical-favoring kernel so adjacent letters along a line do NOT merge horizontally
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_closed, connectivity=8)
    raw_rects = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w < int(round(1.0 * px_per_mm)) and h < int(round(1.0 * px_per_mm)):
            continue
        # If component spans multiple lines vertically, split into horizontal sub-bands
        n_v = max(1, int(round(h / (2.85 * px_per_mm)))) if h > max_char_px else 1
        sub_h = h // n_v
        for iv in range(n_v):
            sy = y + iv * sub_h
            sh = sub_h if iv < n_v - 1 else (y + h - sy)
            # If sub-band spans multiple letters horizontally, split at vertical projection valleys
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
                    if c_hi > c_lo:
                        cut_x = c_lo + int(np.argmin(v_prof[c_lo:c_hi]))
                    else:
                        cut_x = nom_c
                    cuts.append(cut_x)
                cuts.append(w)
                for ih in range(len(cuts) - 1):
                    sx = x + cuts[ih]
                    sw = cuts[ih + 1] - cuts[ih]
                    raw_rects.append((sx, sy, sw, sh))
            else:
                raw_rects.append((x, sy, w, sh))

    # Also sample candidate character peaks along each detected horizontal text line
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
                sx = int(x_idx) - nom_char_w // 2
                raw_rects.append((sx, y0, nom_char_w, nom_char_h))
                if len(picked_xs) >= 8:
                    break

    # Evaluate and deduplicate (NMS) candidate character boxes (strictly >= 2.0 mm x 2.0 mm)
    cand_boxes = []
    for (x, y, w, h) in raw_rects:
        # Pad if needed to reach strictly >= 2.02 mm x 2.02 mm character cell while staying inside crop
        if w < min_char_px:
            pad_x = (min_char_px - w + 1) // 2
            x = max(0, min(W - min_char_px, x - pad_x))
            w = min(min_char_px, W - x)
        if h < min_char_px:
            pad_y = (min_char_px - h + 1) // 2
            y = max(0, min(H - min_char_px, y - pad_y))
            h = min(min_char_px, H - y)
        if w > max_char_px or h > max_char_px:
            continue
        if w < min_char_px or h < min_char_px:
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
        # Use 94th percentile (~5,500 core stroke pixels inside 92,000-px box) vs 30th percentile background
        dn_val = float(np.percentile(patch_dn, 94.0))
        hc_val = float(np.percentile(patch_hc, 94.0))
        ink_p94 = float(np.percentile(patch_clean, 94.0))
        ink_p30 = float(np.percentile(patch_clean, 30.0))
        local_contrast = ink_p94 - ink_p30
        # Require strong dual-family ink (Villa + Hecate), flat sheet, and stroke-vs-background contrast
        if fl_val < 0.52 or dn_val < 68.0 or hc_val < 68.0 or local_contrast < 28.0:
            continue

        score = (0.45 * dn_val + 0.45 * hc_val + 0.30 * local_contrast) * (fl_val ** 0.75)
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        cand_boxes.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "w_mm": round(w / px_per_mm, 2),
            "h_mm": round(h / px_per_mm, 2),
            "cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "dn_p85": round(dn_val, 1),
            "hc_p85": round(hc_val, 1),
            "contrast": round(local_contrast, 1),
            "flatness": round(fl_val, 2),
            "score": round(score, 1),
        })

    cand_boxes.sort(key=lambda b: b["score"], reverse=True)
    boxes = []
    min_center_dist = 1.92 * px_per_mm
    for b in cand_boxes:
        overlap = False
        for kb in boxes:
            dist = np.hypot(b["cx"] - kb["cx"], b["cy"] - kb["cy"])
            if dist < min_center_dist:
                overlap = True
                break
        if not overlap:
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
    best_k, pair_label, best_1d, infocus_2d = select_infocus_depths(meta, seg_dir, orient, valid)

    # High-pass normalize the in-focus maps + Villa/Hecate
    dn_hp = highpass_norm(dn_md, valid, sigma=42.0, p_low=52.0, p_high=99.6)
    hc_hp = highpass_norm(hc_md, valid, sigma=50.0, p_low=50.0, p_high=99.6)
    best1_hp = highpass_norm(best_1d, valid, sigma=42.0, p_low=55.0, p_high=99.6)
    infocus2_hp = highpass_norm(infocus_2d, valid, sigma=42.0, p_low=55.0, p_high=99.6)

    # In-Focus Calligraphic Blend: 50% in-focus 2-depth geometric ensemble + 50% additive (Villa_hp + Hecate_hp)
    callig_f = 0.50 * infocus2_hp.astype(np.float32) + 0.26 * dn_hp.astype(np.float32) + 0.24 * hc_hp.astype(np.float32)
    callig_hp = highpass_norm(callig_f, valid, sigma=42.0, p_low=50.0, p_high=99.6)

    y0, x0, wh, ww, win_sc, mean_flat = find_best_flat_2x2cm_window(callig_hp, flat_w, valid, win_2cm_px)
    sl = (slice(y0, y0 + wh), slice(x0, x0 + ww))

    crop_ct = ct[sl]
    crop_valid = valid[sl]
    crop_flat = flat_w[sl]
    crop_callig = callig_hp[sl]
    crop_best1 = best1_hp[sl]
    crop_inf2 = infocus2_hp[sl]
    crop_dn = dn_hp[sl]
    crop_hc = hc_hp[sl]

    boxes, line_ys = detect_lines_and_letter_boxes(crop_callig, crop_dn, crop_hc, crop_flat, vox_um)

    out_dir = out_root / scroll / seg
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep clean gold image un-annotated for clean crops and galleries
    p1_gold_clean = make_calligraphic_gold(crop_ct, crop_callig, crop_valid)
    cv2.imwrite(str(out_dir / "clean_2x2cm_infocus_gold.png"), p1_gold_clean)
    cv2.imwrite(str(out_dir / "clean_2x2cm_infocus_gray.png"), crop_callig)

    p1_gold = p1_gold_clean.copy()
    p2_best1 = cv2.cvtColor(crop_best1, cv2.COLOR_GRAY2BGR)
    p3_callig = cv2.cvtColor(crop_callig, cv2.COLOR_GRAY2BGR)
    p4_annot = p1_gold_clean.copy()

    # Draw horizontal text-line guides & candidate letter boxes on Panel 4
    for ly in line_ys:
        cv2.line(p4_annot, (0, ly), (ww - 1, ly), (255, 200, 80), 1, cv2.LINE_AA)

    for idx_b, b in enumerate(boxes[:20]):
        bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
        color = (80, 255, 120) if idx_b < 12 else (120, 220, 255)
        cv2.rectangle(p4_annot, (bx, by), (bx + bw_b, by + bh_b), color, 2)
        cv2.putText(p4_annot, f"#{idx_b+1} ({b['w_mm']:.1f}x{b['h_mm']:.1f}mm)",
                    (bx, max(16, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # Draw 2x2mm reference box on all 4 panels
    step_2mm = int(round(2.0 * px_per_mm))
    for p_im, title in [
        (p1_gold, f"1. {scroll}/{seg} In-Focus Gold (2x2cm, flat={mean_flat:.2f})"),
        (p2_best1, f"2. Single Sharpest Depth: {best_k}"),
        (p3_callig, f"3. In-Focus Calligraphic ({pair_label} + Villa/Hecate)"),
        (p4_annot, f"4. {len(boxes)} Dual-Model Letter Candidates (>=2.0mm)"),
    ]:
        cv2.rectangle(p_im, (ww - 24 - step_2mm, 24), (ww - 24, 24 + step_2mm), (0, 235, 255), 2)
        cv2.putText(p_im, "2x2mm", (ww - 28 - step_2mm, 42 + step_2mm), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 235, 255), 1, cv2.LINE_AA)
        cv2.rectangle(p_im, (8, 8), (min(ww - 8, 18 + len(title) * 9), 34), (15, 15, 15), -1)
        cv2.putText(p_im, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 245, 255), 1, cv2.LINE_AA)

    tile_sz = 820
    def rsz(im):
        return cv2.resize(im, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)

    quad = np.vstack([
        np.hstack([rsz(p1_gold), rsz(p3_callig)]),
        np.hstack([rsz(p2_best1), rsz(p4_annot)]),
    ])
    cv2.imwrite(str(out_dir / "infocus_quad_2x2cm.png"), quad)
    cv2.imwrite(str(out_dir / "annot_2x2cm_letters.png"), p4_annot)

    # Also build a zoomed 5-row strip of all 5 horizontal text lines (each strip 4.5 mm tall x 20 mm wide)
    strip_h = int(round(4.5 * px_per_mm))
    strips = []
    for i_l, ly in enumerate(line_ys):
        sy0 = max(0, min(wh - strip_h, ly - strip_h // 2))
        sy1 = sy0 + strip_h
        s_gold = p1_gold_clean[sy0:sy1, :].copy()
        s_gray = cv2.cvtColor(crop_callig[sy0:sy1, :], cv2.COLOR_GRAY2BGR)
        # Draw 2 mm tick marks along top of strip
        for x_tick in range(0, ww, step_2mm):
            cv2.line(s_gold, (x_tick, 0), (x_tick, 12), (0, 235, 255), 2)
            cv2.line(s_gray, (x_tick, 0), (x_tick, 12), (0, 235, 255), 2)
        cv2.putText(s_gold, f"Line {i_l+1} (y={ly}px, 4.5x20mm, ticks=2mm)", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 235, 255), 1, cv2.LINE_AA)
        strips.append(np.vstack([s_gold, s_gray]))

    if strips:
        line_zoom_img = np.vstack(strips)
        cv2.imwrite(str(out_dir / "line_strips_zoom.png"), line_zoom_img)

    # Build a 3x4 zoomed gallery of the top 12 individual character candidates (CT+Gold | In-Focus Gray)
    if boxes:
        cards = []
        cell_h, cell_w = 200, 400
        for idx_b in range(min(12, len(boxes))):
            b = boxes[idx_b]
            bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
            # Include a small 0.35 mm context margin around the character box
            pad = int(round(0.35 * px_per_mm))
            x0_c = max(0, bx - pad)
            y0_c = max(0, by - pad)
            x1_c = min(ww, bx + bw_b + pad)
            y1_c = min(wh, by + bh_b + pad)
            g_crop = cv2.resize(p1_gold_clean[y0_c:y1_c, x0_c:x1_c], (cell_h, cell_h), interpolation=cv2.INTER_CUBIC)
            k_crop = cv2.cvtColor(
                cv2.resize(crop_callig[y0_c:y1_c, x0_c:x1_c], (cell_h, cell_h), interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_GRAY2BGR,
            )
            card = np.hstack([g_crop, k_crop])
            cv2.rectangle(card, (0, 0), (cell_w - 1, 26), (18, 18, 18), -1)
            lbl = f"#{idx_b+1} {b['w_mm']:.1f}x{b['h_mm']:.1f}mm V={b['dn_p85']:.0f} H={b['hc_p85']:.0f} fl={b['flatness']:.2f}"
            cv2.putText(card, lbl, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 140), 1, cv2.LINE_AA)
            cv2.rectangle(card, (0, 0), (cell_w - 1, cell_h - 1), (70, 70, 70), 1)
            cards.append(card)
        while len(cards) % 4 != 0:
            cards.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        rows = [np.hstack(cards[r:r+4]) for r in range(0, len(cards), 4)]
        cv2.imwrite(str(out_dir / "letter_gallery_top12.png"), np.vstack(rows))

    md = meta.get("multidepth", {})
    summary = {
        "scroll": scroll,
        "segment": seg,
        "orient": orient,
        "vox_um": vox_um,
        "best_single_depth": best_k,
        "best_depth_pair": pair_label,
        "r_all": md.get("r_villa_hecate", 0.0),
        "r_2x2cm": md.get("max_2x2cm_r_villa_hecate", 0.0),
        "flat_2x2cm_yxwh": [int(y0), int(x0), int(wh), int(ww)],
        "mean_sheet_flatness": round(float(mean_flat), 3),
        "num_dual_model_letter_candidates": len(boxes),
        "top_12_letter_boxes": boxes[:12],
        "out_dir": str(out_dir),
    }
    (out_dir / "stage_e2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Stage E2] {scroll}/{seg} ({orient}): best_d={best_k} ({pair_label}) flat={mean_flat:.2f} letters={len(boxes)} r_2x2cm={summary['r_2x2cm']:.4f}")
    return summary


def main():
    STAGE_E2_DIR.mkdir(parents=True, exist_ok=True)
    seg_dirs = sorted([p.parent for p in STAGE_D_DIR.rglob("metrics_stage_d.json")])
    summaries = []
    for sd in seg_dirs:
        s = process_segment(sd, STAGE_E2_DIR)
        if s is not None:
            summaries.append(s)
    summaries.sort(key=lambda x: (x["num_dual_model_letter_candidates"] >= 10, x["r_2x2cm"]), reverse=True)
    (STAGE_E2_DIR / "leaderboard_stage_e2.json").write_text(json.dumps(summaries, indent=2))
    print(f"Saved {len(summaries)} segments to {STAGE_E2_DIR / 'leaderboard_stage_e2.json'}")


if __name__ == "__main__":
    main()
