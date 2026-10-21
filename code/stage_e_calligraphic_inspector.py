#!/usr/bin/env python3
"""
Stage E Calligraphic & Multi-Depth Inspector for Unread Herculaneum Scrolls.
Produces high-clarity calligraphic inspection panels and 2x2 cm (4 cm^2) zoomed
views with 2 mm scale grids for visual verification of Greek letters.

Key enhancements over raw Stage D outputs:
1. Local high-pass background subtraction (sigma = 45 px ~ 3.5 mm) on Villa dnative
   to remove the ~80/255 baseline haze and reveal fine letter crossbars.
2. Soft additive consensus (0.55 * Villa_hp + 0.45 * Hecate_hp) alongside geometric
   consensus (ens_z), preventing partial stroke dropout when one model misses part of a letter.
3. Multi-Depth False-Color RGB Composite:
   - Red channel: shallow layers (d = -8, -4 vox)
   - Green channel: nominal surface (d = 0 vox)
   - Blue channel: deeper layers (d = +4, +8 vox)
   True ink lines on a slightly undulating papyrus surface form continuous white/yellow/cyan
   trajectories along the horizontal papyrus fibers.
4. Soft-alpha gold overlay on CT with 2 mm x 2 mm grid ticks (Prize criterion: >= 2x2 mm letters
   inside a 2x2 cm = 4 cm^2 window).
"""
import json
from pathlib import Path
import cv2
import numpy as np


BASE = Path(__file__).resolve().parent.parent
STAGE_D_DIR = BASE / "results_stage_d"
STAGE_E_DIR = BASE / "results_stage_e"


def load_gray(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def highpass_stretch(img: np.ndarray, valid: np.ndarray, sigma: float = 45.0,
                     p_low: float = 55.0, p_high: float = 99.6) -> np.ndarray:
    """Subtract slow background drift (sigma ~ 3.5 mm) and stretch [p_low, p_high] to [0, 255]."""
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hp = np.maximum(0.0, f - bg * 0.85)
    vals = hp[valid]
    if vals.size < 100:
        return np.zeros_like(img)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo + 1e-4:
        return np.zeros_like(img)
    out = np.clip((hp - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = 0.0
    return (out * 255.0).astype(np.uint8)


def find_top_k_windows(score_map: np.ndarray, valid: np.ndarray, win_px: int, k: int = 3) -> list[tuple[int, int, float]]:
    """Find top-k non-overlapping (or minimally overlapping) 2x2 cm windows."""
    H, W = score_map.shape
    wh = min(win_px, H)
    ww = min(win_px, W)
    s_f = (score_map.astype(np.float32) * valid.astype(np.float32))
    ii = cv2.integral(s_f)
    iv = cv2.integral(valid.astype(np.uint8))

    step = max(16, win_px // 16)
    candidates = []
    for y0 in range(0, H - wh + 1, step):
        y1 = y0 + wh
        for x0 in range(0, W - ww + 1, step):
            x1 = x0 + ww
            v_cnt = int(iv[y1, x1] - iv[y0, x1] - iv[y1, x0] + iv[y0, x0])
            if v_cnt < 0.45 * wh * ww:
                continue
            s_sum = float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
            candidates.append((s_sum / max(1, v_cnt), y0, x0))

    candidates.sort(key=lambda t: t[0], reverse=True)
    picked = []
    for sc, y0, x0 in candidates:
        # Check overlap with already picked windows
        too_close = False
        for _, py, px in picked:
            if abs(y0 - py) < wh * 0.45 and abs(x0 - px) < ww * 0.45:
                too_close = True
                break
        if not too_close:
            picked.append((sc, y0, x0))
            if len(picked) >= k:
                break
    if not picked:
        picked.append((0.0, 0, 0))
    return [(y0, x0, sc) for sc, y0, x0 in picked]


def draw_2mm_grid_and_scale(canvas_bgr: np.ndarray, vox_um: float, title: str = "") -> np.ndarray:
    """Draw subtle 2 mm grid ticks along borders and a 2 mm x 2 mm reference box."""
    out = canvas_bgr.copy()
    H, W = out.shape[:2]
    px_per_mm = 1000.0 / vox_um
    step_2mm = int(round(2.0 * px_per_mm))

    # Draw subtle border tick marks every 2 mm
    for y in range(0, H, step_2mm):
        cv2.line(out, (0, y), (14, y), (140, 220, 255), 1, cv2.LINE_AA)
        cv2.line(out, (W - 15, y), (W - 1, y), (140, 220, 255), 1, cv2.LINE_AA)
        # Very faint horizontal guide line every 2 mm to aid reading horizontal lines of text
        overlay = out.copy()
        cv2.line(overlay, (0, y), (W - 1, y), (80, 120, 140), 1)
        out = cv2.addWeighted(overlay, 0.14, out, 0.86, 0)

    for x in range(0, W, step_2mm):
        cv2.line(out, (x, 0), (x, 14), (140, 220, 255), 1, cv2.LINE_AA)
        cv2.line(out, (x, H - 15), (x, H - 1), (140, 220, 255), 1, cv2.LINE_AA)

    # Draw a 2 mm x 2 mm reference box in top-right corner
    bx1 = W - 24
    bx0 = bx1 - step_2mm
    by0 = 24
    by1 = by0 + step_2mm
    if bx0 > 0 and by1 < H:
        cv2.rectangle(out, (bx0, by0), (bx1, by1), (0, 230, 255), 2)
        cv2.putText(out, "2x2mm", (bx0 - 8, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

    if title:
        cv2.rectangle(out, (8, 8), (min(W - 8, 18 + len(title) * 9), 34), (15, 15, 15), -1)
        cv2.putText(out, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 245, 255), 1, cv2.LINE_AA)
    return out


def make_soft_gold_overlay(ct: np.ndarray, ink_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Create a calligraphic gold overlay on CT that preserves softer letter crossbars."""
    ct_f = ct.astype(np.float32)
    vals = ct_f[valid]
    if vals.size > 100:
        lo, hi = np.percentile(vals, [2.0, 98.5])
        ct_norm = np.clip((ct_f - lo) / max(1.0, hi - lo), 0.0, 1.0)
    else:
        ct_norm = ct_f / 255.0
    base = (ct_norm * 165.0).astype(np.uint8)
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    ink_f = ink_map.astype(np.float32)
    ivals = ink_f[valid]
    if ivals.size > 100:
        p75 = float(np.percentile(ivals, 75.0))
        p998 = float(np.percentile(ivals, 99.8))
    else:
        p75, p998 = 64.0, 255.0
    alpha = np.clip((ink_f - p75) / max(1.0, p998 - p75), 0.0, 1.0) ** 1.15
    alpha[~valid] = 0.0

    # Warm gold-amber ink color (BGR: 65, 215, 255)
    gold = np.zeros_like(bgr, dtype=np.float32)
    gold[:, :, 0] = 65.0
    gold[:, :, 1] = 215.0
    gold[:, :, 2] = 255.0

    out = bgr.astype(np.float32) * (1.0 - 0.82 * alpha[:, :, None]) + gold * (0.82 * alpha[:, :, None])
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

    ct = load_gray(seg_dir / "ct_mid.png")
    dn_md = load_gray(seg_dir / f"dnative_{orient}_multidepth.png")
    hc_md = load_gray(seg_dir / f"hecate_{orient}_multidepth.png")
    ens_md = load_gray(seg_dir / f"ens_z_{orient}_multidepth.png")

    if ct is None or dn_md is None or hc_md is None or ens_md is None:
        return None

    valid = ct > 0
    # High-pass background subtraction on Villa & Hecate
    dn_hp = highpass_stretch(dn_md, valid, sigma=45.0, p_low=52.0, p_high=99.6)
    hc_hp = highpass_stretch(hc_md, valid, sigma=55.0, p_low=50.0, p_high=99.6)
    ens_hp = highpass_stretch(ens_md, valid, sigma=45.0, p_low=55.0, p_high=99.6)

    # Soft additive calligraphic consensus: preserves continuous strokes from both models
    add_cons_f = 0.52 * dn_hp.astype(np.float32) + 0.48 * hc_hp.astype(np.float32)
    add_cons = highpass_stretch(add_cons_f, valid, sigma=45.0, p_low=50.0, p_high=99.6)

    # Load depth stack for RGB depth composite
    d_m8 = load_gray(seg_dir / f"ens_z_{orient}_d-8.png")
    d_m4 = load_gray(seg_dir / f"ens_z_{orient}_d-4.png")
    d_p0 = load_gray(seg_dir / f"ens_z_{orient}_d+0.png")
    d_p4 = load_gray(seg_dir / f"ens_z_{orient}_d+4.png")
    d_p8 = load_gray(seg_dir / f"ens_z_{orient}_d+8.png")

    if all(x is not None for x in [d_m8, d_m4, d_p0, d_p4, d_p8]):
        r_ch = highpass_stretch(np.maximum(d_m8, d_m4), valid, sigma=45.0, p_low=60.0, p_high=99.6)
        g_ch = highpass_stretch(d_p0, valid, sigma=45.0, p_low=60.0, p_high=99.6)
        b_ch = highpass_stretch(np.maximum(d_p4, d_p8), valid, sigma=45.0, p_low=60.0, p_high=99.6)
        rgb_depth = np.stack([b_ch, g_ch, r_ch], axis=-1)  # BGR for OpenCV
    else:
        rgb_depth = cv2.cvtColor(ens_hp, cv2.COLOR_GRAY2BGR)

    out_dir = out_root / scroll / seg
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full-sheet enhanced maps
    soft_gold_full = make_soft_gold_overlay(ct, add_cons, valid)
    cv2.imwrite(str(out_dir / "full_soft_gold.png"), soft_gold_full)
    cv2.imwrite(str(out_dir / "full_additive_consensus.png"), add_cons)
    cv2.imwrite(str(out_dir / "full_dnative_hp.png"), dn_hp)
    cv2.imwrite(str(out_dir / "full_hecate_hp.png"), hc_hp)
    cv2.imwrite(str(out_dir / "full_rgb_depth.png"), rgb_depth)

    # Find top 2 non-overlapping 2x2 cm windows using combined score (ens_hp + add_cons)
    comb_score = 0.5 * ens_hp.astype(np.float32) + 0.5 * add_cons.astype(np.float32)
    windows = find_top_k_windows(comb_score, valid, win_2cm_px, k=2)

    md_metrics = meta.get("multidepth", {})
    box_corr = md_metrics.get("max_2x2cm_box_yxwh", None)
    if box_corr and len(box_corr) == 4:
        cy0, cx0, _, _ = box_corr
        # Prepend correlation-optimal 2x2cm box if distinct
        windows.insert(0, (int(cy0), int(cx0), float(md_metrics.get("max_2x2cm_r_villa_hecate", 0.0))))

    H, W = ct.shape
    wh = min(win_2cm_px, H)
    ww = min(win_2cm_px, W)

    win_summaries = []
    for idx, (y0, x0, sc) in enumerate(windows[:2]):
        y0 = max(0, min(H - wh, int(y0)))
        x0 = max(0, min(W - ww, int(x0)))
        y1, x1 = y0 + wh, x0 + ww
        sl = (slice(y0, y1), slice(x0, x1))

        p_gold = draw_2mm_grid_and_scale(soft_gold_full[sl], vox_um, f"{scroll}/{seg} W{idx+1} SoftGold (2x2cm)")
        p_dn = draw_2mm_grid_and_scale(cv2.cvtColor(dn_hp[sl], cv2.COLOR_GRAY2BGR), vox_um, "1. Villa dnative (HighPass)")
        p_hc = draw_2mm_grid_and_scale(cv2.cvtColor(hc_hp[sl], cv2.COLOR_GRAY2BGR), vox_um, "2. Hecate (HighPass)")
        p_add = draw_2mm_grid_and_scale(cv2.cvtColor(add_cons[sl], cv2.COLOR_GRAY2BGR), vox_um, "3. Additive Consensus (Villa+Hecate)")
        p_ens = draw_2mm_grid_and_scale(cv2.cvtColor(ens_hp[sl], cv2.COLOR_GRAY2BGR), vox_um, "4. Geometric Consensus (ens_z HP)")
        p_rgb = draw_2mm_grid_and_scale(rgb_depth[sl], vox_um, "5. Depth RGB (R:-8/-4, G:0, B:+4/+8)")

        tile_sz = 780
        def rsz(im):
            return cv2.resize(im, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)

        row1 = np.hstack([rsz(p_gold), rsz(p_add), rsz(p_ens)])
        row2 = np.hstack([rsz(p_dn), rsz(p_hc), rsz(p_rgb)])
        panel6 = np.vstack([row1, row2])
        cv2.imwrite(str(out_dir / f"calligraphic_2x2cm_w{idx+1}.png"), panel6)
        cv2.imwrite(str(out_dir / f"crop_2x2cm_w{idx+1}_soft_gold.png"), p_gold)
        cv2.imwrite(str(out_dir / f"crop_2x2cm_w{idx+1}_additive.png"), p_add)

        win_summaries.append({"win_idx": idx + 1, "y0": int(y0), "x0": int(x0), "h": int(wh), "w": int(ww), "score": round(float(sc), 4)})

    summary = {
        "scroll": scroll,
        "seg": seg,
        "orient": orient,
        "vox_um": vox_um,
        "r_all": md_metrics.get("r_villa_hecate", 0.0),
        "r_2x2cm": md_metrics.get("max_2x2cm_r_villa_hecate", 0.0),
        "ens4_p99.5": md_metrics.get("ens4_p995", 0.0),
        "hc_p99.5": md_metrics.get("hecate_p995", 0.0),
        "windows": win_summaries,
        "out_dir": str(out_dir),
    }
    (out_dir / "stage_e_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Stage E] {scroll}/{seg} ({orient}): r_all={summary['r_all']:.4f} r_2x2cm={summary['r_2x2cm']:.4f} -> {out_dir}")
    return summary


def main():
    STAGE_E_DIR.mkdir(parents=True, exist_ok=True)
    seg_dirs = sorted([p.parent for p in STAGE_D_DIR.rglob("metrics_stage_d.json")])
    print(f"Found {len(seg_dirs)} unpacked Stage D segments.")
    all_summaries = []
    for sd in seg_dirs:
        s = process_segment(sd, STAGE_E_DIR)
        if s is not None:
            all_summaries.append(s)
    all_summaries.sort(key=lambda x: x["r_2x2cm"], reverse=True)
    (STAGE_E_DIR / "leaderboard_stage_e.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"Wrote {len(all_summaries)} segments to {STAGE_E_DIR / 'leaderboard_stage_e.json'}")


if __name__ == "__main__":
    main()
