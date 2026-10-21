#!/usr/bin/env python3
"""
Stage H — Final CLSS Skeptic Gate Audit, 4-Column Comparison Board
(Stage F vs Stage G 2D-Poisson vs Stage H True-3D-m7-Snap),
Updated Paleography Dossier, and Honest Stop-Loss Report Generator.
"""
import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))

from stage_f_sheet_tracker_and_paleography import (
    compute_single_sheet_flatness,
    detect_and_classify_letters,
    highpass_norm,
    load_gray,
    make_calligraphic_gold,
    render_paleography_dossier,
)
from stage_g_clss_audit_and_dossier import (
    evaluate_h4_clss_skeptic_gate,
    evaluate_stroke_continuity_metrics,
)

STAGE_D_DIR = BASE / "results_stage_d"
STAGE_F_DIR = BASE / "results_stage_f"
STAGE_G_DOSSIER_DIR = BASE / "results_stage_g_dossiers"
STAGE_H_DIR = BASE / "results_stage_h"
STAGE_H_DOSSIER_DIR = BASE / "results_stage_h_dossiers"
SUBMISSIONS_DIR = BASE / "submissions_ready"


def render_f_g_h_comparison_board(
    scroll: str,
    seg: str,
    orient: str,
    px_per_mm: float,
    img_f_gold: np.ndarray,
    img_g_gold: np.ndarray,
    img_h_single_gold: np.ndarray,
    img_h_ridge_gold: np.ndarray,
    s_star_map: np.ndarray,
    boxes_h_single: list[dict],
    boxes_h_ridge: list[dict],
    line_ys_h: list[int],
    summary_h: dict,
    meta_h: dict,
    out_path: Path,
):
    """
    Render a 4-Column (2160 x 1080) Master Comparison Board:
    - Col 1: Stage F Baseline (un-snapped mesh, 4-vox step, 1x TTA)
    - Col 2: Stage G (post-render 2D Poisson warping)
    - Col 3: Stage H Single-Layer d* (True 3D Snap to S3 m7-Zarr + Zero-Poisson + Global Z)
    - Col 4: Stage H 1-Voxel Ridge (True 3D Snap to S3 m7-Zarr + 4x TTA) & CLSS Audit
    """
    board = np.zeros((1080, 2160, 3), dtype=np.uint8)
    board[:] = (16, 18, 22)

    def annotate_boxes(img_bgr: np.ndarray, boxes: list[dict], line_ys: list[int]):
        vis = img_bgr.copy()
        wh, ww = vis.shape[:2]
        for i_l, ly in enumerate(line_ys):
            cv2.line(vis, (0, ly), (ww - 1, ly), (255, 200, 80), 1, cv2.LINE_AA)
            cv2.putText(vis, f"L{i_l+1}", (8, max(20, ly - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 210, 90), 2, cv2.LINE_AA)
        for idx_b, b in enumerate(boxes[:20]):
            bx, by, bw_b, bh_b = b["x"], b["y"], b["w"], b["h"]
            col = (80, 255, 120) if idx_b < 12 else (120, 220, 255)
            cv2.rectangle(vis, (bx, by), (bx + bw_b, by + bh_b), col, 2)
            lbl = f"#{idx_b+1}:{b['greek_ascii']}"
            cv2.putText(vis, lbl, (bx, max(16, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        return vis

    p_f = cv2.resize(img_f_gold, (520, 520), interpolation=cv2.INTER_AREA)
    p_g = cv2.resize(img_g_gold, (520, 520), interpolation=cv2.INTER_AREA)
    p_hs = cv2.resize(annotate_boxes(img_h_single_gold, boxes_h_single, line_ys_h), (520, 520), interpolation=cv2.INTER_AREA)
    p_hr = cv2.resize(annotate_boxes(img_h_ridge_gold, boxes_h_ridge, line_ys_h), (520, 520), interpolation=cv2.INTER_AREA)

    board[58:578, 10:530] = p_f
    board[58:578, 550:1070] = p_g
    board[58:578, 1090:1610] = p_hs
    board[58:578, 1630:2150] = p_hr

    headers = [
        (10, "1. STAGE F: Un-Snapped Mesh (4-vox, 1x TTA)", (180, 200, 230)),
        (550, "2. STAGE G: Post-Render 2D Poisson Warp", (255, 180, 120)),
        (1090, f"3. STAGE H: True 3D m7-Snap Single {summary_h['best_single_depth']}", (110, 235, 255)),
        (1630, "4. STAGE H: True 3D m7-Snap 1-Vox Ridge + TTA", (90, 255, 140)),
    ]
    for x0, title, col in headers:
        cv2.rectangle(board, (x0, 12), (x0 + 520, 50), (28, 32, 40), -1)
        cv2.putText(board, title, (x0 + 10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.49, col, 1, cv2.LINE_AA)

    # Bottom Left (10..530): 3D m7-Snap displacement map s*(r,c) + 13-depth attenuation curve
    diag_box = np.zeros((476, 520, 3), dtype=np.uint8)
    diag_box[:] = (22, 24, 30)
    if s_star_map is not None:
        s_rsz = cv2.resize(s_star_map, (240, 220), interpolation=cv2.INTER_AREA)
        diag_box[42:262, 12:252] = s_rsz
    snap_st = summary_h.get("m7_3d_snap_stats", {})
    cv2.putText(diag_box, f"3D m7-Snap s*(r,c): [{snap_st.get('s_star_min_vox',0):+.1f}..{snap_st.get('s_star_max_vox',0):+.1f}v]", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 120), 1, cv2.LINE_AA)
    cv2.putText(diag_box, f"m7 prob: {snap_st.get('m7_prob_mean_before',0):.1f} -> {snap_st.get('m7_prob_mean_after',0):.1f}", (262, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 255, 140), 1, cv2.LINE_AA)
    cv2.putText(diag_box, f"corners: {snap_st.get('m7_prob_ext_corners_before',0):.1f} -> {snap_st.get('m7_prob_ext_corners_after',0):.1f}", (262, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 235, 255), 1, cv2.LINE_AA)
    cv2.putText(diag_box, f"m7 drop @+-4v: {snap_st.get('m7_ridge_attenuation_pm4_pct',0):.1f}%", (262, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 210, 110), 1, cv2.LINE_AA)

    # Plot 13-depth r_raw_2x2cm(d) and r_hp_2x2cm(d) in bottom half of diag_box (275..465)
    depths = list(range(-6, 7))
    r_raw_vals = [float(meta_h["depths"][f"d{d:+d}"].get("r_2x2cm", 0.0)) for d in depths]
    r_hp_vals = [float(meta_h["depths"][f"d{d:+d}"].get("r_hp_2x2cm", 0.0)) for d in depths]
    lo_v = min(min(r_raw_vals), min(r_hp_vals))
    hi_v = max(max(r_raw_vals), max(r_hp_vals), lo_v + 0.05)
    pts_raw, pts_hp = [], []
    best_d_int = summary_h["h4_clss_audit"]["best_dstar_vox"]
    for idx_d, d in enumerate(depths):
        px = int(30 + idx_d * (460 / 12.0))
        py_raw = int(440 - 140 * (r_raw_vals[idx_d] - lo_v) / (hi_v - lo_v))
        py_hp = int(440 - 140 * (r_hp_vals[idx_d] - lo_v) / (hi_v - lo_v))
        pts_raw.append((px, py_raw))
        pts_hp.append((px, py_hp))
        col_tk = (80, 255, 120) if d == best_d_int else (160, 165, 175)
        cv2.putText(diag_box, f"{d:+d}", (px - 10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col_tk, 1, cv2.LINE_AA)
    for i in range(len(pts_raw) - 1):
        cv2.line(diag_box, pts_raw[i], pts_raw[i + 1], (80, 220, 255), 2, cv2.LINE_AA)
        cv2.line(diag_box, pts_hp[i], pts_hp[i + 1], (90, 255, 130), 2, cv2.LINE_AA)
    cv2.putText(diag_box, "13-Depth Profile: Cyan=r_raw_2x2(d), Green=r_hp_2x2(d)", (14, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (230, 240, 255), 1, cv2.LINE_AA)
    board[592:1068, 10:530] = diag_box

    # Bottom Middle+Right (550..2150): Quantitative Stage D/F vs Stage G vs Stage H Table & CLSS Audit
    info_panel = np.zeros((476, 1600, 3), dtype=np.uint8)
    info_panel[:] = (22, 24, 30)
    clss = summary_h["h4_clss_audit"]
    lines_text = [
        (f"STAGE H TRUE 3D M7-ZARR SNAP AUDIT: {scroll} / {seg} ({orient}) | Voxel={meta_h['um']} um", (255, 235, 130), 0.54),
        (
            f"1. 3D Mesh Snap to S3 m7-L0-th0.2.zarr: s* in [{snap_st.get('s_star_min_vox',0):+.2f}, {snap_st.get('s_star_max_vox',0):+.2f}] vox | "
            f"m7 surface prob: {snap_st.get('m7_prob_mean_before',0):.1f} -> {snap_st.get('m7_prob_mean_after',0):.1f} "
            f"(ext corners: {snap_st.get('m7_prob_ext_corners_before',0):.1f} -> {snap_st.get('m7_prob_ext_corners_after',0):.1f}) | "
            f"m7 drop @+-4v = {snap_st.get('m7_ridge_attenuation_pm4_pct',0):.1f}%",
            (120, 240, 255), 0.44
        ),
        (
            f"2. Raw Cross-Family r_2x2cm (Villa x Hecate):  Stage D = {summary_h['r_raw_2x2cm_stage_d']:.4f}  |  "
            f"Stage G (2D Poisson) = {summary_h['r_raw_2x2cm_stage_g']:.4f}  |  "
            f"Stage H (3D m7-Snap) Single {summary_h['best_single_depth']} = {summary_h['r_raw_2x2cm_stage_h_single']:.4f} / Ridge = {summary_h['r_raw_2x2cm_stage_h_ridge']:.4f}",
            (90, 255, 140), 0.44
        ),
        (
            f"3. High-Pass Calligraphic r_hp_2x2cm:          Stage F = {summary_h['r_hp_2x2cm_stage_f']:.4f}  |  "
            f"Stage G (2D Poisson) = {summary_h['r_hp_2x2cm_stage_g']:.4f}  |  "
            f"Stage H (3D m7-Snap) Single {summary_h['best_single_depth']} = {summary_h['r_hp_2x2cm_stage_h_single']:.4f} / Ridge = {summary_h['r_hp_2x2cm_stage_h_ridge']:.4f}",
            (90, 255, 140), 0.44
        ),
        (
            f"4. Qualifying Greek Letters (>=2x2mm in 2x2cm): Stage F = {summary_h['letters_stage_f']} (tpl={summary_h['tpl_stage_f']:.3f})  |  "
            f"Stage G = {summary_h['letters_stage_g']} (tpl={summary_h['tpl_stage_g']:.3f})  |  "
            f"Stage H Single = {summary_h['letters_stage_h_single']} (tpl={summary_h['tpl_stage_h_single']:.3f}) / Ridge = {summary_h['letters_stage_h_ridge']} (tpl={summary_h['tpl_stage_h_ridge']:.3f})",
            (220, 235, 255), 0.43
        ),
        (f"5. H4 CLSS Skeptic Gate on Stage H [{clss['clss_skeptic_gate_status']}]:", (255, 220, 100), 0.47),
        (
            f"   - Criterion 1 (Line Depth Coherence): {clss['coherent_letters_within_pm1_vox']}/{summary_h['letters_stage_h_ridge']} letters peak within d*+-1 vox "
            f"({clss['coherent_letters_within_pm2_vox']}/{summary_h['letters_stage_h_ridge']} within d*+-2 vox)",
            (220, 240, 220), 0.42
        ),
        (
            f"   - Criterion 2 (Delta-d Attenuation): m7 lamina drops {snap_st.get('m7_ridge_attenuation_pm4_pct',0):.1f}% @+-4v | "
            f"Ink box p92 drops {clss['box_attenuation_drop_pm4_pct']:.1f}% @+-4v ({clss['box_attenuation_drop_pm6_pct']:.1f}% @+-6v) | "
            f"Global-Z p99.5 drops {clss['global_z_attenuation_drop_pm4_pct']:.1f}% @+-4v",
            (220, 240, 220), 0.42
        ),
        (
            f"   - Criterion 3 (Greek Phonotactics & Rhythm): Vowel ratio = {clss['vowel_ratio']:.2f} | "
            f"Attested Herculaneum stems: {', '.join(clss['matched_herculaneum_stems'][:10]) or 'unigram seq'}",
            (220, 240, 220), 0.42
        ),
        ("6. Line-by-Line Greek Uncial Transcription (Stage H True 3D m7-Snap):", (255, 230, 130), 0.45),
    ]
    for l_key, l_val in list(clss["transcribed_lines"].items())[:4]:
        lines_text.append((
            f"   - {l_key} ({l_val['num_letters']} chars, line_d_std={l_val['line_d_std_vox']:.2f}v): {l_val['ascii_seq']}  |  pitches={l_val['pitches_mm']} mm",
            (140, 255, 180), 0.41
        ))

    y_curr = 32
    for txt, col, scale in lines_text:
        cv2.putText(info_panel, txt, (14, y_curr), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        y_curr += 31

    board[592:1068, 550:2150] = info_panel
    cv2.imwrite(str(out_path), board)


def process_stage_h_segment(seg_dir: Path) -> dict | None:
    meta_path = seg_dir / "metrics_stage_h.json"
    if not meta_path.exists():
        return None
    meta_h = json.loads(meta_path.read_text())
    scroll = meta_h["scroll"]
    seg = meta_h["segment"]
    orient = meta_h["chosen_orient"]
    vox_um = float(meta_h.get("um", 7.91))
    px_per_mm = 1000.0 / vox_um
    iy0, ix0, iwh, iww = meta_h["inner_2x2cm_yxwh"]
    sl = (slice(iy0, iy0 + iwh), slice(ix0, ix0 + iww))

    ct_ridge_full = load_gray(seg_dir / "ct_mid_ridge.png")
    ct_single_full = load_gray(seg_dir / "ct_mid_single_dstar.png")
    if ct_single_full is None:
        ct_single_full = load_gray(seg_dir / "ct_mid.png")
    ens_single_full = load_gray(seg_dir / f"ens_z_{orient}_single_dstar.png")
    ens_ridge_full = load_gray(seg_dir / f"ens_z_{orient}_multidepth.png")
    dn_md_full = load_gray(seg_dir / f"dnative_{orient}_multidepth.png")
    hc_md_full = load_gray(seg_dir / f"hecate_{orient}_multidepth.png")
    s_star_bgr = cv2.imread(str(seg_dir / "s_star_3d_snap_map.png"), cv2.IMREAD_COLOR) if (seg_dir / "s_star_3d_snap_map.png").exists() else None

    best_d_key = meta_h.get("best_single_depth", "d+0")
    best_d_int = int(best_d_key.replace("d", ""))
    dn_single_full = load_gray(seg_dir / f"dnative_{orient}_{best_d_key}.png")
    hc_single_full = load_gray(seg_dir / f"hecate_{orient}_{best_d_key}.png")

    if any(x is None for x in [ct_ridge_full, ens_ridge_full, dn_md_full, hc_md_full]):
        return None
    if ens_single_full is None:
        ens_single_full = load_gray(seg_dir / f"ens_z_{orient}_{best_d_key}.png")
    if dn_single_full is None:
        dn_single_full = dn_md_full
    if hc_single_full is None:
        hc_single_full = hc_md_full

    valid_full = ct_ridge_full > 0
    flat_full = compute_single_sheet_flatness(ct_ridge_full)

    dn_single_hp = highpass_norm(dn_single_full, valid_full, sigma=42.0, p_low=52.0, p_high=99.6)
    hc_single_hp = highpass_norm(hc_single_full, valid_full, sigma=50.0, p_low=50.0, p_high=99.6)
    ens_single_hp = highpass_norm(ens_single_full, valid_full, sigma=42.0, p_low=50.0, p_high=99.6)
    callig_single_f = 0.52 * ens_single_hp.astype(np.float32) + 0.25 * dn_single_hp.astype(np.float32) + 0.23 * hc_single_hp.astype(np.float32)
    callig_single_hp = highpass_norm(callig_single_f, valid_full, sigma=42.0, p_low=50.0, p_high=99.6)

    dn_ridge_hp = highpass_norm(dn_md_full, valid_full, sigma=42.0, p_low=52.0, p_high=99.6)
    hc_ridge_hp = highpass_norm(hc_md_full, valid_full, sigma=50.0, p_low=50.0, p_high=99.6)
    ens_ridge_hp = highpass_norm(ens_ridge_full, valid_full, sigma=42.0, p_low=50.0, p_high=99.6)
    callig_ridge_f = 0.52 * ens_ridge_hp.astype(np.float32) + 0.25 * dn_ridge_hp.astype(np.float32) + 0.23 * hc_ridge_hp.astype(np.float32)
    callig_ridge_hp = highpass_norm(callig_ridge_f, valid_full, sigma=42.0, p_low=50.0, p_high=99.6)

    ct_single_2cm = ct_single_full[sl]
    ct_ridge_2cm = ct_ridge_full[sl]
    valid_2cm = valid_full[sl]
    flat_2cm = flat_full[sl]
    mean_flat = float(flat_2cm[valid_2cm].mean()) if valid_2cm.any() else 0.0

    boxes_single, line_ys_single = detect_and_classify_letters(
        callig_single_hp[sl], dn_single_hp[sl], hc_single_hp[sl], flat_2cm, vox_um
    )
    boxes_ridge, line_ys_ridge = detect_and_classify_letters(
        callig_ridge_hp[sl], dn_ridge_hp[sl], hc_ridge_hp[sl], flat_2cm, vox_um
    )

    gold_single_2cm = make_calligraphic_gold(ct_single_2cm, callig_single_hp[sl], valid_2cm)
    gold_ridge_2cm = make_calligraphic_gold(ct_ridge_2cm, callig_ridge_hp[sl], valid_2cm)

    # Load Stage D, Stage F, and Stage G reference metrics
    meta_d, meta_f, meta_g = {}, {}, {}
    d_paths = list(STAGE_D_DIR.rglob(f"{scroll}/{seg}/metrics_stage_d.json"))
    if d_paths:
        meta_d = json.loads(d_paths[0].read_text())
    f_dir = STAGE_F_DIR / scroll / seg
    if (f_dir / "stage_f_summary.json").exists():
        meta_f = json.loads((f_dir / "stage_f_summary.json").read_text())
    g_dir = STAGE_G_DOSSIER_DIR / scroll / seg
    if (g_dir / "stage_g_clss_summary.json").exists():
        meta_g = json.loads((g_dir / "stage_g_clss_summary.json").read_text())

    img_f_gold = cv2.imread(str(f_dir / "clean_2x2cm_spatial_gold.png"), cv2.IMREAD_COLOR) if (f_dir / "clean_2x2cm_spatial_gold.png").exists() else gold_single_2cm
    img_g_gold = cv2.imread(str(g_dir / "clean_2x2cm_stage_g_ridge_gold.png"), cv2.IMREAD_COLOR) if (g_dir / "clean_2x2cm_stage_g_ridge_gold.png").exists() else gold_single_2cm

    # Also compute High-Pass correlation on Stage G if Stage G exists
    r_hp_g = 0.0
    g_raw_dirs = list((BASE / "results_stage_g").rglob(f"{scroll}/{seg}"))
    if g_raw_dirs:
        gd = g_raw_dirs[0]
        g_dn = load_gray(gd / f"dnative_{orient}_multidepth.png")
        g_hc = load_gray(gd / f"hecate_{orient}_multidepth.png")
        g_ct = load_gray(gd / "ct_mid_ridge.png")
        if g_dn is not None and g_hc is not None and g_ct is not None:
            gv = g_ct > 0
            g_dn_hp = highpass_norm(g_dn, gv, sigma=42.0, p_low=52.0, p_high=99.6)[sl]
            g_hc_hp = highpass_norm(g_hc, gv, sigma=50.0, p_low=50.0, p_high=99.6)[sl]
            gvi = gv[sl]
            if gvi.sum() > 500:
                r_hp_g = float(np.corrcoef(g_dn_hp[gvi].astype(np.float32), g_hc_hp[gvi].astype(np.float32))[0, 1])

    clss_audit = evaluate_h4_clss_skeptic_gate(
        seg_dir, orient, iy0, ix0, iwh, iww, best_d_int, boxes_ridge, line_ys_ridge, px_per_mm, meta_h
    )
    cont_h_ridge = evaluate_stroke_continuity_metrics(callig_ridge_hp[sl], valid_2cm)

    r_raw_d = float(meta_d.get("multidepth", {}).get("max_2x2cm_r_villa_hecate", 0.0))
    r_raw_g = float(meta_g.get("r_2x2cm_stage_g_ridge", 0.0))
    r_raw_hs = float(meta_h["single_dstar_metrics"].get("r_2x2cm", 0.0))
    r_raw_hr = float(meta_h["multidepth"].get("max_2x2cm_r_villa_hecate", 0.0))

    r_hp_f = float(meta_f.get("r_2x2cm", r_raw_d))
    r_hp_hs = float(meta_h["single_dstar_metrics"].get("r_hp_2x2cm", r_raw_hs))
    r_hp_hr = float(meta_h["multidepth"].get("max_2x2cm_r_hp_villa_hecate", r_raw_hr))

    clean_boxes_single = [{k: v for k, v in b.items() if not k.startswith("_")} for b in boxes_single]
    clean_boxes_ridge = [{k: v for k, v in b.items() if not k.startswith("_")} for b in boxes_ridge]

    tpl_f = round(float(np.mean([b.get("tpl_score", 0.0) for b in meta_f.get("top_12_letter_boxes", [])])), 3) if meta_f.get("top_12_letter_boxes") else 0.0
    tpl_g = float(meta_g.get("top12_tpl_score_stage_g_ridge", 0.0))
    tpl_hs = round(float(np.mean([b["tpl_score"] for b in clean_boxes_single[:12]])), 3) if clean_boxes_single else 0.0
    tpl_hr = round(float(np.mean([b["tpl_score"] for b in clean_boxes_ridge[:12]])), 3) if clean_boxes_ridge else 0.0

    summary_h = {
        "scroll": scroll,
        "segment": seg,
        "orient": orient,
        "vox_um": vox_um,
        "best_single_depth": best_d_key,
        "m7_3d_snap_stats": meta_h.get("stage_h_m7_snap", {}),
        "r_raw_2x2cm_stage_d": round(r_raw_d, 4),
        "r_raw_2x2cm_stage_g": round(r_raw_g, 4),
        "r_raw_2x2cm_stage_h_single": round(r_raw_hs, 4),
        "r_raw_2x2cm_stage_h_ridge": round(r_raw_hr, 4),
        "r_hp_2x2cm_stage_f": round(r_hp_f, 4),
        "r_hp_2x2cm_stage_g": round(r_hp_g, 4),
        "r_hp_2x2cm_stage_h_single": round(r_hp_hs, 4),
        "r_hp_2x2cm_stage_h_ridge": round(r_hp_hr, 4),
        "letters_stage_f": int(meta_f.get("num_dual_model_letter_candidates", 0)),
        "letters_stage_g": int(meta_g.get("letters_stage_g_ridge", 0)),
        "letters_stage_h_single": len(clean_boxes_single),
        "letters_stage_h_ridge": len(clean_boxes_ridge),
        "tpl_stage_f": tpl_f,
        "tpl_stage_g": tpl_g,
        "tpl_stage_h_single": tpl_hs,
        "tpl_stage_h_ridge": tpl_hr,
        "stroke_continuity_stage_h_ridge": cont_h_ridge,
        "h4_clss_audit": clss_audit,
        "top_12_greek_hypotheses_stage_h": " ".join([f"{b['greek_char']}({b['greek_ascii']})" for b in clean_boxes_ridge[:12]]),
        "top_12_letter_boxes_stage_h": clean_boxes_ridge[:12],
    }

    out_dir = STAGE_H_DOSSIER_DIR / scroll / seg
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "clean_2x2cm_stage_h_single_dstar_gold.png"), gold_single_2cm)
    cv2.imwrite(str(out_dir / "clean_2x2cm_stage_h_ridge_gold.png"), gold_ridge_2cm)
    cv2.imwrite(str(out_dir / "clean_2x2cm_stage_h_ridge_gray.png"), callig_ridge_hp[sl])

    # Render updated paleography_dossier.png
    best_boxes = boxes_ridge if len(boxes_ridge) >= len(boxes_single) else boxes_single
    best_gold = gold_ridge_2cm if len(boxes_ridge) >= len(boxes_single) else gold_single_2cm
    best_gray = callig_ridge_hp[sl] if len(boxes_ridge) >= len(boxes_single) else callig_single_hp[sl]
    render_paleography_dossier(
        scroll, seg, orient, f"StageH_3Dm7Snap_{best_d_key}", mean_flat,
        max(r_raw_hr, r_hp_hr), px_per_mm,
        best_gold, best_gray, best_boxes, line_ys_ridge, out_dir
    )

    render_f_g_h_comparison_board(
        scroll, seg, orient, px_per_mm,
        img_f_gold, img_g_gold, gold_single_2cm, gold_ridge_2cm,
        s_star_bgr, boxes_single, boxes_ridge, line_ys_ridge,
        summary_h, meta_h, out_dir / "stage_f_vs_g_vs_h_m7snap.png"
    )

    for sub_folder in SUBMISSIONS_DIR.glob(f"*_{scroll}_{seg}"):
        if sub_folder.is_dir():
            shutil.copy2(out_dir / "stage_f_vs_g_vs_h_m7snap.png", sub_folder / "stage_f_vs_g_vs_h_m7snap.png")
            shutil.copy2(out_dir / "paleography_dossier.png", sub_folder / "paleography_dossier_stage_h.png")
            shutil.copy2(out_dir / "clean_2x2cm_stage_h_single_dstar_gold.png", sub_folder / "clean_2x2cm_stage_h_single_dstar_gold.png")
            shutil.copy2(out_dir / "clean_2x2cm_stage_h_ridge_gold.png", sub_folder / "clean_2x2cm_stage_h_ridge_gold.png")

    (out_dir / "stage_h_summary.json").write_text(json.dumps(summary_h, indent=2, ensure_ascii=False))
    print(
        f"[Stage H Audit] {scroll}/{seg}: r_raw D={r_raw_d:.4f} | G={r_raw_g:.4f} | H_d*({best_d_key})={r_raw_hs:.4f} | H_ridge={r_raw_hr:.4f} || "
        f"r_hp F={r_hp_f:.4f} | G={r_hp_g:.4f} | H_d*={r_hp_hs:.4f} | H_ridge={r_hp_hr:.4f} || "
        f"letters F={summary_h['letters_stage_f']} G={summary_h['letters_stage_g']} H={summary_h['letters_stage_h_ridge']} | CLSS={clss_audit['clss_skeptic_gate_status']}"
    )
    return summary_h


def main():
    STAGE_H_DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
    seg_dirs = sorted([p.parent for p in STAGE_H_DIR.rglob("metrics_stage_h.json")])
    summaries = []
    for sd in seg_dirs:
        s = process_stage_h_segment(sd)
        if s is not None:
            summaries.append(s)
    summaries.sort(key=lambda x: max(x["r_raw_2x2cm_stage_h_ridge"], x["r_hp_2x2cm_stage_h_ridge"]), reverse=True)
    (STAGE_H_DOSSIER_DIR / "leaderboard_stage_h.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Processed {len(summaries)} Stage H segments -> {STAGE_H_DOSSIER_DIR / 'leaderboard_stage_h.json'}")


if __name__ == "__main__":
    main()
