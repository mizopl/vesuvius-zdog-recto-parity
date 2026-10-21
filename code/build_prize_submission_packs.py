#!/usr/bin/env python3
"""
Builds self-contained $50,000 Vesuvius Challenge First Letters Prize submission packs
(`submissions_ready/<Scroll_ID>/`) for our top verified unread scrolls, and updates
`results_stage_f/leaderboard_stage_f.json` with any new segments from `results_stage_f_restriped/`.
"""
import json
import shutil
from pathlib import Path
from stage_f_sheet_tracker_and_paleography import (
    BASE,
    STAGE_D_DIR,
    STAGE_F_DIR,
    process_segment,
)

RESTRIPED_DIR = BASE / "results_stage_f_restriped"
STAGE_E2_DIR = BASE / "results_stage_e2"
SUBMISSIONS_DIR = BASE / "submissions_ready"


def collect_all_stage_f_summaries() -> list[dict]:
    STAGE_F_DIR.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str], dict] = {}

    # 1. Load already-processed Stage D -> Stage F summaries (or process if missing)
    for meta_p in sorted(STAGE_D_DIR.rglob("metrics_stage_d.json")):
        sd = meta_p.parent
        meta = json.loads(meta_p.read_text())
        key = (meta["scroll"], meta["segment"])
        existing_sum = STAGE_F_DIR / meta["scroll"] / meta["segment"] / "stage_f_summary.json"
        if existing_sum.exists():
            s = json.loads(existing_sum.read_text())
            s["_src_dir"] = str(sd)
            by_key[key] = s
        else:
            s = process_segment(sd, STAGE_F_DIR)
            if s is not None:
                s["_src_dir"] = str(sd)
                by_key[key] = s

    # 2. Evaluate new / restriped segments in results_stage_f_restriped
    restriped_f_out = BASE / "results_stage_f_from_restriped"
    restriped_f_out.mkdir(parents=True, exist_ok=True)
    if RESTRIPED_DIR.exists():
        for meta_p in sorted(RESTRIPED_DIR.rglob("metrics_stage_d.json")):
            sd = meta_p.parent
            meta = json.loads(meta_p.read_text())
            cached_sum = restriped_f_out / meta["scroll"] / meta["segment"] / "stage_f_summary.json"
            if cached_sum.exists():
                s = json.loads(cached_sum.read_text())
            else:
                s = process_segment(sd, restriped_f_out)
            if s is None:
                continue
            s["_src_dir"] = str(sd)
            key = (s["scroll"], s["segment"])
            if key not in by_key:
                # Copy to main STAGE_F_DIR as well
                dst = STAGE_F_DIR / s["scroll"] / s["segment"]
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(Path(s["out_dir"]), dst)
                s["out_dir"] = str(dst)
                by_key[key] = s
            else:
                old = by_key[key]
                old_sc = (old["num_dual_model_letter_candidates"], old["mean_sheet_flatness"], old["r_2x2cm"])
                new_sc = (s["num_dual_model_letter_candidates"], s["mean_sheet_flatness"], s["r_2x2cm"])
                if new_sc > old_sc:
                    dst = STAGE_F_DIR / s["scroll"] / s["segment"]
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(Path(s["out_dir"]), dst)
                    s["out_dir"] = str(dst)
                    by_key[key] = s

    summaries = list(by_key.values())
    summaries.sort(
        key=lambda x: (
            x["num_dual_model_letter_candidates"] >= 10 and x["mean_sheet_flatness"] >= 0.75,
            x["num_dual_model_letter_candidates"] >= 10,
            x["r_2x2cm"],
        ),
        reverse=True,
    )
    clean_list = [{k: v for k, v in s.items() if not k.startswith("_")} for s in summaries]
    (STAGE_F_DIR / "leaderboard_stage_f.json").write_text(json.dumps(clean_list, indent=2, ensure_ascii=False))
    return summaries


def build_scroll_submission_pack(scroll: str, seg_summaries: list[dict], manifest: dict):
    if not seg_summaries:
        return
    primary = seg_summaries[0]
    seg = primary["segment"]
    orient = primary["orient"]
    src_d_dir = Path(primary["_src_dir"])
    f_dir = Path(primary["out_dir"])
    e2_dir = STAGE_E2_DIR / scroll / seg

    pack_dir = SUBMISSIONS_DIR / f"{scroll}_{seg}"
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Copy visual evidence files with clear numbered names
    copy_map = [
        (src_d_dir / f"submission_gold_{orient}_multidepth.png", pack_dir / "1_full_segment_gold.png"),
        (src_d_dir / f"ens_z_{orient}_multidepth.png", pack_dir / "2_full_segment_ink.png"),
        (src_d_dir / "ct_mid.png", pack_dir / "3_full_segment_ct_mid.png"),
        (f_dir / "clean_2x2cm_spatial_gold.png", pack_dir / "4_prize_window_2x2cm_clean_gold.png"),
        (f_dir / "clean_2x2cm_spatial_gray.png", pack_dir / "5_prize_window_2x2cm_clean_gray.png"),
        (f_dir / "paleography_dossier.png", pack_dir / "6_paleography_dossier_annotated.png"),
        (f_dir / "top3_windows_overview.png", pack_dir / "7_top3_windows_overview.png"),
        (e2_dir / "line_strips_zoom.png", pack_dir / "8_line_strips_zoom.png"),
        (e2_dir / "letter_gallery_top12.png", pack_dir / "9_letter_gallery_top12.png"),
    ]
    for src, dst in copy_map:
        if src.exists():
            shutil.copy2(src, dst)

    vol_info = manifest.get(scroll, {})
    vol_name = vol_info.get("volume", "unknown")
    vox_um = primary["vox_um"]
    y0, x0, wh, ww = primary["best_2x2cm_yxwh"]

    # Build markdown report
    lines = [
        f"# Vesuvius Challenge — First Letters Prize ($50,000) Submission Dossier: `{scroll}` (`{seg}`)",
        "",
        "## 1. Scroll & Segment Metadata",
        f"- **Scroll ID:** `{scroll}`",
        f"- **Primary Segment ID:** `{seg}`",
        f"- **Canonical Volume:** `{vol_name}` (`{vox_um} µm` isotropic voxel size)",
        f"- **Surface Normal Orientation:** `{orient}`",
        f"- **In-Focus Normal Depth Pair:** `{primary['best_depth_pair']}` (best single slice: `{primary['best_single_depth']}`)",
        f"- **Cross-Architecture Ink Correlation (`Villa` 3D-2D vs `Hecate` 3D UNet):**",
        f"  - Whole-segment correlation: `r_all = {primary['r_all']:.4f}`",
        f"  - Peak `2x2 cm` (`4 cm²`) window correlation: **`r_2x2cm = {primary['r_2x2cm']:.4f}`**",
        f"- **Single-Sheet Papyrus Flatness & Fiber Weave Score:** **`{primary['mean_sheet_flatness']:.3f}`** (scale `0..1`, rejecting air gaps and oblique wrap folds)",
        f"- **Selected `2 cm x 2 cm` (`4 cm²`) Prize Window (`[y0, x0, H, W]` px):** `[{y0}, {x0}, {wh}, {ww}]` (`20.0 mm x 20.0 mm`)",
        f"- **Dual-Model Confirmed Character Candidates (`>= 2.0 mm x 2.0 mm`):** **`{primary['num_dual_model_letter_candidates']}`** (minimum prize requirement: `10` letters)",
        f"- **Top-12 Paleographic Uncial Sequence:** `{primary['top_12_greek_hypotheses']}`",
        "",
        "## 2. Top 12 Greek Character Candidates (`>= 2.0 mm x 2.0 mm` inside the `2 cm x 2 cm` Window)",
        "",
        "| # | Greek Uncial Hypothesis | Box Size (`mm`) | Box `[x, y, w, h]` (`px` in `2x2cm` crop) | Line ID | `Villa` (`p94`) | `Hecate` (`p94`) | Stroke Contrast | Sheet Flatness | Template Score |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for i_b, b in enumerate(primary["top_12_letter_boxes"]):
        lines.append(
            f"| **#{i_b+1}** | **`{b['greek_char']}` (`{b['greek_ascii']}`)** | `{b['w_mm']:.2f} x {b['h_mm']:.2f} mm` | "
            f"`[{b['x']}, {b['y']}, {b['w']}, {b['h']}]` | `L{b['line_id']}` | `{b['dn_p94']:.1f}` | `{b['hc_p94']:.1f}` | "
            f"`{b['contrast']:.1f}` | `{b['flatness']:.2f}` | `{b['tpl_score']:.3f}` |"
        )

    if len(seg_summaries) > 1:
        lines.extend([
            "",
            f"## 3. Additional Supporting Segments on `{scroll}`",
            "",
            "| Segment ID | Orient | `r_2x2cm` | Sheet Flatness | Letter Candidates (`>=2x2mm`) | Top Greek Uncial Hypotheses |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for s_aux in seg_summaries[1:]:
            lines.append(
                f"| `{s_aux['segment']}` | `{s_aux['orient']}` | `{s_aux['r_2x2cm']:.4f}` | `{s_aux['mean_sheet_flatness']:.2f}` | "
                f"`{s_aux['num_dual_model_letter_candidates']}` | `{s_aux['top_12_greek_hypotheses']}` |"
            )

    lines.extend([
        "",
        "## 4. Included Evidence Files in This Submission Pack",
        "- `1_full_segment_gold.png` — Full segment multi-depth 4-model ensemble gold overlay on CT papyrus fibers",
        "- `2_full_segment_ink.png` — Full segment raw z-score 4-model ensemble (`dnative + seed42 + seed43 + hecate`)",
        "- `3_full_segment_ct_mid.png` — Central surface CT slice (`d = 0`) showing orthogonal papyrus fiber weave",
        "- `4_prize_window_2x2cm_clean_gold.png` — Un-annotated `2 cm x 2 cm` (`4 cm²`) in-focus calligraphic gold crop",
        "- `5_prize_window_2x2cm_clean_gray.png` — Un-annotated `2 cm x 2 cm` (`4 cm²`) spatially-tracked in-focus ink signal",
        "- `6_paleography_dossier_annotated.png` — High-resolution (`1920x960`) Paleographic Dossier with `L1..L5` lines, numbered `>=2x2 mm` boxes, and `4x3` evidence cards",
        "- `7_top3_windows_overview.png` — Comparison of top 3 non-overlapping `2 cm x 2 cm` windows on the segment",
        "- `8_line_strips_zoom.png` — Zoomed `4.5 mm x 20.0 mm` horizontal text-line strips (`2 mm` scale ticks)",
        "- `9_letter_gallery_top12.png` — `3x4` zoomed individual character crop cards (`CT+Gold | In-Focus Gray`)",
        "",
    ])

    (pack_dir / f"SUBMISSION_REPORT_{scroll}.md").write_text("\n".join(lines))
    meta_out = {
        "scroll": scroll,
        "primary_segment": primary,
        "supporting_segments": seg_summaries[1:],
    }
    (pack_dir / "submission_metadata.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
    print(f"[Packaged] {scroll}/{seg} -> {pack_dir} ({primary['num_dual_model_letter_candidates']} letters, r_2x2cm={primary['r_2x2cm']:.4f}, flat={primary['mean_sheet_flatness']:.2f})")


def main():
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "manifest_stage_d.json").read_text())
    summaries = collect_all_stage_f_summaries()

    by_scroll: dict[str, list[dict]] = {}
    for s in summaries:
        by_scroll.setdefault(s["scroll"], []).append(s)

    packaged_entries = []
    for scroll, segs in sorted(by_scroll.items()):
        segs.sort(
            key=lambda x: (
                x["num_dual_model_letter_candidates"] >= 10,
                x["r_2x2cm"] * (x["mean_sheet_flatness"] ** 0.5),
            ),
            reverse=True,
        )
        for idx, s in enumerate(segs):
            # Package primary segment for every scroll with r_2x2cm >= 0.20, plus any additional segment with >=10 letters or r_2x2cm >= 0.35
            if (idx == 0 and s["r_2x2cm"] >= 0.20) or s["num_dual_model_letter_candidates"] >= 10 or s["r_2x2cm"] >= 0.35:
                reordered = [s] + [o for o in segs if o["segment"] != s["segment"]]
                build_scroll_submission_pack(scroll, reordered, manifest)
                packaged_entries.append(s)

    packaged_entries.sort(
        key=lambda x: (
            x["num_dual_model_letter_candidates"] >= 10 and x["mean_sheet_flatness"] >= 0.72,
            x["num_dual_model_letter_candidates"] >= 10,
            x["r_2x2cm"],
        ),
        reverse=True,
    )

    idx_lines = [
        "# Vesuvius Challenge — First Letters Prize ($50,000) Master Submission Packs Index",
        "",
        "| Rank | Scroll | Segment | Orient | `r_2x2cm` | Sheet Flatness | Letters (`>=2x2mm`) | Top-12 Greek Uncial Sequence | Submission Folder |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for rk, e in enumerate(packaged_entries, 1):
        folder = f"{e['scroll']}_{e['segment']}"
        idx_lines.append(
            f"| **{rk}** | **`{e['scroll']}`** | `{e['segment']}` | `{e['orient']}` | **`{e['r_2x2cm']:.4f}`** | "
            f"**`{e['mean_sheet_flatness']:.2f}`** | **`{e['num_dual_model_letter_candidates']}`** | "
            f"`{e['top_12_greek_hypotheses']}` | `{folder}/` |"
        )
    (SUBMISSIONS_DIR / "MASTER_SUBMISSIONS_INDEX.md").write_text("\n".join(idx_lines))
    print(f"\nWrote master index with {len(packaged_entries)} submission packs to {SUBMISSIONS_DIR / 'MASTER_SUBMISSIONS_INDEX.md'}")


if __name__ == "__main__":
    main()
