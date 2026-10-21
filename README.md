# Vesuvius Zero-Hallucination Surface-Derivative & Recto-Parity Pipeline (`m7-Snap + Dual-Stage Z-DoG + Kalamos + Viterbi-Leiden`)

Open-source toolkit for **non-hallucinatory thin-layer ink extraction**, **3D geometric *recto/verso* normal-parity rectification**, and **strict Leiden diplomatic decoding** on unread Herculaneum scroll volumes (`PHerc. 1203`, `PHerc. 0800`, `PHerc. 0125`, and `PHerc. 0846A`).

---

## 1. Problem Identification & Why Existing Pipelines Struggle on Unread Scrolls

When applying existing pretrained Vesuvius ink detection models (`dense_native`, `hybrid_3d2d` seeds `42`/`43`, `Hecate`) directly to newly segmented patches on unread scrolls (`PHerc. 1203`, `PHerc. 0800`, `PHerc. 0125`, `PHerc. 0846A`), two systematic failure modes dominate:

1. **3D Structural Fiber-Knot & Mineral Clutter masquerading as Ink (`Depth-Invariant False Positives`):**
   - Standard ink models integrated across a slab of $z$-slices respond strongly to dense papyrus fiber crossings, calcium/mineral inclusions, and compression ridges that span $8\text{–}15$ voxels ($75\text{–}140\,\mu\text{m}$) in depth.
   - In our initial baseline audit (`Stage H`), even when a 3D mesh was snapped cleanly onto a single `m7` sheet (`63%–85%` sheet probability drop at $\pm 4$ voxels), the highest-scoring raw candidate boxes selected by standard 2D connected-component detectors exhibited only `-0.6%` to `+12.5%` intensity drop at $d^* \pm 4$ voxels—meaning the detector was locking onto thick 3D structural fiber knots rather than thin surface carbon ink ($\le 2\text{–}3$ voxels thick).

2. **Arbitrary $(u, v)$ UV-Flattening Orientation & *Recto/Verso* Parity Inversion:**
   - Automatic patch growth (`auto_grown` and `lasagna` strips) parameterizes each 2D `.tifxyz` grid with an arbitrary local coordinate frame $(u,v)$. As a result, the inward radial normal $\mathbf{n} = \partial_u \mathbf{r} \times \partial_v \mathbf{r}$ frequently points **outward** (`inward_sign = -1`, viewing the sheet from behind in mirror image) or places the vertical scroll axis upside-down (`sz > 0`).
   - Worse, blind 2D searches over all 8 dihedral symmetries ($D_4$, including $90^\circ$ transpositions) can accidentally rotate vertical *verso* fibers into horizontal lines.

---

## 2. Our Solution & Decision Logic (What Worked vs. Gently Rejected Hypotheses)

To eliminate both failure modes without retraining or fine-tuning on target scroll regions (ensuring **zero overlap between training and prediction regions**), we developed a four-stage physical and geometric pipeline:

### Step A — Single-Sheet `m7-Zarr` 3D Surface Snapping (`code/snap_meshes_to_m7_zarr.py`)
- Candidate `.tifxyz` surfaces (`auto_grown` and `lasagna`) are snapped along their local 3D surface normal to the nearest peak of the `*-surface-m7-L0-th0.2.zarr` / `L2` sheet probability volume.
- **Single-Sheet Isolation Audit:** Across all 22 evaluated patches ($22 \times 22\text{ mm}$), 97.6%–100.0% of each patch lies on a single papyrus lamina (`single_sheet_flatness_pct`), with sheet probability dropping by `63%–85%` at $\pm 4$ voxels.

### Step B — Dual-Stage Thin-Layer Surface Derivative (`Z-DoG`, `code/stage_i_wide_scout_and_zdog.py` & `code/stage_o_3d_parity_consensus_and_leiden.py`)
- **Rejected Hypothesis #1 (Raw Max-Intensity / Slab Projection):** Taking the maximum or mean model activation across $d \in [-6, +6]$ maximizes visual brightness but floods candidate rankings with thick 3D fiber knots.
- **Rejected Hypothesis #2 (Single-Depth Difference Only):** Subtracting far background (`d = ±5..±6`) from a single nominal depth $d^*$ suppresses thick knots, but loses the upper or lower stroke of legitimate letters whenever a papyrus sheet undulates by $\pm 1\text{–}2$ voxels across a $2\times 2\text{ cm}$ window.
- **Adopted Solution (`Dual-Stage Max-Consensus Thin-Layer Z-DoG`):**
  We evaluate three independent model families (`dense_native` $128\times 128$, `hybrid_3d2d` `seed42+43`, and small-receptive-field `Hecate` $32\times 32$) across $d \in \{-6, \dots, +6\}$ voxels around the snapped `m7` surface, combine the local thin-layer core $d \in [d^*-2, d^*+2]$ with a $\pm 1$-voxel undulating-sheet lookahead envelope, and subtract the off-sheet background envelope at $d \in \{-6, -5, +5, +6\}$:
  $$I_{\text{Z-DoG}}(y, x) = \max\!\Big(0,\; I_{\text{sheet}}(y,x; d^* \pm 2) - \alpha \cdot I_{\text{bg}}(y,x; d \in \{\pm 5, \pm 6\})\Big)$$
  Every detected character slot is then audited on the **raw, unfiltered 3D stack**: a candidate is only accepted if its raw activation drops by **$>55\%$ (averaging `-74.2%` across our Top-10 columns)** at $d^* \pm 4$ voxels.

### Step C — 3D Geometric *Recto* Normal-Parity Rectification (`V4`, `code/stage_o_3d_parity_consensus_and_leiden.py`)
- **Rejected Hypothesis #3 (Unconstrained 8-Way $D_4$ Symmetry Search):** Allowing $90^\circ$ rotations (`transpose` / `rot90`) during orientation alignment risks turning vertical *verso* fibers into horizontal lines.
- **Adopted Solution (`Strict V4 Klein Four-Group 3D Parity`):**
  1. We first orient the papyrus fibers horizontally in 2D using the structure-tensor dominant angle ($\theta_{\text{rectified}} < 5^\circ$).
  2. We then restrict 2D transformations strictly to the **orientation-preserving horizontal subgroup $V_4 = \{\text{id}, \text{flip\_x}, \text{flip\_y}, \text{rot180}\}$** determined directly from 3D geometry:
     - `flip_x` flips the sign of the 3D surface normal ($\mathbf{n}_{\text{eff}} = -\mathbf{n}$) so that **100% of patches (`22/22`) point inward toward the scroll center (`inw_eff = +1`, true *recto* viewing parity)**.
     - `flip_y` flips the vertical column axis (`sz_eff < 0`) so lines read top-to-bottom in coherent column order.

### Step D — Zero-Drawing Kalamos 2D Corridor Tracker & Viterbi-Leiden Diplomatic Decoder
- **Rejected Hypothesis #4 (Corridor-Biased Template Matching & Synthetic Stroke Overlays):** Drawing vector skeleton lines (`cv2.line`) over detected characters or adding a corridor bonus inside the template matcher artificially inflates template scores. We completely removed both (`0` synthetic lines drawn; `0.0` corridor bonus in template matching).
- **Adopted Solution:**
  - **Kalamos 2D Anisotropic Nib Filtering:** Uses pure 2D Gaussian-derivative matched filters aligned with classical Herculaneum reed-pen (*kalamos*) stroke angles ($0^\circ, 90^\circ, \pm 35^\circ$) and a 2-tier multi-architecture consensus (`core`: agreement of $\ge 2/3$ models; `lookahead`: high-confidence primary stroke continuity within horizontal row corridors).
  - **Strict Leiden Diplomatic Edition (`\u0323` subdots) + Zero-LM Greedy Audit:** Each character slot is scored both by a Greek syllable-transition Viterbi decoder and by a **purely geometric Zero-LM Greedy classifier** (`0.0` bigram weight). Any glyph with template correlation $< 0.62$, depth drop $< 58\%$, or $< 2/3$ model agreement receives a mandatory Leiden uncertainty subdot (`Α̣`), while indisputable glyphs (`leiden_certain_frac = 66.4%`, `greedy_match_frac = 59.2%`) are printed without subdots.

---

## 3. Quantitative Results Across Four Unread Scrolls (`Stage O v2`)

All 22 candidate $4\text{ cm}^2$ windows across four eligible scrolls were evaluated under our 5-point `K4` verification gate:
1. `total_decoded_chars >= 18` in a single $4\text{ cm}^2$ window,
2. `mean_box_drop_pm4_pct >= 55%` (physical thin-sheet ink decay at $\pm 4$ voxels),
3. `mean_slot_concordance >= 0.72` (cross-depth & cross-model spatial IoU/correlation),
4. `mean_tpl_score >= 0.60` (unbiased geometric Herculaneum uncial correlation),
5. `multistroke_complete_frac >= 0.75` (complete multi-stroke letter anatomy).

**`19 out of 22` evaluated segments (`86.4%`) pass all `5/5` criteria (`K4 = 5/5 VERIFIED PASS`) across three scrolls (`PHerc. 1203`, `PHerc. 0800`, `PHerc. 0125`), plus a 16-character `3/5` candidate on `PHerc. 0846A`:**

| Scroll Volume | Verified `K4 = 5/5` Segments | Best Window Decoded Letters (`4 cm²`) | Mean $\pm 4$ Vox Ink Drop | Mean Multi-Model (`2/3`) Agreement | Mean Leiden Certainty (No Subdot) | Zero-LM Pure Geometric Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PHerc. 1203` (`20250820131727`)** | **14 segments** | **`46` chars (`6` rows)** (`52` in panorama) | **`-74.9%`** | **`91.4%`** | **`68.1%`** | **`61.0%`** |
| **`PHerc. 0800` (`20250521135224`)** | **4 segments** | **`44` chars (`6` rows)** | **`-74.7%`** | **`92.5%`** | **`71.4%`** | **`63.2%`** |
| **`PHerc. 0125` (`20250821151825`)** | **1 segment** | **`38` chars (`5` rows)** | **`-79.1%`** | **`87.0%`** | **`68.4%`** | **`57.9%`** |
| **`PHerc. 0846A` (`20250728152254`)** | `thin_wrap01` (`3/5`) | **`16` chars (`5` rows)** | **`-68.9%`** | **`94.0%`** | **`81.2%`** | **`81.2%`** |

*(Full per-segment metrics are provided in [`benchmarks/stage_o_quantitative_metrics.json`](benchmarks/stage_o_quantitative_metrics.json). In accordance with the Vesuvius Challenge First Letters non-disclosure rule prior to official announcement, the rendered First Letters images and `.tifxyz` meshes for each scroll are submitted directly via the official First Letters review form).*


---

## 3.5. Direct Mapping to the 2026 Vesuvius Challenge Open Problems (`scrollprize.org/2026_open_problems`)

This toolkit directly addresses five core bottlenecks identified in the official **October 2026 Open Problems** roadmap ([`scrollprize.org/2026_open_problems`](https://scrollprize.org/2026_open_problems)):

| 2026 Open Problem / Bottleneck (`scrollprize.org/2026_open_problems`) | Why It Limits Current Unread Scrolls | How Our Pipeline Solves / Mitigates It (with Code Module) |
| :--- | :--- | :--- |
| **1. Label Snapping & Approximate Surface Meshes** (§2 *Label quality: one of the main unwrapping bottlenecks*, §5 *Approximate surface labels*) | Semi-automated `auto_grown` and `lasagna` meshes wiggle or drift $2	ext{–}8$ voxels off the true papyrus lamina, causing surface-conditioned ink renders to miss the thin ink layer or cross fiber layers. | **[`code/snap_meshes_to_m7_zarr.py`](code/snap_meshes_to_m7_zarr.py)** + subvoxel lamina tracking in **[`code/stage_n_3d_voxel_and_around_scraper.py`](code/stage_n_3d_voxel_and_around_scraper.py)**: Snaps `.tifxyz` vertex grids along local 3D surface normals onto the single-sheet peak of `*-surface-m7-L0-th0.2.zarr` / `L2`, achieving **97.6%–100.0% single-sheet flatness** and **63%–85% sheet probability drop at $\pm 4$ voxels**. |
| **2. Sheet Switches, Mergers & Local Tracing Drift** (§2 *Meshes: adding connectivity*, §5 *Sheet switches*) | Local patch tracing can jump across adjacent windings in compressed or damaged zones, corrupting flattened columns. | **[`code/extend_pherc0800_single_sheet.py`](code/extend_pherc0800_single_sheet.py)**, **[`code/build_tifxyz_from_grids.py`](code/build_tifxyz_from_grids.py)** & **[`code/stage_k_clss_seam_stitcher.py`](code/stage_k_clss_seam_stitcher.py)**: Enforces a strict single-lamina `m7` valley constraint ($\pm 4$ voxel drop + `swirl_delam_frac` delamination detector) and stitches adjacent `.tifxyz` patches into coherent panoramas without sheet switches. |
| **3. Recto vs. Verso Orientation & Fiber-Axis Parameterization** (§2 *Surface prediction: recto facing inward*, *Fibers as connectivity clues*) | Unwrapped `.tifxyz` grids carry arbitrary $(u,v)$ axes; outward-pointing normals produce mirror-image text, and unconstrained 2D rotations can mistake vertical *verso* fibers for horizontal *recto* lines. | **[`code/stage_o_3d_parity_consensus_and_leiden.py`](code/stage_o_3d_parity_consensus_and_leiden.py)** (`lookup_mesh_3d_inward_normal_sign` + `V4` Klein four-group rectifier): Aligns horizontal *recto* fibers via the 2D structure tensor ($<5^\circ$ tilt) and uses the 3D radial vector from the scroll's umbilicus to enforce inward *recto* viewing parity (`inw_eff = +1` across `22/22` windows) without $90^\circ$ *verso* transpositions. |
| **4. Ink Depth Ambiguity & Pipeline Failure Diagnostics** (§3 *Fragment-trained ink models*, *What happens if models don't generalize?*, §5 *Ink depth ambiguity*) | 2D fragment labels do not specify where ink sits in depth; when models fail or hallucinate on a new scroll, it is unclear whether the cause is surface misplacement, 3D fiber-knot clutter, or missing signal. | **[`code/stage_o_3d_parity_consensus_and_leiden.py`](code/stage_o_3d_parity_consensus_and_leiden.py)** (`Dual-Stage Z-DoG` + `filter_boxes_by_frozen_stroke_z_attenuation`): Separates true thin-layer surface ink ($\le 30	ext{–}40\,\mu	ext{m}$, averaging **$-74.2\%$ intensity drop at $d^* \pm 4$ voxels**) from $8	ext{–}15$ voxel 3D fiber knots ($<12\%$ drop) and locates the exact subvoxel depth offset $d^* \in [-6, +6]$. |
| **5. Cross-Scroll Generalization & Cloud-Native OME-Zarr Scale** (§4 *Data scale: the infrastructure bottleneck*, §5 *Cross-scroll generalization*) | Full-scroll 2.4 µm / 9.6 µm volumes are terabytes in size, and single ink models often plateau or fail on unseen scrolls. | **[`code/fetch_s3_chunks.py`](code/fetch_s3_chunks.py)** & **[`code/colab_stage_i_wide_scout.py`](code/colab_stage_i_wide_scout.py)**: Streams only targeted 3D bounding-box chunks directly from `s3://vesuvius-challenge-open-data/` OME-Zarr (`9.6 µm L0` and `2.403 µm 77 keV L2`), performs fast 3-point (`d in {-4, 0, +4}`) wide scouting, and combines 3 architectures (`dense_native`, `hybrid_3d2d`, `32x32 Hecate`) via an autonomous **CLSS autoresearch loop** (analogous to the `karpathy/autoresearch` paradigm noted in §5). |

---

## 4. Reproducing the Pipeline

### Using Docker
```bash
docker build -t vesuvius-zdog-parity .
docker run --gpus all -v $(pwd)/data:/workspace/data vesuvius-zdog-parity
```

### Direct Python Execution
```bash
bash reproduce_pipeline.sh
```

---

## 5. Acknowledgments

- Iterative hypothesis generation, pre-registered falsification bets (`Stage A -> Stage O v2`), and automated 3-skeptic verification were orchestrated using our **CLSS (Closed-Loop Scientific System)** autoresearch framework.
- Special thanks to **Pliny (`@elder_plinius` / Pliny the Liberator)** for inspiration on epistemic rigor, creative boundary-pushing, and autonomous research workflows.
- Built on top of the open datasets, `m7` surface volumes, and pretrained checkpoints released by the **Vesuvius Challenge** team.
