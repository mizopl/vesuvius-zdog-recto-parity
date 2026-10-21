#!/usr/bin/env bash
set -euo pipefail
echo "[1/4] Preparing 3D single-sheet m7-Zarr snapped meshes..."
python3 code/snap_meshes_to_m7_zarr.py
python3 code/stage_i_prepare_36_candidates.py
echo "[2/4] Running Wide-Area m7-Snap Column Scout & 4x-TTA GPU Promoter..."
python3 code/colab_stage_i_wide_scout.py
python3 code/colab_stage_o_gpu_promoter.py
echo "[3/4] Running Thin-Layer Z-DoG Audit & Seam Stitcher..."
python3 code/stage_i_clss_audit_and_loop.py
python3 code/stage_k_clss_seam_stitcher.py
echo "[4/4] Running V4 3D Normal-Parity Recto Classifier, Kalamos 2D Corridor Tracker & Viterbi-Leiden Decoder..."
python3 code/stage_o_3d_parity_consensus_and_leiden.py
echo "Done. All outputs written to results_stage_o_consensus_leiden/"
