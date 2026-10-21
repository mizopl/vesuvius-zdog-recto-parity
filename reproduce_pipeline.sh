#!/usr/bin/env bash
set -euo pipefail
echo "[1/3] Snapping candidate tifxyz meshes to single-sheet m7-Zarr surfaces..."
python3 code/snap_meshes_to_m7_zarr.py
echo "[2/3] Running Wide-Area m7-Snap Column Scout & Dual-Stage Z-DoG Ink Extraction..."
python3 code/stage_i_wide_scout_and_zdog.py
echo "[3/3] Running V4 3D Normal-Parity Recto Classifier, Kalamos 2D Corridor Tracker & Viterbi-Leiden Decoder..."
python3 code/stage_o_3d_parity_consensus_and_leiden.py
echo "Done. All outputs written to results_stage_o_consensus_leiden/"
