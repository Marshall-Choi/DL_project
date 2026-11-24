#!/usr/bin/env bash
# Proposal milestone (Nov 15–28): INT8 package + NumPy ref + C++ v1 + verify
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.conda/bin/python}"
export PYTHON

cd "$ROOT"
echo "== export INT8 package (QAT weight-only backbone) =="
"$PYTHON" scripts/export_int8.py \
  --checkpoint runs/qat_chen_weight_only/best.pt \
  --deploy-path weight_only_fp_act

echo "== full INT8 PTQ calibration sweep =="
"$PYTHON" scripts/eval_full_int8.py

echo "== layer dumps for C++ =="
"$PYTHON" scripts/dump_layers.py --num-samples 4

echo "== NumPy ref vs PTQ =="
"$PYTHON" scripts/verify_int_ref.py --max-samples 64

echo "== build C++ engine =="
bash engine/build.sh

echo "== C++ vs NumPy ref (weight-only deploy path) =="
"$PYTHON" scripts/verify_cpp_engine.py --max-samples 16

echo "== C++ vs NumPy ref (full INT8 arithmetic path) =="
"$PYTHON" scripts/verify_cpp_engine.py --mode full_int8 --max-samples 8

echo "== RQ1 report =="
"$PYTHON" scripts/build_rq1_report.py

echo "Done. See runs/rq1/RQ1_TABLE.md, runs/quant_chen/, docs/HANDOFF_CPP.md"