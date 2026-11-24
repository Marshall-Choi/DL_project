# RQ1 — Accuracy (Chen SOC, seed 12321)

| Method | Weights | Activations | Test acc | Δ vs FP32 |
|--------|---------|-------------|----------|-----------|
| FP32 | — | FP32 | 98.64% | -0.00pp |
| PTQ | per_channel | FP32 (weight-only) | 98.64% | -0.00pp |
| PTQ | per_tensor | FP32 (weight-only) | 98.60% | -0.04pp |
| QAT | per_channel | FP32 (weight-only fake-quant) | 98.72% | +0.08pp |
| PTQ | per_channel | INT8 per-tensor max-abs (old) | 47.75% | -50.89pp |
| PTQ | per_channel | INT8 per-channel p99.9 | 98.72% | +0.08pp |

## RQ2 handoff (correctness)

- Weight-only ref verify: `/home/asdl/DL_project/runs/quant_chen/int_ref_verify/verify_report.json`
- C++ vs NumPy: `/home/asdl/DL_project/runs/quant_chen/cpp_verify/cpp_verify_report.json`

## Export

- Canonical INT8 package: `runs/quant_chen/int8_package/`
- QAT backbone for export: `/home/asdl/DL_project/runs/qat_chen_weight_only/best.pt`
- Rounding contract: `docs/ROUNDING.md`
