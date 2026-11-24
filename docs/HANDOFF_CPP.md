# C++ INT8 engine handoff (Muryeol → Sung woo)

팀 전체 설명은 `docs/TEAM_GUIDE.md`를 먼저 본다. 이 파일은 엔진 파일 위치만 적는다.

활성 눈금은 `artifacts/act_scales/full_int8_p999_ch.pt`다. `int8_package/manifest.json`의 `act_scales`는 최댓값 방식이라 쓰지 않는다.

## Canonical artifacts

| Path | Contents |
|------|----------|
| `runs/quant_chen/int8_package/` | `manifest.json`, `conv1..conv5/{weight_q,scale_w,bias_fp}.npy` |
| `runs/quant_chen/layer_dumps/` | `weight_only/` and `full_int8/` sample dumps |
| `docs/ROUNDING.md` | Single rounding contract |

Default export uses **QAT weight-only** backbone when you pass:

```bash
python scripts/export_int8.py \
  --checkpoint runs/qat_chen_weight_only/best.pt \
  --deploy-path weight_only_fp_act
```

Full INT8 arithmetic (RQ2) uses the same package; `act_scales` in `manifest.json` apply to `forward_full_int8`.

## Build and verify

```bash
bash engine/build.sh
python scripts/verify_cpp_engine.py --max-samples 16
python scripts/verify_cpp_engine.py --mode full_int8 --max-samples 8
```

C++ entry: `engine/build/aconvnet_int8` — args `--package`, `--input`, `--output`, `--mode weight_only|full_int8`.

## Layer bit-exact workflow

1. Muryeol: `python scripts/dump_layers.py --num-samples 4`
2. Sung woo: run C++ forward, save same tensor names under `layer_dumps/<mode>/cpp_*`
3. Compare with `np.testing.assert_allclose` per `ROUNDING.md`

Keys for **full_int8**: `input_q`, `after_conv1_q`, …, `logits` (see `quantize/int_ref.py`).

Keys for **weight_only**: `input`, `after_conv1`, …, `logits`.

## Scripts index

| Script | Role |
|--------|------|
| `scripts/run_main_path.sh` | One-shot milestone pipeline |
| `scripts/build_rq1_report.py` | RQ1 table → `runs/rq1/` |
| `scripts/dump_layers.py` | Layer dumps for C++ |
| `scripts/verify_int_ref.py` | PTQ vs NumPy ref |
| `scripts/verify_cpp_engine.py` | C++ vs NumPy ref |

ViT quantization is **out of scope** (proposal optional, dropped).

## Full INT8 status

- **C++ ↔ NumPy ref (full_int8):** logits bit-exact on test batches (`verify_cpp_engine.py`).
- **PTQ full INT8 accuracy:** ~48% after calib sweep (`runs/quant_chen/full_int8_calib.json`). Report as separate RQ1 row; RQ2 does not require high full-act accuracy.
- **Layer dumps:** `runs/quant_chen/layer_dumps/manifest.json` — weight-only ref↔PTQ logits aligned; full_int8 logits may differ from PyTorch fake-quant in float (argmax still matches on dump set).