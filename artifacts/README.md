# 공식 산출물 (팀 공유용)

GitHub `artifacts/` + 소스에 모델이 포함되어 있다. **9GB 학습 데이터는 Git에 없음** — RQ3(현석)만 `processed/` 필요. C++(성우)는 **데이터 없이** clone만으로 작업 가능. 팀 역할: `docs/TEAM_GUIDE.md`.

## 체크포인트

| 파일 | 용도 |
|------|------|
| `checkpoints/fp32_chen_seed12321.pt` | 공식 FP32 baseline (98.64%) |
| `checkpoints/qat_weight_only.pt` | QAT weight-only (98.72%), INT8 export 권장 백본 |

## INT8 (C++ / NumPy ref)

| 경로 | 용도 |
|------|------|
| `int8_package/` | conv1–5 `weight_q.npy`, `scale_w.npy`, `bias_fp.npy` |
| `act_scales/full_int8_p999_ch.pt` | **활성 스케일 (채널별 p99.9)** — full INT8 98.72%용 |
| `../docs/ROUNDING.md` | 반올림·정수 conv 규칙 |

`int8_package/manifest.json`의 `act_scales`는 구버전(max-abs)이다. **엔진은 `act_scales/full_int8_p999_ch.pt`만 사용.**

## 보고용

| 파일 | 내용 |
|------|------|
| `reports/RQ1_TABLE.md` | RQ1 정확도 표 |
| `reports/full_int8_recovered_summary.json` | full INT8 캘리브 스윕 결과 |

## 데이터 (이 repo 밖)

Chen SOC 학습/테스트:

`processed/aconvnet_patches_chen/` (약 9.2GB, 서버 `/home/asdl/DL_project/processed/aconvnet_patches_chen`)

재학습:

```bash
python scripts/train_baseline.py --chen-repro \
  --data-dir processed/aconvnet_patches_chen \
  --out-dir runs/baseline_chen_soc --device cuda
```