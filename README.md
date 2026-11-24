# DL_project — SAR ATR INT8 파이프라인

UNIST CSE36501 · Chen A-ConvNet + **INT8 양자화** + **C++ 추론 (NumPy bit-exact)**

**Repository:** https://github.com/Marshall-Choi/DL_project

| 팀원 | 담당 | 지금 할 일 |
|------|------|------------|
| 무열 최 | RQ1 양자화 | **완료** (`artifacts/`) |
| 이성우 | RQ2 C++ 엔진 | `engine/` + 검증·벤치 — **데이터 불필요** |
| 임현석 | RQ3 평가 정직성 | split별 **FP32 from scratch** 재학습 — **데이터 필요** |

**팀원 상세 안내 (필독):** [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md)  
ViT는 하지 않음.

---

## 프로젝트가 말하는 이야기

1. **메인:** MSTAR SOC에서 10종 분류 CNN을 **INT8**로 줄이고(**RQ1**), **PyTorch 없이 C++**로 같은 답을 내게 한다(**RQ2**).
2. **부록(RQ3):** 「98%」가 **쉬운 train/test 나누기** 때문에 부풀지 않았는지 확인한다.  
   → 현석은 **양자화 안 함.** split만 바꿔 **FP32를 처음부터 다시 학습**해 표를 만든다.  
   → 무열이 준 `.pt`로 **이어 학습하는 게 아님** (공식 98.64%는 **비교 기준 숫자**).

```
RQ1 (무열) INT8 정확도     ─┐
RQ2 (성우) C++ 속도·일치   ─┼─ 메인 제출
RQ3 (현석) split FP32 표   ─┘   「SOC 점수가 정직한가」
```

---

## 결과 요약 (공식 SOC, seed 12321)

표: [`artifacts/reports/RQ1_TABLE.md`](artifacts/reports/RQ1_TABLE.md)

| 설정 | test acc |
|------|----------|
| FP32 | **98.64%** |
| INT8 weight-only PTQ/QAT | ~98.6–98.72% |
| INT8 full (채널별 p99.9 활성 스케일) | **98.72%** |
| INT8 full (max 활성 스케일, 구버전) | ~48% — 사용 금지 |

---

## 30초 시작

```bash
git clone git@github.com:Marshall-Choi/DL_project.git && cd DL_project
```

| 역할 | 다음 스텝 |
|------|-----------|
| **성우** | `docs/HANDOFF_CPP.md` → `bash engine/build.sh` → `python scripts/verify_cpp_engine.py` |
| **현석** | `docs/TEAM_GUIDE.md` 「임현석」절 → split 데이터 준비 → `scripts/train_baseline.py` |
| **보고** | `artifacts/reports/RQ1_TABLE.md` + 각자 RQ2/RQ3 표 |

---

## 저장소 구조

```text
artifacts/     모델·INT8·스케일·RQ1 표 (Git 포함)
docs/          TEAM_GUIDE, SHARING, ROUNDING, HANDOFF_CPP
engine/        C++ 추론
quantize/      양자화·NumPy ref
scripts/       학습·검증
models/, mstar/
```

`processed/`·`data/`(학습용 ~9GB)는 **Git 없음**. 현석만 필요 시 서버 경로 또는 `docs/SHARING.md`.

---

## `artifacts/` 핵심

| 경로 | 용도 |
|------|------|
| `checkpoints/fp32_chen_seed12321.pt` | 공식 FP32 (98.64%) — **현석 비교용, fine-tune용 아님** |
| `checkpoints/qat_weight_only.pt` | QAT weight-only |
| `int8_package/` | C++ INT8 conv 가중치 |
| `act_scales/full_int8_p999_ch.pt` | full INT8 활성 스케일 (**manifest.json act_scales 대신 이것**) |

---

## 연구 질문 ↔ 문서

| RQ | 문서 |
|----|------|
| RQ1 | `artifacts/reports/`, `quantize/` |
| RQ2 | `docs/HANDOFF_CPP.md`, `engine/` |
| RQ3 | `docs/TEAM_GUIDE.md` (임현석 절) |

---

## 참고

Chen et al. TGRS 2016 · Jacob et al. CVPR 2018 · MSTAR Public Release (AFRL SDMS)