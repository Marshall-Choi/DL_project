# DL_project — SAR ATR INT8 파이프라인

UNIST CSE36501 (Deep Learning, 2026) 팀 프로젝트  
**목표:** MSTAR SOC 차량 **10-class 분류**를 Chen A-ConvNet으로 학습한 뒤 **INT8 양자화(PTQ/QAT)** 하고, **NumPy 정수 레퍼런스**와 **프레임워크 없는 C++ 추론**이 같은 답을 내는지 검증한다.

| 팀원 | 역할 |
|------|------|
| 무열 최 (20201373) | 양자화, NumPy ref, 실험·산출물 정리 |
| 임현석 (20211281) | MSTAR/SOC 데이터, FP32, RQ3 validity |
| 이성우 (20221447) | C++ INT8 엔진, bit-exact, 벤치마크 |

제안서: `Proposal_Written.tex` / ViT 비교는 **범위에서 제외(optional drop)**.

---

## 한눈에 보는 결과 (Chen SOC, seed 12321)

공식 표: `artifacts/reports/RQ1_TABLE.md`

| 방법 | 활성 | 테스트 정확도 |
|------|------|----------------|
| FP32 | — | **98.64%** |
| PTQ, 가중치만 (per-channel) | FP32 | 98.64% |
| PTQ, 가중치만 (per-tensor) | FP32 | 98.60% |
| QAT, 가중치만 | FP32 | **98.72%** |
| PTQ, 가중치+활성 (채널별 p99.9 스케일) | INT8 | **98.72%** |
| PTQ, 활성 스케일 = 전체 최댓값 (구버전) | INT8 | ~48% (사용 금지) |

**RQ1 요약:** INT8로도 Chen baseline과 같은 수준을 유지할 수 있다. 가중치만 양자화해도 손실이 거의 없고, **활성까지 INT8**로 넣을 때는 **채널마다 99.9% 백분위로 스케일을 잡는 것**이 필수다(크기 채널 스펙클 피크 때문에 max 스케일은 대부분 픽셀을 0으로 만든다).

**RQ2 요약:** C++ 초안(`engine/`)과 NumPy ref는 **full INT8 경로에서 로짓 bit-exact**까지 확인됨. 엔진은 **채널별 활성 스케일** 반영이 남은 작업(성우).

---

## 저장소 구조

```text
DL_project/
├── README.md              ← 이 파일
├── artifacts/             ← Git에 포함: 모델·INT8·스케일·RQ1 표
├── docs/
│   ├── TEAM_GUIDE.md      ← 팀원별 할 일 (자세함)
│   ├── SHARING.md         ← 클론·데이터·모델 전달
│   ├── ROUNDING.md        ← C++와 NumPy가 맞출 반올림 규칙
│   └── HANDOFF_CPP.md     ← C++ 인수 요약
├── models/                A-ConvNet (PyTorch)
├── mstar/                 데이터 로더·SOC 전처리
├── quantize/              fake-quant, PTQ/QAT, NumPy ref
├── scripts/               학습·양자화·검증 스크립트
└── engine/                C++ INT8 추론 (v1)
```

로컬 전용(Git 없음): `processed/`, `data/` — **분할 실험·재학습할 때만** 필요 (`docs/SHARING.md`).

---

## 빠른 시작

### 1) 코드 받기

```bash
git clone git@github.com:Marshall-Choi/DL_project.git
cd DL_project
```

**모델·INT8 가중치는 `artifacts/`에 포함**되어 있다. **clone만 하면 C++·ref 검증에 데이터는 필요 없다.**

| 누가 | 9GB 학습 데이터 필요? |
|------|------------------------|
| **성우 (C++)** | **아니오** — `artifacts/` + `engine/` + (선택) 테스트 2425장만 전체 정확도 재고 싶을 때 |
| **현석 (RQ3)** | **예** — split/재학습 실험용 (`processed/` 또는 원본 `data/`) |
| **무열** | 이미 끝 — 재실험할 때만 |

### 2) 환경 (C++만 할 때)

C++만이면: `g++`, NumPy 있는 Python(검증 스크립트용) 정도면 된다.

PyTorch까지 쓰려면:

```bash
pip install -r requirements.txt
```

### 3) 대표 명령

| 목적 | 명령 |
|------|------|
| C++ 빌드·검증 (데이터 불필요) | `bash engine/build.sh` → `python scripts/verify_cpp_engine.py --mode full_int8` |
| FP32 재학습 (현석 등, **데이터 필요**) | `python scripts/train_baseline.py --chen-repro --data-dir … --device cuda` |
| full INT8 스케일 (이미 `artifacts/`에 있음) | `python scripts/recover_full_int8.py` |
| INT8 export (재실험 시) | `python scripts/export_int8.py --checkpoint artifacts/checkpoints/qat_weight_only.pt` |
| NumPy vs PTQ (데이터 있을 때) | `python scripts/verify_int_ref.py` |

---

## `artifacts/` (모델·양자화 산출물)

| 경로 | 설명 |
|------|------|
| `checkpoints/fp32_chen_seed12321.pt` | FP32 best |
| `checkpoints/qat_weight_only.pt` | QAT weight-only, export 백본 권장 |
| `int8_package/` | C++·ref용 INT8 conv 가중치 (`weight_q`, `scale_w`, `bias_fp`) |
| `act_scales/full_int8_p999_ch.pt` | **full INT8 활성 스케일 (채널별 p99.9)** — 98.72% |
| `reports/RQ1_TABLE.md` | 제출용 정확도 표 |

**주의:** `int8_package/manifest.json` 안의 `act_scales`는 예전 max-abs 방식이다. **C++ full INT8은 `full_int8_p999_ch.pt`만 사용**한다. 자세한 규칙은 `docs/ROUNDING.md`.

---

## 모델·입력 규격 (보고서용)

- MSTAR **SOC** 10클래스, 우리 실험은 **17° train / 15° test** (이미 학습된 가중치로 고정)
- 입력 **2×88×88** (magnitude + phase), A-ConvNet 5 conv

---

## 연구 질문별 문서

| RQ | 내용 | 문서/산출물 |
|----|------|-------------|
| RQ1 정확도 | FP32 vs PTQ vs QAT vs full INT8 | `artifacts/reports/`, `scripts/recover_full_int8.py` |
| RQ2 효율·정확성 | C++ 엔진, bit-exact, 벤치(진행 중) | `engine/`, `docs/HANDOFF_CPP.md` |
| RQ3 validity | split·size cue (진행 중) | `docs/TEAM_GUIDE.md` 현석 파트 |

---

## 팀원에게

**성우:** `docs/HANDOFF_CPP.md` + `docs/ROUNDING.md`  
**현석:** `docs/TEAM_GUIDE.md` (split 실험 — 데이터 필요)  
9GB 데이터 받는 법은 **현석만** `docs/SHARING.md` 참고.

질문·이슈는 GitHub Issues 또는 팀 채널로.

---

## 참고

- Jacob et al., CVPR 2018 — QAT / fake-quant  
- Chen et al., IEEE TGRS 2016 — A-ConvNet SAR SOC