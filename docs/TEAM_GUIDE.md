# 팀 작업 안내

**먼저 읽기 (본인 이름만 골라도 됨)**

| 누가 | 읽을 곳 |
|------|---------|
| 성우 | 아래 「프로젝트 큰 그림」→ 「이성우: C++」전체 |
| 현석 | 아래 「프로젝트 큰 그림」→ 「임현석: RQ3」전체 |
| 무열 | 「이미 끝난 것」— 추가 작업 최소 |

자세한 파일 위치·클론: `README.md`, `docs/SHARING.md`, `docs/HANDOFF_CPP.md`.

---

## 프로젝트 큰 그림 (세 사람 공통)

**메인 주제는 양자화 + C++ 효율**이다. 제안서는 이걸 세 줄로 나눈다.

| RQ | 질문 | 담당 | 상태 |
|----|------|------|------|
| **RQ1** | INT8로 정확도가 얼마나 떨어지나, QAT가 PTQ보다 나은가 | 무열 | **완료** (~98.7%) |
| **RQ2** | C++ INT8 엔진이 빠르고, NumPy와 같은 답을 내는가 | 성우 | **진행 중** |
| **RQ3** | train/test 나누기를 잘못하면 점수가 부풀지 않는가 | 현석 | **진행 중** |

**RQ3는 양자화랑 따로 노는 게 아니다.**  
우리가 말하는 「FP32 98.64%, INT8 98.72%」는 **공식 MSTAR SOC(17° 학습 / 15° 시험)**에서 나온 숫자다. Random split처럼 **쉬운 나누기**면 90%대가 넘어가 **속일 수 있다**. 현석 일은 **「우리 메인 점수가 까다로운 설정에서 나왔다」**고 보고서에서 말할 근거를 만드는 것이다.

```
메인 스토리:  FP32 baseline → INT8 (RQ1) → C++ 엔진 (RQ2)
                          ↑
              이 baseline 숫자가 정직한지 (RQ3, FP32만 재학습)
```

---

## 먼저 알 단어

| 말 | 뜻 |
|----|----|
| **추론 (inference)** | 이미 학습된 가중치로 **새 사진 분류만** 하는 것. 에폭·역전파 없음. |
| FP32 | 소수점 학습·추론. 기준 정확도. |
| INT8 | -128~127 정수로 줄이기. 용량·연산 감소. |
| PTQ / QAT | 학습 후 양자화 / 양자화 넣고 추가 학습. |
| bit-exact | C++ 출력과 NumPy ref 로짓이 완전히 같은 것. |
| **SOC split** | 17° train, 15° test — Chen 벤치마크 **공식** 나누기. |

공식 테스트 기준: **98.64%** (2425장, seed 12321).

---

## 이미 끝난 것 (무열)

양자화·ViT 제외 실험 완료. 공식 표: `artifacts/reports/RQ1_TABLE.md`.

| 방법 | 테스트 정확도 |
|------|----------------|
| FP32 | 98.64% |
| PTQ weight-only (per-ch) | 98.64% |
| QAT weight-only | 98.72% |
| PTQ full INT8 (채널별 p99.9) | 98.72% |
| PTQ full INT8 (max 스케일, 구버전) | 47.75% — **사용 금지** |

---

## 이성우 — C++ (RQ2)

### 이 역할이 뭔지

**추론 프로그램** = PyTorch 없이, **사진 1장 → 10개 로짓(종류 점수)** 을 내는 `engine/build/aconvnet_int8` 실행 파일.  
**다시 학습하지 않는다.** `artifacts/`에 있는 INT8 가중치를 읽어 conv·ReLU·pool만 C++로 돌린다.

난이도: 범용 딥러닝 프레임워크가 아니라 **conv 5개짜리 전용 계산기**. 뼈대는 `engine/`에 있음.  
AI로 코드 짜도 되지만 **완료 조건은 `scripts/verify_cpp_engine.py` 통과**다.

### 데이터 필요? **아니오**

`git clone` + `artifacts/`면 bit-exact·벤치(샘플) 가능. 전체 test 2425장 정확도까지 보려면 그때만 `test_x.npy` 필요.

### 받을 파일

| 파일 | 내용 |
|------|------|
| `artifacts/int8_package/conv1`…`conv5` | `weight_q.npy`, `scale_w.npy`, `bias_fp.npy` |
| `artifacts/act_scales/full_int8_p999_ch.pt` | **활성 눈금 (채널별 p99.9)** — 98.72% 설정 |
| `docs/ROUNDING.md` | 반올림·정수 conv 규칙 |
| `engine/`, `quantize/int_ref.py` | C++ 초안·파이썬 정답 |

**주의:** `int8_package/manifest.json`의 `act_scales`는 옛날 max-abs(47%대). **엔진은 `full_int8_p999_ch.pt`만 쓴다.**

### 할 일 순서

1. `bash engine/build.sh`
2. `python scripts/verify_cpp_engine.py --max-samples 32` (weight-only)
3. C++에 **채널별 활성 스케일** 반영 → `verify_cpp_engine.py --mode full_int8`
4. (선택) test 전체로 ~98.72% 확인, **지연·메모리·모델 크기** 표

### 하지 않는 것

모델 학습, `quantize/` 수정, ViT/SIMD/보드 포팅(일정 밀리면 생략).

---

## 임현석 — RQ3 (평가 정직성)

### 이 역할이 뭔지 (왜 양자화 안 하냐)

**목적:** 「98%」가 **모델·INT8 성능**인지, **쉬운 시험지(나쁜 split)** 때문인지 가른다.  
**방법:** split만 바꿔서 **FP32 A-ConvNet을 매번 처음부터 학습**하고 테스트 정확도를 표로 만든다.

| 하지 않는 것 | 하는 것 |
|--------------|---------|
| INT8, QAT, C++, `quantize/` 수정 | 데이터 train/test **나누기** + `train_baseline.py` + **정확도 표** |
| 무열이 준 `.pt`로 **이어서 학습 (fine-tune)** | split마다 **가중치 랜덤 초기화 → 새로 학습** |

`artifacts/checkpoints/fp32_chen_seed12321.pt`는 **공식 SOC 98.64% 비교 숫자**일 뿐, random/block 실험에 **불러 쓰지 않는다**.

### 제안서가 의심하는 두 가지

1. **각도 누수:** train·test에 비슷한 각도가 섞이면 점수가 **실제보다 높게** 나온다.
2. **크기 단서:** 클래스마다 칩 픽셀 크기가 다르면 **차량이 아니라 크기**로 맞힌다.  
   우리 메인은 **88×88 크롭**으로 크기 단서를 줄인 상태다.

### 만들 표 (예시)

| train / test 나누기 | 88×88 크롭 | test acc | 비고 |
|---------------------|------------|----------|------|
| **공식 SOC** 17° / 15° | 예 | **98.64%** | 무열 결과, **다시 학습 안 해도 됨** |
| **Random** (칩 무작위 train/test) | 예 | 측정 | 공식보다 **많이 높으면** 누수 설명 |
| **Block** (연속 인덱스 덩어리) | 예 | 측정 | random과 비슷한 경고 |
| 공식 SOC | **아니오** (원본 칩 크기) | (선택) | size cue 실험 |

### 학습 방법 (split마다 **from scratch**)

1. `processed/...` 또는 새로 만든 폴더에 `train_x.npy`, `train_y.npy`, `test_x.npy`, `test_y.npy`, `config.json` 준비.  
   Random/block 폴더는 **아직 없음** — 칩 단위로 train/test 재배치하면 됨.
2. **매 실험마다** 아래 실행 (체크포인트 로드 없음):

```bash
python scripts/train_baseline.py --chen-repro \
  --data-dir processed/너가_만든_폴더 \
  --out-dir runs/baseline_실험이름 \
  --seed 12321 \
  --device cuda
```

3. 보고서에 한 줄:  
   「Random/block에서 acc가 SOC보다 크면, 평가가 느슨해 과대평가될 수 있다. **RQ1·2는 공식 SOC 위에서 수행했다.**」

### 데이터 필요? **예 (현석만)**

약 9GB `processed/aconvnet_patches_chen` 또는 팀 서버 경로. **GitHub에는 데이터 없음.** `docs/SHARING.md`.

칩 크기 실험 시간 없으면 **88 크롭 + SOC vs random**만 해도 RQ3 핵심은 된다.

---

## 서로 건드리지 않을 것

- **성우:** `quantize/`·학습 코드 수정 X. `artifacts/` + `ROUNDING.md` 기준으로 `engine/`만.
- **현석:** `engine/`, `quantize/` 수정 X. split·FP32 학습·표만.
- **무열**이 확정한 RQ1 숫자(98.64 / 98.72 / 47.75)를 다른 실험으로 **덮어쓰지 말 것**.

---

## GitHub / 환경

- Repo: https://github.com/Marshall-Choi/DL_project  
- 모델·INT8: `artifacts/` (clone에 포함)  
- 같은 서버: `conda activate /home/asdl/DL_project/.conda` (선택)