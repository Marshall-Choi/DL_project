# 팀 공유 방법 (디렉터리 정리 후)

## 1. 무엇을 GitHub에 올리나

**올림 (수 MB~수십 MB)**

- `models/`, `mstar/`, `quantize/`, `scripts/`, `engine/` (소스, `engine/build/` 제외)
- `docs/` (`TEAM_GUIDE.md`, `SHARING.md`, `ROUNDING.md`)
- `artifacts/` — 체크포인트·INT8 패키지·활성 스케일·RQ1 표
- `requirements.txt`, `Proposal_Written.tex` (선택)

**올리지 않음**

| 경로 | 이유 | 팀원이 받는 방법 |
|------|------|------------------|
| `processed/` (~10GB) | Git LFS도 부담 | **같은 서버 경로** 또는 USB/압축 전달 |
| `data/` (~1.7GB) | 원본 MSTAR | Hyunseok만 필요 시 동일 |
| `.conda/` | 환경 | `pip install -r requirements.txt` + CUDA torch |
| 루트 `.zip` | 이미 `data/`에 풀림 | 삭제함 |

`.gitignore`는 `processed/`, `runs/` 전체를 막고 **`artifacts/`만 예외**로 둔다.

## 2. 저장소 만들기 (무열이 한 번만)

```bash
cd /home/asdl/DL_project
git init
git add .gitignore models mstar quantize scripts engine docs artifacts requirements.txt Proposal_Written.tex
git commit -m "Team handoff: code + artifacts + docs"
# GitHub에 빈 repo 만든 뒤:
git remote add origin git@github.com:ORG/DL_project.git
git push -u origin main
```

팀원:

```bash
git clone git@github.com:ORG/DL_project.git
cd DL_project
conda create -n dl python=3.11 -y
conda activate dl
pip install torch numpy   # GPU CUDA 버전은 pytorch.org 안내
```

## 3. 모델을 어떻게 주나

**A. Git clone만 (대부분 충분)**

`artifacts/` 안에 있다:

- `checkpoints/fp32_chen_seed12321.pt` — FP32
- `checkpoints/qat_weight_only.pt` — QAT·export용
- `int8_package/` — C++ 가중치
- `act_scales/full_int8_p999_ch.pt` — full INT8 활성 눈금

성우: `artifacts/int8_package` + `artifacts/act_scales/full_int8_p999_ch.pt`  
현석: `artifacts/checkpoints/fp32_chen_seed12321.pt` + `processed/...` (데이터만 별도)

**B. 같은 서버 쓰는 경우 (가장 쉬움)**

코드: `git pull`  
데이터: 그대로 `/home/asdl/DL_project/processed/aconvnet_patches_chen`  
환경: `/home/asdl/DL_project/.conda` (선택)

**C. 데이터만 따로 줄 때**

```bash
# 테스트만 소량 (~300MB) — 구조 확인용
tar czf mstar_chen_test_only.tgz -C processed/aconvnet_patches_chen test_x.npy test_y.npy config.json

# 전체 학습용 (~9GB)
tar czf mstar_chen_full.tgz -C processed aconvnet_patches_chen
```

학습 전체가 필요 없으면 **테스트 npy + artifacts**만으로 C++ 검증·정확도 측정 가능.

## 4. 정리된 디렉터리 구조

```text
DL_project/
  artifacts/          ← 팀 공식 (모델·INT8·표)
  docs/               ← TEAM_GUIDE, SHARING, ROUNDING
  engine/             ← C++
  models/ quantize/ mstar/ scripts/
  processed/
    aconvnet_patches_chen/   ← Chen SOC (학습, Git 밖)
    aconvnet_128/            ← 전처리 중간 (선택, ~650MB)
    soc_88/                  ← 레거시 (필요 없으면 나중에 삭제)
  data/               ← 원본 MSTAR (Git 밖)
  runs/               ← 로컬 실험 잔여 (Git 무시)
```

## 5. 회의에서 한 줄

> “코드랑 모델은 **GitHub `artifacts/`**, 학습 데이터는 **서버 processed** 또는 압축 파일. 가이드는 `docs/TEAM_GUIDE.md`.”