## CODI + ISAC/MoLSAKI 실험 실행 가이드

이 문서는 이 레포(CODI 원본 + ISAC 확장)를 처음부터 빌드하고, 학습을 돌리고(단일 GPU / 다중 GPU DDP), 테스트하고, 문제가 생겼을 때 뭘 봐야 하는지까지 한 번에 정리한 실전 가이드입니다. 구조/수식/설계 근거는 `CLAUDE.md`, `ISAC.md`를 참고하고, 이 문서는 "터미널에 뭘 쳐야 하는가"에 집중합니다.

**이 문서는 특정 서버/계정에 종속되지 않도록 일반화되어 있습니다.** 아래 값들은 각자 환경에 맞게 바꿔서 쓰세요:

| 플레이스홀더 | 의미 | 예시 |
|---|---|---|
| `<SERVER_HOST>` | 원격 GPU 서버 주소 | `cs.gs.hs.kr`, `123.45.67.89`, `myserver.example.com` |
| `<SERVER_USER>` | 서버 계정명 | `gs24110` |
| `<SERVER_REPO_PATH>` | 서버에서 이 레포가 있는 절대/홈 기준 경로 | `/src/gs24110`, `~/CODI` |
| `<LOCAL_REPO_PATH>` | 로컬(맥/리눅스) 머신에서 이 레포가 있는 경로 | `~/Downloads/ISAC_implementation` |
| `<CONDA_ENV>` | conda 환경 이름 (자유롭게 지어도 됨) | `codi` |
| `<GPU_IDX>` | 실제로 비어있는 GPU 번호 (`nvidia-smi`로 확인) | `0`, `3`, `3,4,5,6` |

서버가 아니라 로컬 머신에서 바로 작업한다면 1~2번(SSH 접속, rsync)은 건너뛰고 나머지는 그대로 로컬 셸에서 실행하면 됩니다.

**목차**: [1. 환경 빌드](#1-환경-빌드) · [2. 파일 동기화](#2-로컬--서버-파일-동기화) · [3. teacher attention 캐시](#3-isac용-teacher-attention-캐시-만들기-한-번만-하면-됨) · [4. 학습 스크립트 종류](#4-학습-스크립트-종류) · [5. 학습(GPU 1개씩 병렬)](#5-학습-실행---방법-a-gpu-1개씩-여러-실험-병렬) · [6. 학습(DDP)](#6-학습-실행---방법-b-다중-gpu-ddp로-모델-하나만) · [7. 하이퍼파라미터](#7-학습-관련-초기값하이퍼파라미터-바꾸는-법) · [8. 테스트](#8-테스트평가-방법) · [9. 모니터링](#9-진행-상황-모니터링) · [10. 에러 표](#10-자주-나는-에러와-확인-순서) · [11. 체크포인트](#11-체크포인트재시작-관리) · [12. 실험 설계](#12-실험-설계-참고) · [13. 새로 빌린(렌탈) GPU 머신 체크리스트](#13-새로-빌린렌탈-gpu-머신-체크리스트)

---

### 1. 환경 빌드

원격 서버를 쓴다면 먼저 접속:

```bash
ssh <SERVER_USER>@<SERVER_HOST>
```

그 다음 (서버든 로컬이든 동일):

```bash
conda create --name <CONDA_ENV> python=3.12
conda activate <CONDA_ENV>
cd <SERVER_REPO_PATH>      # 로컬 작업이면 이 레포의 로컬 경로
pip install -r requirements.txt
```

HuggingFace 인증 (Llama-3.2-1B-Instruct 등 gated 모델을 쓸 경우 필수):

```bash
huggingface-cli login
```
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)에서 발급받은 토큰 붙여넣기. gated 모델(예: `meta-llama/Llama-3.2-1B-Instruct`) 페이지에서 라이선스 동의도 미리 해둬야 함 (계정 단위라 한 번만 하면 됨).

GPU 확인:

```bash
nvidia-smi
```

여러 명이 같이 쓰는 서버라면 `Memory-Usage`가 거의 0인 GPU 번호를 찾아서 그 번호로만 작업해야 함 (`CUDA_VISIBLE_DEVICES=<GPU_IDX>`). 이미 다른 사람이 쓰고 있는 GPU를 잡으면 그 사람 작업이 죽거나 내 쪽이 OOM 남. 혼자 쓰는 머신이면 이 확인은 생략 가능하지만, GPU가 여러 개면 어떤 걸 쓸지는 여전히 명시하는 게 안전함(HF `Trainer`의 자동 GPU 감지가 예상과 다르게 동작할 수 있음, 5번 참고).

---

### 2. 로컬 ↔ 서버 파일 동기화

원격 서버를 쓸 때만 해당. **반드시 새 로컬 터미널(서버 세션 안이 아닌)에서** 실행:

```bash
rsync -av <LOCAL_REPO_PATH>/ <SERVER_USER>@<SERVER_HOST>:<SERVER_REPO_PATH>/
```

파일 하나만 옮길 때 (예: `train.py`만 수정했을 때):

```bash
rsync -av <LOCAL_REPO_PATH>/train.py <SERVER_USER>@<SERVER_HOST>:<SERVER_REPO_PATH>/train.py
```

`src/` 하위 파일은 경로 그대로 맞춰서 보내야 함:

```bash
rsync -av <LOCAL_REPO_PATH>/src/model.py <SERVER_USER>@<SERVER_HOST>:<SERVER_REPO_PATH>/src/model.py
```

결과물(로그, PNG 등)을 로컬로 가져올 때는 반대 방향:

```bash
scp <SERVER_USER>@<SERVER_HOST>:~/codi_ckpt/<경로>/loss_plot.png ~/Downloads/
```

로컬에서 직접 작업하는 경우 이 단계 전체가 필요 없음 — 파일을 바로 수정하고 바로 실행하면 됨.

---

### 3. ISAC용 teacher attention 캐시 만들기 (한 번만 하면 됨)

ISAC/MoLSAKI 학습(`use_att_loss True`)은 `att_cache_dir`에 미리 캐싱된 teacher 모델의 attention이 있어야 동작함. 이미 캐시가 만들어져 있다면 이 단계는 생략. 새로 만들어야 한다면 (아래는 teacher로 Qwen2.5-7B-Instruct를 쓰는 예시 — 실제로는 자신의 `ISAC.md`/실험계획에서 정한 teacher 모델로 바꿀 것):

```bash
# 소규모 시험
python cache_teacher_attention.py \
    --teacher_model_name_or_path <TEACHER_MODEL_NAME> \
    --att_cache_dir <ATT_CACHE_DIR> \
    --max_examples 50

# meta.json 확인 (num_cached가 0보다 크고 50 근처인지)
cat <ATT_CACHE_DIR>/meta.json

# 문제 없으면 전체 빌드 (--max_examples 빼고 그대로)
python cache_teacher_attention.py \
    --teacher_model_name_or_path <TEACHER_MODEL_NAME> \
    --att_cache_dir <ATT_CACHE_DIR>
```

`<TEACHER_MODEL_NAME>`은 HuggingFace 모델 ID(예: `Qwen/Qwen2.5-7B-Instruct`), `<ATT_CACHE_DIR>`은 캐시를 저장할 아무 경로(예: `~/att_cache/<teacher이름>`). teacher 모델이 크면(7B 이상) 로컬 GPU 메모리가 부족할 수 있음 — 그럴 땐 메모리가 더 큰 서버에서 이 단계만 따로 수행.

캐시 커버리지 확인은 항상 이걸로:

```bash
cat <ATT_CACHE_DIR>/meta.json
```
`num_cached`가 실제로 쓸 학습셋 크기에 가까운지 확인. 작으면(트라이얼만 해놓은 상태면) 학습 중 `L_att`가 캐시 없는 예제를 많이 스킵하게 됨 (아래 7-D 참고).

이 캐시는 `num_latent`(즉 ISAC vs MoLSAKI-only)와 무관하게 재사용 가능 — 같은 teacher로 여러 실험(다른 student 모델, ISAC/MoLSAKI 등)을 돌릴 거라면 `att_cache_dir` 하나만 만들어두고 계속 재사용하면 됨.

---

### 4. 학습 스크립트 종류

`scripts/` 밑에 GSM8K-Aug-NL(`icot-full`) 학습 스크립트들이 있음 (ISAC/MoLSAKI는 ISAC.md §3.2 규정상 이 데이터셋만 지원). 이 레포엔 기본적으로 4개 조합(모델 2종 × 설정 2종)이 준비돼 있음:

| 스크립트 | 모델 | 설정 |
|---|---|---|
| `train_gpt2_gsm8k-aug-nl_isac.sh` | gpt2 | ISAC (`num_latent 6`) |
| `train_llama1b_gsm8k-aug-nl_isac.sh` | Llama-3.2-1B-Instruct | ISAC (`num_latent 6`) |
| `train_gpt2_gsm8k-aug-nl_molsaki.sh` | gpt2 | MoLSAKI-only 베이스라인 (`num_latent 0`) |
| `train_llama1b_gsm8k-aug-nl_molsaki.sh` | Llama-3.2-1B-Instruct | MoLSAKI-only 베이스라인 (`num_latent 0`) |

각 스크립트 상단의 `SAVE_DIR=...`에 체크포인트 저장 경로가 정의돼 있음(스크립트 열어서 확인).

이 4개는 서로 다른 모델/설정이라 **비교실험용**임. 하나로 합쳐서 학습할 수 없음(모델 구조 자체가 다름). 여러 GPU가 있다면 모델 하나를 여러 GPU가 나눠 학습하는 것(DDP, 6번)과, 서로 다른 실험을 GPU별로 병렬로 돌리는 것(5번)은 별개의 선택지임 — 뭘 원하는지에 따라 골라 쓸 것.

---

### 5. 학습 실행 — 방법 A: GPU 1개씩, 여러 실험 병렬

여러 실험(다른 모델/설정)을 동시에 각자 GPU 하나씩 써서 돌리고 싶을 때. GPU 번호는 `nvidia-smi`로 비어있는 걸 매번 확인하고 바꿔서 쓸 것 (아래 `<GPU_IDX_N>`은 예시일 뿐, 실제 비어있는 번호로 교체):

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX_1> nohup bash scripts/train_gpt2_gsm8k-aug-nl_isac.sh > /tmp/gpt2_isac.log 2>&1 &
disown
sleep 5
CUDA_VISIBLE_DEVICES=<GPU_IDX_2> nohup bash scripts/train_llama1b_gsm8k-aug-nl_isac.sh > /tmp/llama_isac.log 2>&1 &
disown
sleep 5
CUDA_VISIBLE_DEVICES=<GPU_IDX_3> nohup bash scripts/train_gpt2_gsm8k-aug-nl_molsaki.sh > /tmp/gpt2_molsaki.log 2>&1 &
disown
sleep 5
CUDA_VISIBLE_DEVICES=<GPU_IDX_4> nohup bash scripts/train_llama1b_gsm8k-aug-nl_molsaki.sh > /tmp/llama_molsaki.log 2>&1 &
disown
```

GPU가 1개뿐이면 위 4개 명령을 동시에 띄우지 말고 하나씩 순서대로(앞 실험이 끝난 뒤 다음 실험) 실행할 것.

**주의점**:
- `&`로 백그라운드에 던질 땐 `tee`(파이프)보다 `>` 리다이렉트가 안전함 — `tee`를 백그라운드에서 쓰면 터미널 세션이 끊겼을 때 파이프가 죽으면서 프로세스도 같이 죽는 경우가 있음.
- 여러 개를 동시에(간격 없이) 띄우면 HF Hub 캐시 락/모델 다운로드 경합으로 조용히 죽는(로그가 텅 빈 채로 종료코드만 남는) 경우가 있었음 → 각 명령 사이에 `sleep 5` 정도 간격을 두는 게 안전.
- `CUDA_VISIBLE_DEVICES`를 아예 안 주면 HF `Trainer`가 머신에서 보이는 GPU 전부를 `nn.DataParallel`로 자동으로 물려고 시도함. GPU가 많은 멀티유저 서버에서 이게 `RuntimeError: CUDA error: peer mapping resources exhausted`로 이어지는 걸 실제로 겪었음(드라이버/커널 조합에 따라 GPU를 한 프로세스가 많이 잡으려 하면 peer mapping 한도를 넘음). GPU가 1~2개뿐인 개인 머신이면 안 겪을 수도 있지만, **어느 환경이든 `CUDA_VISIBLE_DEVICES`를 명시하는 습관을 들이는 게 안전함.**

---

### 6. 학습 실행 — 방법 B: 다중 GPU DDP로 모델 하나만

"모델 하나를 GPU 여러 개가 데이터 나눠서 학습" 방식(분산 데이터 병렬, DDP). 원한다면 어떤 학습 스크립트도 이 방식으로 바꿀 수 있음(`torchrun`으로 실행하도록 스크립트 안 커맨드만 바꾸면 됨). `train.py`/`src/model.py`는 텐서 device를 하드코딩하지 않고 항상 그 텐서 자신의 `.device`를 따라가도록 짜여 있어서, HF `Trainer`의 표준 `torchrun` 기반 DDP가 코드 수정 없이 그대로 동작함.

스크립트를 DDP용으로 바꾸는 방법:
1. 스크립트 마지막 실행부의 `python train.py \` → `torchrun --nproc_per_node=<GPU개수> --master_port=<임의의 빈 포트> train.py \`로 변경.
2. `--gradient_accumulation_steps`를 GPU 개수만큼 나눠서 실질 배치(`per_device_train_batch_size × GPU개수 × gradient_accumulation_steps`)를 기존과 동일하게 유지. 예: 1-GPU에서 accum 8 쓰던 걸 4-GPU에선 accum 2로.
3. `--ddp_find_unused_parameters True` 추가 (StudentMoL 등 일부 모듈이 배치에 따라 안 쓰이는 스텝이 있을 수 있어서 DDP 에러 방지용으로 필요).

실행:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX_LIST> nohup bash scripts/<수정한 스크립트>.sh > /tmp/train_ddp.log 2>&1 &
disown
```
`<GPU_IDX_LIST>`는 콤마로 나열한 GPU 번호들(예: `0,1,2,3`) — 반드시 `--nproc_per_node`에 지정한 개수와 일치해야 함.

**커널 버전이 낮은 서버에서는 NCCL 초기화 중 hang이 날 수 있음.** `accelerate`가 시작 시 `Detected kernel version X.X.X, which is below the recommended minimum of 5.5.0; this can cause the process to hang` 같은 경고를 띄우는 걸 봤다면, 실제로 그 경고대로 hang이 발생할 수 있음(이 프로젝트를 개발한 서버에서 실제로 겪음). 이 경고를 봤거나 DDP 실행이 원인 불명으로 멈춘다면:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=<GPU_IDX_LIST> nohup bash scripts/<수정한 스크립트>.sh > /tmp/train_ddp.log 2>&1 &
disown
```
커널 버전은 `uname -r`로 확인 가능. 이 경고가 안 뜨는 최신 커널/드라이버 환경이면 이 옵션 없이도 문제없이 돌아갈 수 있음 — 먼저 옵션 없이 시도해보고, hang이 관찰되면 그때 추가해도 됨.

---

### 7. 학습 관련 초기값(하이퍼파라미터) 바꾸는 법

전부 `scripts/train_*.sh` 안에서 `python train.py \` (또는 `torchrun ... train.py \`) 뒤에 나열된 인자들. 인자 순서는 상관없음 (`HfArgumentParser`라 파싱만 되면 됨).

#### 7-A. 학습 데이터 수 바꾸기

```
--exp_mode True \
--exp_data_num 10000 \
```
`exp_mode False`면 데이터셋 전체를 다 씀. `exp_mode True`면 `exp_data_num`개만 뽑아서 씀(빠른 검증용). **중요**: GSM8K-Aug/GSM8K-Aug-NL은 베이스 문제 하나를 ~50번씩 증강해서 이어붙인 데이터라, `exp_data_num`으로 서브샘플링할 때 앞에서부터 자르면 베이스 문제 몇 개만 반복해서 보게 됨(과적합 위험). 이 레포의 `train.py`(`SupervisedDataset.__init__`, 2026-08-28 수정)는 이미 전체 데이터셋에 고르게 stride를 둬서 뽑도록 고쳐놨기 때문에, `exp_data_num`만 바꾸면 자동으로 넓은 범위에서 균등하게 뽑힘. 별도 조치 필요 없음. (다른 데이터셋을 쓴다면 이 데이터 증강 특성이 다를 수 있으니 직접 확인할 것.)

숫자를 바꾸려면 해당 줄의 숫자만 고치면 됨:
```bash
sed -i 's/--exp_data_num 2000/--exp_data_num 10000/' scripts/<스크립트 이름>.sh
```
또는 편집기로 직접 (vim 예시):
```bash
vim scripts/<스크립트 이름>.sh
# /exp_data_num 검색(엔터) → cw로 숫자 지우고 새 값 입력 → Esc → :wq
```

#### 7-B. 총 스텝 수 계산법

```
epoch당 스텝 수 = ceil(exp_data_num / 실질배치)
총 스텝 수 = epoch당 스텝 수 × num_train_epochs
실질배치 = per_device_train_batch_size × GPU개수(DDP면) × gradient_accumulation_steps
```
예: `exp_data_num 10000`, `batch 16`, `accum 8`, GPU 1개, `epochs 40` → 실질배치 128 → epoch당 79스텝 → 총 3160스텝.

스텝 수를 늘리고 싶으면 `--num_train_epochs`를 올리거나, `--gradient_accumulation_steps`를 줄이거나(메모리 여유 있어야 함), `--exp_data_num`을 늘리면 됨.

#### 7-C. 배치/메모리 관련

```
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 8 \
```
OOM 나면 `per_device_train_batch_size`를 줄이고 `gradient_accumulation_steps`를 늘려서 실질배치를 유지하는 식으로 대응 (예: 64/2 → 16/8). ISAC의 `L_att` 경로는 배치 안 예제 하나하나에 대해 추가 forward를 돌기 때문에(`use_att_loss True`일 때), `per_device_train_batch_size`가 클수록 메모리 사용량이 거의 선형으로 늘어남 — OOM 나면 제일 먼저 의심할 부분. 이 값들의 적정 범위는 GPU 메모리 용량에 따라 다르므로, 자신의 GPU 메모리(`nvidia-smi`의 `Memory-Usage`/전체 용량)에 맞춰 작게 시작해서 점점 올려보는 게 안전함.

#### 7-D. ISAC/MoLSAKI 관련 플래그

```
--use_student_mol True \       # StudentMoL 모듈 등록 (use_att_loss 쓰려면 필수)
--use_att_loss True \          # L_att 켜기 (MoLSAKI 손실)
--teacher_model_name_or_path <TEACHER_MODEL_NAME> \   # 기록용, 실제로 로드 안 함
--att_cache_dir <ATT_CACHE_DIR> \                     # 3번에서 만든 캐시 경로
--att_loss_factor 1.0 \        # c_att (β), L_att 가중치
--mol_tau_teacher 0.1 \        # τ1
--mol_tau_student 0.5 \        # τ2
--critical_token_mode numeric \  # "numeric"만 구현됨, "keyword"는 미구현
```
`use_att_loss True`인데 `use_student_mol False`거나 `att_cache_dir`가 없으면 `CODI.__init__`에서 바로 `ValueError`로 죽음(의도된 동작).

`num_latent 0`이면 MoLSAKI-only 베이스라인(CODI의 implicit/latent path를 완전히 끔). `num_latent 6`이면 원래 ISAC(CODI 위에 L_att 얹은 버전) — 숫자는 원하는 latent 개수로 바꿀 수 있음.

#### 7-E. DDP 관련

```
--ddp_find_unused_parameters True \
```
DDP(`torchrun`)로 돌릴 때만 의미 있음. 단일 GPU에서도 있으면 무시되니 넣어놔도 무해함.

---

### 8. 테스트(평가) 방법

**주의**: `num_latent 0`(MoLSAKI-only)로 학습한 체크포인트는 `test.py`가 아니라 반드시 `test_molsaki.py`로 평가해야 함. `bot_id`/latent/`eot_id` 구조 자체를 학습 안 했기 때문에 `test.py`로 돌리면 틀린 방식으로 평가하게 됨.

#### 8-A. ISAC 체크포인트 (`num_latent 6`) → `test.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX> python test.py \
    --data_name "gsm8k" \
    --output_dir "./outputs" \
    --model_name_or_path gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 128 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir <해당 실험의 checkpoint-N 절대경로> \
    2>&1 | tee /tmp/test_run.log
```
`--model_name_or_path`/`--prj_dim`은 학습에 쓴 모델과 반드시 일치해야 함(예: Llama1b라면 `--model_name_or_path meta-llama/Llama-3.2-1B-Instruct`, `--prj_dim 2048`).

`--ckpt_dir`는 항상 실제로 존재하는 `checkpoint-N` 디렉토리를 정확히 가리켜야 함 — `ls`로 먼저 확인:
```bash
ls <SAVE_DIR>/<expt_name>/<model_name>/ep_<epochs>/lr_<lr>/seed_<seed>/
```
(경로 패턴은 11번 참고)

`--data_name`을 `svamp`, `gsm-hard`, `multi-arith`(OOD 수학 벤치마크) 등으로 바꿔서 다른 데이터셋 평가도 가능. `--inf_num_iterations`를 늘리면(기본 샘플링이 greedy가 아니면 매번 결과가 달라지므로) 여러 번 돌려서 평균 정확도를 냄. `--greedy True`면 결정적이라 1번만 돌려도 됨.

#### 8-B. MoLSAKI-only 체크포인트 (`num_latent 0`) → `test_molsaki.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX> python test_molsaki.py \
    --data_name "gsm8k" \
    --output_dir "./outputs" \
    --model_name_or_path gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 128 \
    --greedy True \
    --num_latent 0 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir <해당 실험의 checkpoint-N 절대경로> \
    2>&1 | tee /tmp/test_molsaki_run.log
```
`test.py`와 CLI 인자는 거의 같지만 `--use_prj`/`--prj_dim` 등 latent 관련 인자는 필요 없음(`num_latent 0`이라 애초에 안 씀).

#### 8-C. 결과 읽는 법

```
adapter: None | GSM8K test accuracy: 4.70% |
average length of COT: 6.260803639120546
Average accuracy over 1 sampling: 4.700530705079606
```
`GSM8K test accuracy`가 핵심 숫자. 학습 데이터 수가 적으면(수천 개 수준) 정확도가 낮게 나오는 게 정상 — 이건 파이프라인 자체 문제가 아니라 데이터 부족 문제. 실제 논문 수준 비교를 하려면 `exp_mode False`(전체 데이터)로 본 훈련을 돌려야 함.

---

### 9. 진행 상황 모니터링

#### 9-A. 실시간 로그

```bash
tail -f /tmp/train_ddp.log
```
`[CODI.init] ...` (모델 생성 단계), `[train] ...` (데이터셋/Trainer 준비 단계), `[forward] ...` (매 스텝 진행 단계: student encoder pass, teacher pass, ISAC L_att pass, latent step, ref_ce_loss 계산 등), `loss=..., ce_loss=..., ref_ce_loss=..., att_loss_total=...` (매 `logging_steps`마다), tqdm 진행바(`N/총스텝 [경과<남은시간, 초/스텝]`)가 순서대로 보이면 정상.

DDP(`torchrun`, GPU 여러 개)로 돌릴 때는 랭크 0(첫 번째 GPU)만 이 프린트들을 찍도록 이미 처리해놔서, 로그가 GPU 개수만큼 중복되지 않음.

#### 9-B. GPU 사용량

```bash
nvidia-smi
# 또는 특정 GPU만, 가볍게
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -i <GPU_IDX_LIST>
```
DDP로 여러 GPU를 쓰고 있다면 그 GPU들의 메모리/사용률이 서로 비슷해야 정상. 한쪽만 튀면 데이터가 고르게 안 나뉜 것.

#### 9-C. 프로세스 살아있는지

```bash
jobs -l                    # 같은 터미널에서 & 로 띄운 경우
ps aux | grep train.py
ps aux | grep torchrun
```

#### 9-D. Loss 추이 (CSV/그래프)

학습 중 `src/debug_utils.py`의 `LossHistoryLogger`가 체크포인트와 같은 폴더에 자동으로 남김:
```bash
tail -20 <SAVE_DIR>/.../seed_<seed>/loss_history.csv
```
그래프(PNG)는 같은 폴더의 `loss_plot.png` — 일정 스텝마다 자동 갱신됨. 원격 서버에서 로컬로 가져와서 보려면:
```bash
scp <SERVER_USER>@<SERVER_HOST>:<SAVE_DIR>/.../seed_<seed>/loss_plot.png ~/Downloads/
```
CSV만 있고 PNG가 없으면 matplotlib이 설치 안 된 것 — `pip install matplotlib`로 설치 후, 기존 CSV로부터 그래프만 재생성하고 싶으면:
```bash
python plot_loss_history.py <loss_history.csv 경로> -o loss_plot.png
```

#### 9-E. tensorboard

각 스크립트가 `--report_to tensorboard --logging_dir $SAVE_DIR/logs`로 세팅돼 있음:
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard --logdir <SAVE_DIR>/logs
```
protobuf 버전 충돌이 있는 환경이면 앞의 환경변수 없이 `TypeError: Descriptors cannot be created directly` 에러가 날 수 있음 — 에러가 나면 이 환경변수를 붙여서 재시도. 원격 서버라면 브라우저로 보기 위해 로컬에서 포트 포워딩 필요:
```bash
# 로컬 새 터미널에서
ssh -L 6006:localhost:6006 <SERVER_USER>@<SERVER_HOST>
# 그 세션 안에서 tensorboard 실행 후, 로컬 브라우저에서 localhost:6006 접속
```

---

### 10. 자주 나는 에러와 확인 순서

| 증상 | 원인 | 확인/해결 |
|---|---|---|
| `RuntimeError: CUDA error: peer mapping resources exhausted` | `CUDA_VISIBLE_DEVICES` 없이 실행해서 `Trainer`가 보이는 GPU 전부를 `DataParallel`로 잡으려 함 | 항상 `CUDA_VISIBLE_DEVICES=<GPU_IDX>` 붙여서 실행 |
| `torch.OutOfMemoryError` (특히 `ref_ce_loss` 계산 근처) | `per_device_train_batch_size`가 큼 (ISAC의 `L_att` 예제별 추가 forward가 배치 크기에 비례해 메모리 씀) | batch 줄이고 `gradient_accumulation_steps` 늘려서 실질배치 유지 |
| `RuntimeError: ... found at least two devices, cpu and cuda:0` (attention_loss.py 근처) | `compute_teacher_mol_weights`의 fallback 텐서에 `device=` 누락 (이미 수정된 버그, 사용 중인 파일이 최신인지만 확인) | `src/attention_loss.py`가 최신 버전인지 확인(원격 서버라면 동기화 재확인) |
| 로그 파일이 텅 빔 + 백그라운드 job이 즉시 종료 | 여러 학습을 동시에 띄우면서 HF 캐시/다운로드 경합 등 레이스 컨디션 | 포그라운드로 하나만 단독 실행해서 실제 에러 확인, 여러 개 띄울 땐 `sleep`으로 간격 두기 |
| `nvidia-smi`도 응답 없고 터미널이 아무 명령도 안 먹음 | 십중팔구 foreground로 띄운 학습 프로세스가 아직 shell을 붙들고 있어서 — 새 명령이 실행된 게 아니라 그냥 화면에 타이핑만 된 상태 | `Ctrl+C`로 foreground 프로세스 중단 → 프롬프트 돌아오는지 확인 → `ps aux \| grep train.py`로 좀비 프로세스 있으면 `kill -9` |
| DDP(`torchrun`) 실행 시 `[train] calling trainer.train()...` 이후 아무 로그도 안 뜨고 멈춤, `accelerate.utils.other:Detected kernel version ... can cause the process to hang` 경고가 그 위에 있음 | 사용 중인 커널이 accelerate 권장 최소버전(5.5.0)보다 낮아서 NCCL 초기화 단계에서 hang | `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`을 실행 커맨드 앞에 붙여서 재시도 (6번 참고). `uname -r`로 커널 버전 확인 가능 |
| `FileNotFoundError: ... checkpoint-N/model.safetensors (또는 pytorch_model.bin)` (test.py/test_molsaki.py) | 그 체크포인트 폴더를 지웠거나 애초에 그 스텝까지 학습이 안 감 | `ls <SAVE_DIR>/.../seed_<seed>/`로 실제 존재하는 `checkpoint-N` 확인 후 그 경로로 `--ckpt_dir` 수정 |
| `ConnectionResetError` (test.py, HF Hub 접속 중) | 일시적 네트워크 문제 | 재시도. 계속되면 `curl -I https://huggingface.co`로 네트워크 연결 확인 |
| tensorboard `TypeError: Descriptors cannot be created directly` | protobuf 버전 충돌 | `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 환경변수 붙여서 실행 |
| tmux에서 스크롤하면 화면 안 올라가고 이전에 쳤던 명령이 튀어나옴 | 마우스 모드 꺼짐 | `~/.tmux.conf`에 `set -g mouse on` 추가 후 tmux 재시작, 급하면 `Ctrl+b [`로 copy-mode 진입 |
| `use_att_loss True`인데 `L_att`이 계속 0에 가깝거나 skip이 많음 | `att_cache_dir`가 지금 쓰는 데이터 범위(특히 `exp_data_num`으로 넓게 stride 샘플링할 때)를 다 커버 못 함 | `cat <att_cache_dir>/meta.json`의 `num_cached`가 전체 데이터셋 크기에 가까운지 확인. 로그의 `att_loss_num_skipped_no_cache` 값도 같이 확인 |
| `SAVE_DIR`를 지우지 않고 재실행했더니 예전 세팅으로 이어서 학습됨(의도와 다름) | `train.py`는 같은 `output_dir`가 있으면 자동으로 최신 `checkpoint-N`에서 이어서 학습(resume)함 (의도된 기능) | 완전히 처음부터 다시 돌리고 싶으면 실행 전에 반드시 `rm -rf <SAVE_DIR>` |

기본 진단 순서는 항상: **1) 로그 맨 아래 Traceback 확인 → 2) 어느 파일의 몇 번째 줄인지 확인 → 3) 위 표에서 비슷한 증상 찾기 → 4) 없으면 `nvidia-smi`/`ps aux`로 프로세스·GPU 상태부터 확인.**

---

### 11. 체크포인트/재시작 관리

- 저장 위치 패턴: `{output_dir}/{expt_name}/{model_name}/ep_{epochs}/lr_{lr}/seed_{seed}/checkpoint-{step}` (`output_dir`은 각 스크립트 상단의 `SAVE_DIR`, 나머지는 CLI 인자값으로 자동 채워짐).
- `--save_strategy steps --save_steps 500 --save_total_limit 2`: 500스텝마다 저장, 최신 2개만 보관(오래된 건 자동 삭제). 원하는 저장 빈도로 `--save_steps` 조정 가능.
- 같은 `SAVE_DIR`로 스크립트를 다시 실행하면 `train.py`가 자동으로 최신 `checkpoint-N`부터 이어서 학습함 (모델/옵티마이저/스케줄러/RNG/스텝 수까지 전부 복원). 프로세스가 도중에 죽어도 그냥 같은 명령 다시 치면 됨.
- **완전히 새로 시작하고 싶으면 (설정을 바꿨을 때 등) 반드시 먼저 지울 것**:
  ```bash
  rm -rf <SAVE_DIR>
  ```
  (경로는 실험별로 다름 — 스크립트 상단의 `SAVE_DIR=...` 확인)

---

### 12. 실험 설계 참고

Teacher/student 모델 조합, 비교 기준선(예: CODI-only / MoLSAKI-only / ISAC) 등 구체적인 실험 계획은 이 레포의 `CLAUDE.md`/`ISAC.md`(또는 자신의 프로젝트 문서)에 따로 정리돼 있으면 그걸 참고. 이 문서는 "어떻게 실행하는가"만 다루고 "무엇을 실험할지"는 프로젝트별로 다르므로 별도 문서를 유지하는 걸 권장.

---

### 13. 새로 빌린(렌탈) GPU 머신 체크리스트

vast.ai/runpod류 서비스에서 GPU를 새로 빌렸다면, 기존에 쓰던 서버와 달리 **완전히 빈 머신**이라는 점이 중요합니다. 아래 순서로 진행하세요.

1. **이미지 선택 (CUDA/Python 버전)**: 플랫폼에서 "Torch X.Y (CUDA A, Python B)" 같은 프리셋 이미지를 고르게 되는 경우가 많은데, 이 프리셋의 torch 버전은 신경 쓸 필요 없습니다 — 어차피 아래 2번에서 `conda create`로 독립된 가상환경을 새로 만들고 `requirements.txt`에 고정된 `torch==2.7.1`(CUDA 12.6 런타임 내장)을 그 안에 따로 설치하기 때문입니다. 중요한 건 **머신에 실제로 깔린 NVIDIA 드라이버가 CUDA 12.6 이상을 지원하는지**뿐이고, 플랫폼이 "CUDA 12.8.1" 또는 "CUDA 13.0.1" 이미지를 제공한다면 둘 다 드라이버가 12.6보다 신 버전이라 문제없습니다(드라이버는 하위 호환). Python 버전도 마찬가지로 이미지의 것과 무관하게 conda가 3.12로 새로 만듭니다. 굳이 고른다면 **CUDA 12.8.1** 쪽이 더 오래 검증된 조합이라 약간 더 안전합니다.
2. **환경 빌드**: 1번 섹션 그대로 (`conda create --name codi python=3.12` → `pip install -r requirements.txt`).
3. **HuggingFace 인증**: `huggingface-cli login` — 새 머신이라 로그인 상태가 없습니다. `meta-llama/Llama-3.2-1B-Instruct` 라이선스 동의는 계정 단위라 이전에 승인했으면 다시 안 해도 됩니다.
4. **teacher attention 캐시는 처음부터 다시 만들어야 함**: 이전 서버의 `~/att_cache/qwen25_7b`는 **그 서버에만 있는 로컬 파일**이라 새 머신엔 없습니다. `scp`/`rsync`로 통째로 옮기거나(용량이 크지 않다면 이게 제일 빠름), 새 머신에서 `cache_teacher_attention.py`를 처음부터 다시 돌려야 합니다 (3번 섹션 참고).
   - **비용 절약 팁**: teacher attention 캐싱은 GPU 1개면 충분하고(멀티GPU 필요 없음) 데이터셋 전체를 한 번 훑는 오프라인 작업이라 시간이 꽤 걸립니다. 4-GPU 인스턴스를 통째로 빌린 상태에서 이 단계를 돌리면 GPU 3개가 노는 채로 과금되니, 가능하면 ①옮길 수 있으면 옮기거나 ②캐싱만 저렴한 1-GPU 인스턴스에서 먼저 끝내고 결과 파일만 4-GPU 인스턴스로 옮긴 뒤 학습을 시작하는 게 비용상 유리합니다.
5. **레포/체크포인트 저장 경로 확인**: 렌탈 인스턴스는 보통 종료 시 디스크가 날아갑니다. `SAVE_DIR`(각 스크립트 상단)가 인스턴스 종료 후에도 남는 영구 볼륨/네트워크 스토리지를 가리키는지 미리 확인하세요 — 안 그러면 학습 끝나고 인스턴스 끄는 순간 체크포인트가 통째로 날아갈 수 있습니다.
6. **GPU 4개 다 내 것**: 공유 서버와 달리 다른 사용자와 GPU를 다툴 일이 없으니, `CUDA_VISIBLE_DEVICES=0,1,2,3`로 고정해서 6번 섹션의 DDP 실행 커맨드를 그대로 쓰면 됩니다. NCCL hang 이슈(6번 섹션)는 예전 서버의 오래된 커널 때문이었던 것이라, 새로 빌린 최신 머신에서는 `NCCL_P2P_DISABLE`/`NCCL_IB_DISABLE` 없이 먼저 시도해보고, hang이 관찰될 때만 추가하세요.
7. **비용 관리**: `PYTHONUNBUFFERED=1`을 항상 붙여서 죽을 때 원인이 바로 로그에 남게 하고, `--save_steps`를 충분히 낮게(전체 스텝 수의 5% 이내 권장) 잡아서 중간에 죽어도 과금 시간 대비 손실을 최소화하세요.
