# V3F (정석 설계) 이식 패키지

FACE 기반 전 단계 학습 RQ 단어 ID → 정렬(align) 후 정보 소진 깊이 → 병렬 생성 추천기를 **다른 서버, 다른 데이터셋**에서 처음부터 끝까지 돌리기 위한 패키지이다.
방법론은 `docs/METHOD_FACE_VINFO_v2.md`를 보면 된다. Beauty에서 돌린 V3F와 알고리즘과 하이퍼파라미터가 같다. 바꾼 것은 경로와 데이터셋 이름을 받는 부분뿐이다.

```
cuit_v3f_portable/
  config.sh            # 서버·데이터셋별 설정 (환경변수로 덮어쓰기 가능)
  run_all.sh           # 단계 0~9 드라이버 (재개 가능, DRY_RUN=1 지원)
  docs/METHOD_FACE_VINFO_v2.md
  code/face/           # FACE (cuit1/encoder 사본: LightGCN, FACE map/align, RQ 확장, dump) + pixi.toml
  code/rec/            # 측정기·검정·ID·추천기 (CUIT_vinfo 사본: genrec, scripts, configs/base.yaml) + requirements.txt
  code/prep/           # 데이터 준비 (InATTo_1 사본: GRAM → LOO 분할, OpenAI 프로필, MiniLM 임베딩) + pixi.toml
  code/tools/          # 이 패키지용 보조 스크립트 (maps.pkl, rq_H8.pt, 일관성 검사, GRAM 변환 등)
  work/                # (실행 시 생성) prep 데이터, genrec cache, 로그, results/done.log
```

## 1. 환경

| 변수 | 용도 | 만드는 법 |
|---|---|---|
| `FACE_PY` | FACE 학습, vMF 측정기, 깊이 검정 | `cd code/face && pixi install` (Python 3.11, torch 2.8.0 cu128, transformers 4.46.0, swanlab 0.6.13, pandas 2.3.3, scipy 1.16.2, openpyxl) |
| `REC_PY` | genrec 추천기, 행동 벡터(SASRec), ID 생성 | `python3.12 -m venv code/rec/.venv && code/rec/.venv/bin/pip install -r code/rec/requirements.txt` (torch, transformers 4.56.1, accelerate 1.10.1, scikit-learn 1.7.1, scipy, faiss-cpu) |
| `PREP_PY` | 데이터 준비, OpenAI 프로필 | `cd code/prep && pixi install` (InATTo 환경: transformers, scipy, openai, python-dotenv) |

- `code/rec/requirements.txt`는 Beauty 서버 venv의 `pip freeze` 그대로다. torch는 서버 CUDA에 맞는 빌드로 바꿔 설치한다.
- GPU 사용량은 align 약 21GB, 추천기(슬롯 129개) 약 26GB, 측정기와 검정은 각각 수 GB이다. 단계는 순차로 돌므로 48GB GPU 한 장이면 된다.

## 2. 데이터셋마다 필요한 입력

1. **원시 상호작용 + 아이템 텍스트 (GRAM 형식):** `GRAM_ROOT/<Name>/user_sequence.txt`(사용자 ASIN, 시간순 아이템 ASIN), `item_plain_text.txt`(ASIN과 텍스트)
   - GRAM 릴리스가 없으면 `GRAM_FROM_RPG=1`을 쓴다. genrec 캐시(아래 3번)에서 자동으로 만든다(`code/tools/rpg_to_gram.py`).
   - Beauty 기준으로 시퀀스는 GRAM과 바이트 단위로 같다(22,363명 모두). 아이템 텍스트는 90%(10,922/12,101)가 정확히 같고, 나머지는 특수문자 처리만 조금 다르다.
   - 아이템 텍스트는 LLM 프로필 프롬프트에만 쓰인다.
2. **OpenAI API 키** (`export OPENAI_API_KEY=...`)
   - 단계 1에서 gpt-4o-mini로 사용자와 아이템의 한 문장 프로필을 만든다. 엔티티 하나당 요청 1건이며, Beauty는 약 34k건이었다.
   - 프롬프트는 IGSRec/RLMRec 형식이다(`code/prep/scripts/03_build_profile_batch.py`). 동기 API를 동시 요청 32개로 돌리고, 중단돼도 이어서 실행된다(`04b_profile_sync.py`).
   - Batch API를 쓰려면 `04_submit_profile_batch.py`와 `05_collect_profile_batch.py`를 대신 쓴다.
3. **genrec AmazonReviews2014 캐시:** 단계 2에서 snap.stanford.edu의 `reviews_<Category>_5.json.gz`와 `meta_<Category>.json.gz`를 자동으로 내려받아 처리한다.
   - 이미 처리된 캐시가 있으면 `RPG_CACHE`로 가리키면 된다. 이 서버의 `CUIT_vinfo/cache/AmazonReviews2014/`에는 Toys_and_Games, Sports_and_Outdoors, CDs_and_Vinyl이 있다.
4. **모델 가중치:**
   - `MINILM_DIR` = sentence-transformers/all-MiniLM-L6-v2 스냅샷. FACE 단어 코드북과 프로필 임베딩에 쓴다.
   - `bert-base-uncased`는 vMF 측정기용이고, HF 허브에서 받거나 HF 캐시에 미리 넣어 둔다.
   - flan-t5-xl은 **필요 없다.** sem_ids 디렉터리 이름(`flan-t5-xl_pca0`)으로만 남아 있다.

## 3. 실행

```bash
export OPENAI_API_KEY=...
DATASET=toys CATEGORY=Toys_and_Games GRAM_FROM_RPG=1 ./run_all.sh      # 전체
STAGES="3 4 5" DATASET=toys CATEGORY=Toys_and_Games ./run_all.sh        # 일부 단계만
DRY_RUN=1 DATASET=toys CATEGORY=Toys_and_Games ./run_all.sh             # 명령과 입출력 존재 여부만 출력
```
- 단계마다 출력 파일이 있으면 건너뛴다. 실패하면 즉시 멈추고 `work/logs/<단계>.log` 마지막 20줄을 보여 준다.
- 진행 기록은 `work/results/done.log`에 남는다.
- 데이터셋 키는 다음과 같다: `toys`/`Toys_and_Games`, `sports`/`Sports_and_Outdoors`, `beauty`/`Beauty`. FACE 데이터셋 이름은 `<key>_face`이다.

| 단계 | 내용 | 출력 | Beauty 실측 시간 |
|---|---|---|---|
| 0 | GRAM → LOO 분할, user2id/item2id, maps.pkl | `work/prep/data/<ds>/`, `work/data/<ds>/maps.pkl` | 수 분 |
| 1 | gpt-4o-mini 프로필 → MiniLM 임베딩 → `code/face/data/<ds>_face/` 링크 | `*_prf.pkl`, `*_emb_np.pkl` | API 속도에 따라 다름 (약 34k 요청) |
| 2 | genrec 캐시와 사용자·아이템 집합 일치 검사(불일치면 중단) | `work/data/<ds>/rpg_consistent.ok` | 수 분 (첫 다운로드 별도) |
| 3 | LightGCN | `code/face/encoder/checkpoint/lightgcn/…pth` | epoch당 약 1분 |
| 4 | FACE 원본 map (깊이 1, 우리 map의 웜스타트 출발점) | `…/lightgcn_vq/…_map.pth` | epoch당 약 1.7분 × 약 12 |
| 5 | 우리 map: 단어 코드북 RQ H=8, 모든 단계 학습 (`FACE_MAP_INIT`, `FACE_L1_FULL=1`) | `…_map_v3w.pth` | epoch당 약 4분 × 약 25 |
| 6 | 모든 단계를 켠 align (`FACE_FULL_DEPTH=1`) → 코드 dump → `rq_H8.pt` | `ART/align_codes.npz`, `ART/rq_H8.pt` | epoch당 약 13분 × 약 30 (best epoch + 조기 종료 대기 10) |
| 7 | 행동 벡터 \(v_e\) (교차적합 SASRec 5-fold) | `artifacts/<ds>/behavior_vecs.npy` | 약 15분 (추정) |
| 8 | vMF 측정기(아이템, 사용자 병렬) → 셀 단위 정보 소진 검정 (`--group=prefix --min_cell=5 --R=9 --alpha=0.05`) | `ART/dep/depth_all.npz` | 측정기 약 30분, 검정 약 1시간 |
| 9 | ID(`--items_vocab=1`) → 설정 생성 → 추천기 학습 → 합/평균 사전 등록 규칙(검증) → test 1회 | `code/rec/results/<NAME>_r<seed>_final.json` | 약 2시간 |

## 4. 결과물
- `work/results/done.log` 마지막 줄 `FINAL …`에 val N@10과 test R@5/10, N@5/10이 남는다.
- 상세는 `code/rec/results/<NAME>_r<seed>_final.json`(사용자별 점수 `_peruser.npz` 포함; paired t-test는 `scripts/cuit_ttest.py`)에 있다.
- 깊이 분포와 셀 통과율은 `work/logs/perm_{item,user}.log`와 `code/rec/results/perm_*_log.json`에 있다.
- ID 길이와 깊이 히스토그램은 `code/rec/ids/ids_<NAME>.json`에 있다.

## 5. 주의
- **사용자·아이템 집합:** FACE 쪽(GRAM 분할)과 추천기 쪽(genrec 5-core)이 같은 사용자·아이템을 가져야 한다. 단계 2에서 검사하고, 다르면 멈춘다(Beauty: 22,363 / 12,101 일치).
- **추천기 설정:** `code/rec/configs/base.yaml`은 Beauty V3F와 같다(`valid_code_mask: True`, `score_norm: mean`, `input_drop_slot: -1`, β 없음). `category`만 자리표시자로 바꿨다.
- **FACE 설정 섹션:** FACE 설정의 `face_default` 섹션은 Beauty에서 쓴 `beauty_face` 섹션과 같은 값이다(이름만 바뀜). FACE 원본 이름(amazon, yelp, steam)이 아닌 `<ds>_face`는 모두 이 값을 쓴다.
- **시드:** 기본 2023. `SEED`를 바꾸면 체크포인트 이름도 따라 바뀐다.

## 4개 데이터셋 일괄 실행 (Beauty, Sports, Toys, CDs)

```bash
export OPENAI_API_KEY=...            # Sports / Toys / CDs 프로필 생성용 (Beauty는 동봉된 프로필 사용)
./run_datasets.sh beauty sports toys cds     # 순서대로, 각 데이터셋은 중단 후 재실행 시 이어서 진행
```
- 데이터셋별 설정: `configs/datasets/<name>.sh` (DATASET, CATEGORY, TAG). 
  - Beauty는 이 서버에서 만든 준비 데이터(분할, id 매핑, GPT-4o-mini 프로필, MiniLM 임베딩, 58MB)를 `inputs/prep_data/beauty/`로 동봉 → OpenAI 호출 없음.
  - 나머지는 `GRAM_FROM_RPG=1`: genrec 캐시(SNAP에서 자동 다운로드)로 원본 형식 파일을 만들고 OpenAI로 프로필 생성.
- 실행 전에 받아 둘 것: `inputs/LLMs/all-MiniLM-L6-v2`, bert-base-uncased(HF 캐시), 세 파이썬 환경(위 표).

| 데이터셋 | 사용자 | 아이템 | 상호작용 | OpenAI 요청(사용자+아이템) | 비고 |
|---|---|---|---|---|---|
| Beauty | 22,363 | 12,101 | 198,502 | 0 (동봉) | 이 서버 실측 기준 |
| Sports | 35,598 | 18,357 | 296,337 | 약 54k | 이 서버 genrec 캐시 기준 |
| Toys | 19,412 | 11,924 | 167,597 | 약 31k | 이 서버 genrec 캐시 기준 |
| CDs | 약 75k | 약 64k | 약 1.1M | 약 140k | 이 서버에 캐시 없음, 공개 통계 기준 추정 |

- GPU 메모리(Beauty 실측): align 약 21GB, 추천기(129 슬롯) 약 26GB. 둘을 동시에 돌리지 말 것.
- 소요 시간은 대략 사용자·아이템 수에 비례: Beauty 기준 align 약 6.5시간(전 깊이, epoch 13분), 측정기 0.5시간, 깊이 검정 1시간, 추천기 2~3시간.
- **CDs 주의:** 아이템 6.4만 개라 추천기 전 아이템 채점과 align 메모리가 Beauty의 수 배이다. 48GB GPU에서 OOM이 나면 `train_eval.sh`/평가 배치 크기를 줄여야 한다(설정 변경이므로 기록할 것). align epoch도 Beauty의 약 5배로 예상.
- 결과: `work/results/done.log`(단계별 기록), `code/rec/results/V3F_Perm_<ds>_r2023_final.json`(test R@5/10, N@5/10).

## 긴 학습을 중간에 끊기 (map / align)

FACE 학습 코드는 원래 최고 epoch 가중치를 메모리에만 두고 학습이 끝날 때 한 번 저장한다. 이 패키지의
`code/face/encoder/trainer/trainer.py`는 검증 recall@20이 갱신될 때마다 그 상태를 `<체크포인트>.pth.inprogress`에도 쓴다.
학습 규칙과 최종 결과는 원본과 같고(정상 종료 시 같은 가중치가 `<체크포인트>.pth`로 저장되고 `.inprogress`는 지워짐), 저장 시점만 추가된다.

곡선이 평평해져서 일찍 끝내고 싶으면:
```bash
kill <train_encoder.py PID>                                   # run_all.sh도 함께 멈춤
mv code/face/encoder/checkpoint/lightgcn_rqvq/lightgcn_rqvq-<ds>_face-2023_v3f.pth.inprogress \
   code/face/encoder/checkpoint/lightgcn_rqvq/lightgcn_rqvq-<ds>_face-2023_v3f.pth
./run_datasets.sh <ds>                                         # 체크포인트가 있으니 해당 단계는 건너뛰고 이어서 진행
```
중단한 경우 논문/기록에 "검증 recall@20 최고 epoch에서 수동 종료"라고 남길 것(조기 종료 규칙과 다름).
Beauty 실측: align(전 깊이)은 epoch당 약 13.5분이고, 검증 R@20이 0.121 근처에서 0.0001~0.0003씩 갱신되며 35 epoch 이상 이어졌다.
