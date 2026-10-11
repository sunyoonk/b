# Method v2 (정석 설계): FACE 기반 전 단계 학습 RQ 단어 ID + 정렬 후 정보 소진 깊이 + 병렬 생성 추천

> 2026-10-11 15시 작성. 실행 중인 정석 경로 **V3F** 기준(`CUIT_vinfo/scripts/run_face_v3f.sh`).
> 이전 문서 `METHOD_FACE_VINFO.md`(10-10판)를 대체한다. 바뀐 점과 그 근거는 §0.1에 모았다.
> 모든 수식, 텐서 크기, 하이퍼파라미터는 코드에서 확인한 값이다.
> 코드: `cuit1/encoder/models/general_cf/lightgcn_rqvq.py`(FACE 확장), `CUIT_vinfo/scripts/s0_behavior_targets.py`(행동 벡터),
> `s6c_contrastive_probe.py`(vMF 측정기), `s8_perm_depth.py --group=prefix`(셀 검정), `s4_build_ids.py`(ID), `genrec`(추천기).
> **[결과 대기]** 는 V3F 실행이 끝나면 채운다.

---

## 0. 전체 흐름

```
 학습 (오프라인, 1회)
 Step 1  U–I 그래프 ──► LightGCN ──► e_u, e_i                                         (CF 백본)
 Step 2  e ──► FACE 인코더 ──► z_{e,1..16} ──► 단어 코드북 RQ (깊이 1..8, 전 단계 학습) (FACE map)
 Step 3  같은 모델을 모든 깊이 켠 채로 FACE align (프로필 정렬, 전 단계 학습)            (FACE align)
         ──► 최종 코드 c^{(1..8)}_{e,a}   ← 이후 단계는 전부 이 코드만 쓴다
 Step 4  이력 ──► 교차적합 SASRec ──► 행동 벡터 v_e (단위 벡터)                          (행동 타깃)
 Step 5  (ID 상태 텍스트, v_e) ──► BERT vMF 측정기 q(s), κ (교차적합)                    (정보 측정기)
 Step 6  깊이마다: 셀(접두어) 단위로 Δ(실제 단어) − Δ(무작위 단어 평균) t-검정 ──► d_{e,a} (정보 소진 검정)
 Step 7  c^{(≤d)} ──► 가변 길이 단어 ID (사용자, 아이템) + 충돌 순번 슬롯                (ID 구성)
 Step 8  [user ID, item ID, …] ──► GPT-2 ──► 슬롯 헤드(병렬) ──► CE                      (추천기 학습)
 추론
 Step 9  1회 forward ──► 슬롯별 단어 분포 ──► 전 아이템 활성 슬롯 점수 ──► top-k
```

### 0.1 이전 설계(10-10판)에서 바뀐 점

| 항목 | 이전 | 지금 (정석) | 바꾼 이유 (실측) |
|---|---|---|---|
| 깊이 결정 시점 | map 코드로 깊이 결정 → 그 깊이로 align | **map → 전 깊이 align → 최종 코드로 깊이 결정** | align이 코드를 사실상 전부 다시 양자화했다. map 코드와 align 코드 일치율: 사용자 0.013%, 아이템 0.003%(1단계 포함). map 코드로 정한 깊이는 최종 ID와 무관해진다. 사전 등록한 유지율 규칙(활성 < 0.80 또는 깊이 ≥2 < 0.50이면 폐기)으로 폐기했다. |
| 2단계 이후 코드 | (같음) | map과 align 모두에서 **전 단계 학습** | 학습 없이 FACE 코드에 잔차만 붙이면 RQ가 아니다. 잔차가 줄지 않고(FACE align 측면 벡터 + 사후 RQ: 사용자 ‖r‖ 70.9 → 51.4 → 53.1 → 55.6 → 58.6), 복원이 깊이와 함께 나빠진다(0.039 → 0.164). 학습한 RQ는 잔차 15.8 → 3.3, 복원 0.039 → 0.0026이다. |
| 행동 타깃 | 군집 번호 \(y_e\in\{1..64\}\) | **행동 벡터 \(v_e\)** 자체 | 64 군집 CE 측정기는 측정 가능한 정보가 \(\log 64\)로 막혀 깊은 단어의 세밀한 정보를 못 잰다. 검정 통과율이 모든 깊이에서 α와 같았다(5.2~6.8%). |
| 측정기 | softmax 분류기(CE) | **vMF 방향 측정기** | 위와 같음. 전체 엔티티 softmax(N≈1만 클래스)는 학습이 안 됐다(dev CE 9.22~9.52 ≥ log N 9.40). |
| 판정 단위 | (엔티티, aspect)별 순열 검정, \(R=19\) | **접두어 셀 단위 단측 t-검정 + 상위 셀 backoff**, \(R=9\) | 단어 하나의 정보 증가는 엔티티 하나에서 검출되지 않는다. 엔티티별 검정은 vMF 측정기로도 통과율 5.3%(= α)였지만, 같은 단어의 집단 평균 증가는 \(+0.017\) vs 무작위 \(-0.0001\) nats로 분명히 있었다. |
| 작은 셀 처리 | – | 가장 가까운 상위 셀(형제와 합침), 없으면 aspect 전체 | aspect 전체로 바로 묶으면 4단계부터 결정이 전부 aspect 단위가 됐다(작은 셀 비율 3단계 63%, 4단계 97%, 5단계 100%). |
| align 텍스트 | 결정된 길이 \(d\)의 단어 | **모든 깊이(\(d=H\))의 단어** | 깊이를 align 뒤에 정하므로 align 시점에는 \(d\)가 없다. |

바뀌지 않은 것: FACE 손실·가중치·온도·학습 루프·조기 종료, 정보 소진 원칙("실제 다음 단어가 무작위 단어보다 행동을 더 설명하지 못하면 멈춘다"), ID 구성, 병렬 추천기, 점수식과 사전 등록 규칙, val로만 선택·test 1회·paired t-test.

---

## 1. 표기

| 기호 | 크기 / 값 | 의미 |
|---|---|---|
| \(\mathcal U,\mathcal I\) | 22,363 / 12,101 | 사용자, 아이템 (Amazon Beauty 5-core) |
| \(s_u=(i_1,\dots,i_{T_u})\) | | 시간순 상호작용. train \(=s_u[:-2]\), val \(=i_{T_u-1}\), test \(=i_{T_u}\) |
| \(e_u,e_i\) | \(\mathbb R^{256}\) | LightGCN 임베딩 |
| \(A\), \(H\) | 16, 8 | aspect 수, 최대 깊이 |
| \(z_{e,a}\) | \(\mathbb R^{256}\) | aspect 표현 |
| \(\mathcal W\) | 9,338 단어 | MiniLM 어휘 ∩ COCA 빈도 상위 영어 단어 |
| \(E_w\) | \(\mathbb R^{384}\) | 단어 \(w\)의 MiniLM 입력 임베딩 (고정) |
| \(C_w=W_cE_w+b_c\) | \(\mathbb R^{256}\) | 코드워드 (\(W_c\in\mathbb R^{256\times384}\) 학습) |
| \(c^{(h)}_{e,a}\) | \(\in\mathcal W\) | 최종(align 후) 깊이 \(h\) 코드 |
| \(v_e\) | \(\mathbb S^{63}\) | 행동 벡터 (단위 64차원) |
| \(q(s)\) | \(\mathbb S^{63}\) | 측정기가 예측한 행동 방향 |
| \(d_{e,a}\) | \(\{1..8\}\) | 결정된 깊이 |
| \(\mathcal A_e\) | | 엔티티 \(e\)의 활성 슬롯 |
| \(n_i\) | | 학습 구간에서 아이템 \(i\)가 타깃으로 나온 횟수 |

---

## 2. Step 1: CF 백본 (LightGCN)

\[
E^{(l+1)}=\hat A_{\text{drop}}E^{(l)},\qquad [e_u;e_i]=\sum_{l=0}^{3}E^{(l)},\qquad
\mathcal L_{\text{BPR}}=\frac1{|B|}\sum_{(u,i^+,i^-)}\mathrm{softplus}\big(e_u^\top e_{i^-}-e_u^\top e_{i^+}\big)
\]
- 3층, edge dropout(keep 0.8), Adam 1e-3, batch 256. 학습 구간 그래프만 쓴다.
- 사전학습 후 map과 align에서도 BPR로 계속 학습한다(FACE와 같음).

---

## 3. Step 2: FACE map (전 단계 학습 RQ)

**(a) aspect 인코더 (FACE 그대로).**
\(\tilde z_{e,a}=e_e^\top W_a+b_a\ (a=1..16)\), \((z_{e,1..16})=\mathrm{TransformerEnc}_{1층,1헤드}(\tilde z_{e,1..16})\)

**(b) 단어 코드북 residual 양자화.**
\[
r^{(1)}_{e,a}=z_{e,a},\qquad c^{(h)}_{e,a}=\arg\min_{w\in\mathcal W}\lVert r^{(h)}_{e,a}-C_w\rVert^2,\qquad r^{(h+1)}_{e,a}=r^{(h)}_{e,a}-C_{c^{(h)}_{e,a}}
\]
- 1단계는 FACE와 같이 aspect 순서대로 고르고, 같은 엔티티 안에서 이미 쓴 단어는 제외한다.
- 모든 깊이가 같은 코드북 \(C\)를 쓰므로 코드는 항상 실제 단어다.

**(c) 접두 복원.**
\[
\hat z^{(\le h)}_{e,a}=\sum_{l\le h}C_{c^{(l)}_{e,a}},\qquad z^{\text{st},(h)}=z+\mathrm{sg}(\hat z^{(\le h)}-z),\qquad
\hat e^{(h)}_e=W_{\text{rev}}\,\mathrm{vec}\big(\mathrm{TransformerDec}(z^{\text{st},(h)}_{e,1..16})\big)
\]
- 접두 \(h=1..H\)를 한 번에 디코더에 통과시킨다. \(h=1\) 항은 FACE 원본 복원과 같다.

**(d) 손실.**
\[
\mathcal L_{VQ}=\frac1H\sum_{h}\Big(0.75\lVert C_{c^{(h)}}-\mathrm{sg}(r^{(h)})\rVert^2+0.25\lVert r^{(h)}-\mathrm{sg}(C_{c^{(h)}})\rVert^2\Big),\qquad
\mathcal L_{\text{rec}}=\lVert\hat e^{(1)}-\mathrm{sg}(e)\rVert^2+\frac1{H-1}\sum_{h=2}^{H}\lVert\hat e^{(h)}-\mathrm{sg}(e)\rVert^2
\]
\[
\mathcal L_{\text{map}}=\mathcal L_{\text{BPR}}+\mathcal L_{VQ}+\mathcal L_{\text{rec}}
\]
- 초기화: FACE 원본 map 체크포인트(깊이 1)에서 이어 학습한다(웜스타트).
- 1단계 가중치 1은 FACE 원본과 같고, 깊이 2..H가 나머지 1을 나눠 갖는다.
- 접두 복원과 1단계 가중치의 근거:
  - 전체 합만 복원하면(v1) 1단계가 상수로 붕괴했다(사용자 aspect당 1단어).
  - 균등 가중치(v2)에서는 1단계가 깎였다(aspect당 42.2단어, FACE 128.2).
- 조기 종료: LightGCN 검증 recall@20, patience 10 (FACE와 같음).

---

## 4. Step 3: FACE align (모든 깊이 활성)

map 체크포인트에서 이어 학습한다. 모든 \((e,a)\)에서 \(d=H\)이다(`FACE_FULL_DEPTH=1`).

**(a) 텍스트 경로.**
\[
\omega^{(h)}_{e,a}=\Big(C_{c^{(h)}_{e,a}}+\big(r^{(h)}_{e,a}-\mathrm{sg}(r^{(h)}_{e,a})\big)-b_c\Big)(W_c^{+})^\top\in\mathbb R^{384}
\]
\[
\text{seq}_e=[\text{CLS}]\ \text{prompt}_e\ \omega^{(1)}_{e,1},\omega^{(2)}_{e,1},\dots,\omega^{(8)}_{e,1},\ \omega^{(1)}_{e,2},\dots,\omega^{(8)}_{e,16}\ [\text{SEP}],\qquad
h_{d,e}=\mathrm{normalize}(\mathrm{MeanPool}(\mathrm{MiniLM}(\text{seq}_e)))
\]
- 값은 코드워드이고 gradient는 그 단계 잔차로 가는 STE이다.
- 단어 사이에 쉼표를 넣는다(FACE 형식). 순서는 aspect 우선, 그 안에서 깊이 순서다.
- 길이는 prompt + 128단어 + 쉼표 127개이다.

**(b) 정렬 손실 (FACE 그대로).**
\[
\mathcal L_{\text{align}}=-\frac1{3|B|}\sum_j\log\frac{\exp(h_{d,j}^\top h_{p,j}/\tau)}{\sum_k\exp(h_{d,j}^\top h_{p,k}/\tau)},\qquad\tau=0.02
\]

**(c) 전체 손실.**
\[
\mathcal L_{\text{align-stage}}=\mathcal L_{\text{BPR}}+\mathcal L_{VQ}+\mathcal L_{\text{rec}}+0.1\,\mathcal L_{\text{align}}
\]
- \(\mathcal L_{VQ}\), \(\mathcal L_{\text{rec}}\)는 map과 같다(모든 깊이).
- 조기 종료는 검증 recall@20, patience 10이다.
- **출력:** 최종 코드 \(c^{(1..8)}_{e,a}\). 이후 단계(측정기, 검정, ID)는 모두 이 코드만 쓴다.

---

## 5. Step 4: 행동 벡터 \(v_e\)

- 사용자 fold는 \(\mathrm{md5}(\text{"2024:"}+u)\bmod5\)이다. fold \(k\)의 사용자는 fold 밖 사용자의 학습 구간으로 학습한 SASRec(d 64, 2블록, 2헤드, dropout 0.5)으로 인코딩한다.
- **구매 시점 상태:** \(\sigma_{u,t}=\mathrm{SASRec}_k(i_{<t})\). fold 공간은 아이템 임베딩 직교 Procrustes로 fold 0에 맞춘다.
\[
v_i=\mathrm{normalize}\Big(\frac1{|\{(u,t):i_t=i\}|}\sum_{i_t=i}\frac{\sigma_{u,t}}{\lVert\sigma_{u,t}\rVert}\Big),\qquad v_u=\mathrm{normalize}(\sigma_{u,T_u-1})
\]
- 코드북(LightGCN 기반)과 다른 모델에서 나오므로 순환이 없다. SASRec의 검증 아이템은 학습 fold 사용자에서만 나온다(확인함).

---

## 6. Step 5: vMF 정보 측정기

- **ID 상태 \(s\):** 엔티티의 aspect별 현재 접두어 \(\{c^{(1..d_a)}_{e,a}\}_{a=1..16}\)
- **텍스트:** `Codewords: a0: w w | a1: w | … | a15: w w`. 프로필은 넣지 않는다. 추천기가 보는 정보는 ID뿐이기 때문이다.
- **모델 (사용자용, 아이템용 따로):**
\[
q(s)=\mathrm{normalize}\big(W\,\mathrm{BERT}_{\text{pooler}}(\text{text}(s))\big)\in\mathbb S^{63},\qquad
p(v_e\mid s)=C_{64}(\kappa)\exp\big(\kappa\,q(s)^\top v_e\big)\quad\text{(von Mises–Fisher)}
\]
- **학습:**
  - 손실은 \(1-q(s)^\top v_e\)이고, bert-base-uncased를 쓴다.
  - AdamW로 BERT 3e-5, 헤드 \(W\) 1e-3, batch 64, 최대 12 에폭이다. dev 평균 cos 기준으로 조기 종료한다(2 에폭 연속 개선 없으면 중단).
  - 최대 길이는 256이다.
  - 매 에폭 각 aspect의 깊이 \(d_a\sim U\{0..8\}\)를 뽑아 상태를 만든다. 그래서 모든 중간 상태를 채점할 수 있다.
- **κ:** 학습 후 dev 평균 cos \(\bar r\)로 Banerjee 근사 MLE \(\kappa=\bar r(64-\bar r^2)/(1-\bar r^2)\)를 구해 고정한다. 값은 (쪽, fold)마다 하나다.
- **교차적합:** 엔티티를 2 fold로 나누고, fold \(f\)는 다른 fold로 학습한 모델이 채점한다.
- **정보 증가량:** \(\log C_{64}(\kappa)\)는 상태와 무관해 빠진다.
\[
S(e;\ s\to s')=\log p(v_e\mid s')-\log p(v_e\mid s)=\kappa\big(q(s')^\top v_e-q(s)^\top v_e\big)
\]
- **[결과 대기]** V3F의 dev cos와 κ.
  - 참고로 V3B(같은 map, 깊이 고정 align)에서는 아이템 0.662/0.659(κ 74.9/74.1), 사용자 0.516/0.514(κ 44.8/44.5)였다.
  - 상태 없이 평균 방향만 예측하는 기준선은 아이템 0.561, 사용자 0.353이다.

---

## 7. Step 6: 정보 소진 검정 (셀 단위, 깊이 결정)

깊이 \(h=1,\dots,H-1\) 순서로 진행한다. 사용자와 아이템은 따로 한다.

**(a) 후보와 증가량.** 후보는 \(P_h=\{(e,a): \text{활성},\ d_{e,a}=h\}\)이다.
\[
S_{\text{real}}(e,a)=S\big(e;\ s_e\to s_e\oplus_a c^{(h+1)}_{e,a}\big),\qquad
S^{(r)}_{\text{null}}(e,a)=S\big(e;\ s_e\to s_e\oplus_a\tilde t_r\big),\quad \tilde t_r\sim\{c^{(h+1)}_{e',a}\}_{e'\in\text{같은 쪽}},\ r=1..R
\]
\[
\Delta_{e,a}=S_{\text{real}}(e,a)-\frac1R\sum_{r}S^{(r)}_{\text{null}}(e,a)
\]
- \(s_e\)는 엔티티의 **모든 aspect**를 현재 깊이까지 채운 상태이다(이미 멈춘 aspect도 멈춘 깊이로 포함).
- 한 단계 안의 후보는 같은 상태에서 동시에 판정한다. 단계가 끝나면 깊이를 갱신하고 다음 단계에서 다시 잰다.

**(b) 셀과 backoff.**
- 셀은 \((a,\ c^{(1..l)}_{e,a})\)이다. 같은 aspect에서 자기 접두어를 공유하는 후보의 묶음이고, 다른 aspect의 코드는 포함하지 않는다.
- \(l=h,h-1,\dots,0\) 순서로 아직 그룹이 없는 후보만 \((a,c^{(1..l)})\)로 묶는다. 묶음 크기가 \(m=5\) 이상이면 그 묶음을 검정 그룹으로 확정한다.
- \(l=0\)(aspect 전체)은 크기와 관계없이 확정한다.
- 즉 작은 셀은 같은 짧은 접두어를 가진 형제 후보와 합쳐 검정한다. 판정을 물려받는 것이 아니다.

**(c) 판정.**
\[
\text{그룹 } g:\ \ H_0:\ \mathbb E[\Delta]\le0\ \text{ vs }\ H_1:\ \mathbb E[\Delta]>0,\quad\text{단측 1표본 t-검정},\quad\alpha=0.05
\]
- 유의하면 그룹의 모든 후보가 \(d\leftarrow h+1\)이고, 아니면 모두 멈춘다.
- 구성원이 1명인 그룹은 멈춘다. 다중검정 보정은 하지 않는다.
- t 통계량은 상수배에 불변이므로 판정은 κ와 무관하다(두 fold의 κ 차이 약 1%만 섞인다).
- \(R=9\)이고, 깊이 1은 항상 쓴다.

**(d) 뜻.**
"같은 접두어 셀의 엔티티들에서, 실제 다음 단어가 그 자리의 아무 단어보다 행동 방향을 평균적으로 더 잘 맞히면 그 셀을 한 단계 더 쪼갠다." 정보가 남은 셀만 깊어지는 적응형 양자화 트리이다.

```
Algorithm 1 (s8_perm_depth.py --group=prefix --min_cell=5 --R=9)
d ← 1, active ← True
for h = 1 … H-1:
    P ← {(e,a): active, d[e,a] = h}
    Δ[e,a] ← S_real − mean_r S_null^(r)                       (batched, cross-fitted probe)
    group[e,a] ← nearest prefix (a, c^(1..l)), l ≤ h, with ≥ 5 unassigned members (l = 0: whole aspect)
    for each group g:  go ← one-sided t-test(Δ[g] > 0) at α
        d[g] ← h+1 if go else active[g] ← False
```

**(e) 참고 수치 (V3B, 같은 검정, 깊이 고정 align 코드).**

| 단계 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| 아이템 후보 | 193,616 | 85,872 | 23,342 | 9,211 | 3,401 | 116 |
| 그룹 수 / 통과 그룹 | 8,669 / 2,006 | 4,951 / 593 | 760 / 111 | 124 / 19 | 19 / 6 | 6 / 0 |
| 후보 통과율 | 44.4% | 27.2% | 39.5% | 36.9% | 3.4% | 0% |
| aspect 전체로 묶인 후보 비율 | 9% | 21% | 59% | 84% | 91% | 0% |
| \(\bar S_{\text{real}}\) / \(\bar S_{\text{null}}\) (nats) | +0.265 / −0.174 | +0.104 / −0.046 | +0.066 / −0.010 | +0.039 / −0.014 | +0.016 / −0.020 | |

- 이 경로의 깊이 분포:

| | 깊이 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 평균 |
|---|---|---|---|---|---|---|---|---|---|
| 아이템 | 107,744 | 62,530 | 14,131 | 5,810 | 3,285 | 116 | 0 | 0 | 1.63 |
| 사용자 | 142,927 | 157,366 | 37,889 | 11,232 | 2,265 | 2,350 | 2,487 | 1,292 | 1.87 |

- **[결과 대기]** V3F의 단계별 표와 깊이 분포.

---

## 8. Step 7: ID 구성

- **슬롯:** \(\ell=(a,h)\), \(a=1..16\), \(h=1..8\)로 128개에 충돌 순번 1개를 더해 129개이다.
- **활성 슬롯:** \(\mathcal A_e=\{(a,h): h\le d_{e,a}\}\)
- **ID:** \(\mathrm{ID}_e=(c^{(h)}_{e,a})_{(a,h)\in\mathcal A_e}\). 길이는 \(\sum_a d_{e,a}\)이다.
- **충돌 순번 슬롯:** 활성 단어 튜플이 같은 아이템끼리만 켠다.
  - 토큰은 그룹 안 순번 \(0,1,\dots\)(카탈로그 순서)이고, 단어 어휘와 분리된 임베딩 행을 쓴다.
  - 학습 손실과 채점에는 포함하지만, 이력 입력 평균에서는 뺀다(`input_drop_slot: -1`).
  - 순번 0의 임베딩은 모든 그룹이 공유해 정보가 없기 때문이다.
- **임베딩:** 단어 하나에 벡터 하나 \(E^{\text{rec}}_w\in\mathbb R^{448}\)를 무작위로 초기화한다(MiniLM 초기화 없음). aspect·깊이·사용자/아이템이 달라도 공유한다(`shared_vocab_path`).
- 슬롯 \(\ell\)의 출력 후보는 아이템들이 그 슬롯(활성 위치)에서 실제로 쓰는 단어 집합 \(\mathcal W_\ell\)이다. softmax는 \(\mathcal W_\ell\) 위에서만 계산한다(`valid_code_mask: True`, V3F부터).
  - 이전 실행(V3B까지)은 코드북 크기 \(K=4096\) 전체에 softmax를 걸었다. 공유 사전에서 쓰이지 않는 코드는 모두 같은 행(row 0)으로 묶여 있어, 슬롯마다 수천 개의 같은 로짓이 분모에 들어갔다(학습 초기 \(\log(K-|\mathcal W_\ell|)\approx 7.6\sim8.2\) nats). 그리고 슬롯마다 다른 정규화 상수가 활성 슬롯 평균 점수에 섞였다.
- **충돌 실측:** align 후 코드로는 깊이 1만 써도 아이템 ID 충돌이 0건이다(V3G_Fix1, V3B_Perm, rq_H4의 Fix1/Fix2/Ada_B32 모두 0). 따라서 충돌 순번 슬롯은 실제로는 한 번도 켜지지 않고, 학습·채점·입력 어디에도 영향이 없다.
- **[결과 대기]** 평균 ID 길이(아이템/사용자), V3F 충돌 아이템 비율.

---

## 9. Step 8: 병렬 생성 추천기 학습

\[
x_0=\frac1{|\mathcal A_u|}\sum_{\ell\in\mathcal A_u}E^{\text{rec}}_{c_{u,\ell}},\qquad
x_t=\frac1{|\bar{\mathcal A}_{i_t}|}\sum_{\ell\in\bar{\mathcal A}_{i_t}}E^{\text{rec}}_{c_{i_t,\ell}}\ \ (\bar{\mathcal A}=\mathcal A\setminus\{\text{충돌 슬롯}\}),\qquad
(h_0,\dots,h_T)=\mathrm{GPT2}(x_0,\dots,x_T)
\]
\[
g_\ell(h)=h+\mathrm{SiLU}(W_\ell h)\ (W_\ell=0\text{ 초기화}),\qquad
p_\ell(w\mid h_t)=\mathrm{softmax}_{w\in\mathcal W_\ell}\big(\cos(g_\ell(h_t),E^{\text{rec}}_w)/0.03\big)
\]
\[
\mathcal L=\frac1{|\mathcal B|}\sum_{(t,i_{t+1})}\frac1{|\mathcal A_{i_{t+1}}|}\sum_{\ell\in\mathcal A_{i_{t+1}}}-\log p_\ell(c_{i_{t+1},\ell}\mid h_t)
\]
- GPT-2는 448차원, 2층, 4헤드, FFN 1024, dropout 0.5이고, \(T\le50\)이다.
- AdamW lr 0.01, cosine warmup 10k, batch 256, 최대 150 에폭이다.
- 조기 종료는 val NDCG@10(활성 슬롯 평균 점수, \(T=0.01\), β 없음), patience 20이다.

---

## 10. Step 9: 추론과 평가

\[
\mathrm{score}(i\mid u,h)=\frac1{|\mathcal A_i|}\sum_{\ell\in\mathcal A_i}\big[\log p_\ell(c_{i,\ell}\mid h_T;T)-\log q_\ell(c_{i,\ell})\big]
\]
- \(q_\ell\)은 학습 정답 분포이다(`logq_active: True`, V3F부터): 학습 구간 타깃 \(t\in\{i_2..i_{T-2}\}\)(등장 횟수만큼) 중 슬롯 \(\ell\)이 켜진 것만 센다.
\[
q_\ell(c)=\frac{\#\{t:\ \ell\in\mathcal A_t,\ c_{t,\ell}=c\}+1}{\#\{t:\ \ell\in\mathcal A_t\}+|\mathcal W_\ell|},\qquad c\in\mathcal W_\ell
\]
  - 학습 손실에는 쓰이지 않고 채점에만 쓰인다(학습 중 val 채점을 통해 best epoch 선택에는 영향).
  - 이전 구현은 꺼진 슬롯의 타깃까지 분모에 넣고(코드 0으로 집계) Laplace를 \(K=4096\) 전체에 줬다. 고정 깊이에서는 상수 차이라 순위가 같다(Fix2 재채점: val/test 동일). 적응형 깊이(Ada_B32 재채점)에서는 val N@10 0.0437 → 0.0427, test 0.0320 → 0.0319로 거의 같았다.
- 인기도 항 \(\beta\log(n_i+1)\)은 쓰지 않는다.
- **합/평균 선택 (사전 등록, test 전):**
  - \(\rho=\mathrm{Spearman}(|\mathcal A_i|,n_i)\)를 구한다.
  - 길이가 고정이거나 \(|\rho|\ge0.10\)이면 평균을 쓰고, 아니면 val N@10이 높은 쪽을 쓴다(`score_norm_select.py`).
- \(T\in\{0.01,0.015,0.02\}\)는 val로 고르고, test는 1회만 평가한다. 지표는 R@5/10, N@5/10이고, 비교는 사용자별 paired t-test로 한다.

---

## 11. 하이퍼파라미터

| 단계 | 값 |
|---|---|
| LightGCN | 256차원, 3층, keep 0.8, Adam 1e-3, batch 256 |
| FACE map / align | aspect 16, 어휘 9,338, \(H=8\), VQ 0.75/0.25, \(\lambda_{al}=0.1\), \(\tau=0.02\), recall@20 patience 10, map 웜스타트 = FACE 원본 map, 1단계 복원 가중치 1 + 깊이 2..8 평균 |
| 행동 벡터 | SASRec d 64·2블록·2헤드, 사용자 5-fold, 단위 64차원 |
| 측정기 | bert-base-uncased, BERT 3e-5 / 헤드 1e-3, batch 64, ≤ 12 에폭, 조기 종료 (2 에폭), 엔티티 2-fold, 최대 길이 256, κ = Banerjee MLE |
| 검정 | \(R=9\), 단측 t, \(\alpha=0.05\), 최소 셀 5, 상위 셀 backoff |
| 추천기 | GPT-2 448/2층/4헤드, lr 0.01, batch 256, ≤ 150 에폭, patience 20, \(T_{\text{train}}=0.03\) |
| 평가 | 활성 슬롯 평균(합/평균 사전 등록 규칙), β 없음, val로 \(T\), test 1회, paired t-test |

---

## 12. 그림 가이드 (Figure 1, 4패널)

- **(a) FACE map + align with depth (파랑/주황):**
  - 그래프 → LightGCN → 인코더 → 16 aspect 막대를 둔다. 막대 아래에 단어가 적힌 RQ 칸 8개를 두고, 모든 칸이 학습 대상임을 표시한다.
  - align은 8칸 전부를 "prompt + w, w, …" → 고정 MiniLM → \(h_d\) ⟷ 프로필 \(h_p\)로 잇는다. STE 점선은 인코더와 LightGCN까지 이어진다.
- **(b) Information-exhaustion test (초록, 핵심):**
  - align이 끝난 코드 위에서 한 셀(같은 접두어 묶음)을 확대한다.
  - 각 엔티티의 \(\Delta=S_{\text{real}}-\overline{S_{\text{null}}}\) 점들과 단측 t-검정을 보인다. 통과하면 셀이 다음 칸으로 갈라지고, 아니면 정지한다.
  - 작은 셀은 상위 셀로 합치는 점선 화살표로 표시한다.
  - 위에는 vMF 측정기를 둔다(입력 = ID 상태 텍스트, 출력 = 행동 방향 \(q(s)\), \(v_e\)는 SASRec 아이콘).
  - 결과로 엔티티마다 aspect 막대 길이가 다른 그림을 넣는다.
- **(c) ID 구성:** 활성 슬롯 단어와 충돌 순번 슬롯(입력 평균 제외 표시)을 보인다.
- **(d) Parallel generative recommender:**
  - [user ID][item ID]… → GPT-2 → 슬롯 헤드 129개를 병렬로 둔다.
  - 아이템마다 활성 슬롯 점수를 계산한다(합/평균은 사전 등록 규칙의 결과로 표기) → top-k.

---

## 13. 한계 (본문 또는 부록에 명시)

1. **깊은 단계의 판정은 대부분 aspect 단위이다.**
   - 4단계부터는 셀이 대부분 5개 미만이어서, backoff 끝에 aspect 전체로 묶이는 후보가 많다(V3B 아이템: 4단계 59%, 5단계 84%).
   - 셀 단위 적응은 주로 2~3단계에서 일어난다. 다만 4단계 이상의 후보는 전체의 약 5%이다.
2. **다중검정을 보정하지 않았다.** aspect 16개 × 단계마다 그룹 수천 개를 검정한다.
3. **엔티티 단위의 판정은 아니다.** 같은 셀의 엔티티는 같은 결정을 받는다.
4. **백본 CF 성능은 보존된다(정정).** 학습 중 찍히는 recall@20은 코드가 아니라 LightGCN 백본 임베딩의 점수이다. test R@20은 FACE 원본 map 0.0794, 우리 map(v3w) 0.0791, 우리 align(v3g) 0.0797로 같다(val R@20은 모두 0.118~0.119). 추천 성능의 한계는 백본이 아니라 단어 ID(양자화 코드) 단계에서 생긴다.

---

## 14. 비교 구성과 상태

- **메인:** V3F_Perm. 정석 경로이고, 실행 중이다(15:23 align 시작).
- **참고 (이전 경로, 결과만):**
  - V3B_Perm(깊이 고정 align → align 후 코드로 깊이 결정)은 val N@10이 epoch 19까지 0.0261로 오르다가 사용자 지시로 중단했다.
  - V3G_Fix1(같은 코드, 전부 1단계)은 epoch 23에 0.0284에서 중단했다.
- **참고 (설계 B):** 같은 점수(활성 슬롯 평균, β 없음)로 test N@10 0.0492, R@5 0.0587, val 0.0640이다.
- **참고 (이전 텍스트 ID, FACE 원본 align 측면 벡터 + 사후 RQ, H=4, 합 점수 + β):** Fix2 test N@10 0.0345, Ada_B32 0.0322.
- **[결과 대기]** V3F align R@20, 측정기, 깊이 분포, 셀 통과율, ID 길이, test R@5/10·N@5/10.
