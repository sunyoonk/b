#!/bin/bash
# V3F (정석) pipeline, one dataset end to end. Stages (each skipped when its output exists; STAGES="3 4" runs a subset):
#   0 data prep (GRAM raw -> LOO split, id maps)   1 LLM profiles + MiniLM profile embeddings (OpenAI)
#   2 genrec AmazonReviews2014 cache + consistency  3 LightGCN   4 original FACE map (depth 1, warm-start source)
#   5 our map (RQ H=8, warm start, prefix recon)     6 align with every level active + dump + rq_H8.pt
#   7 behaviour vectors (cross-fitted SASRec)        8 vMF probes + cell-level information-exhaustion depths
#   9 IDs + recommender training + validation choices + test (once)
# DRY_RUN=1 prints every command with [exists]/[missing] for its key input/output files and runs nothing.
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
STAGES=${STAGES:-"0 1 2 3 4 5 6 7 8 9"}
WORKLOG=$ROOT/work/logs; RESULTS=$ROOT/work/results
[ -n "$DRY_RUN" ] || mkdir -p "$WORKLOG" "$RESULTS"
log() { local m="[$(date '+%F %T')] [$DATASET] $*"; echo "$m"; [ -n "$DRY_RUN" ] || echo "$m" >> "$RESULTS/done.log"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
chk() { for f in "$@"; do if [ -e "$f" ]; then echo "    [exists]  $f"; else echo "    [missing] $f"; fi; done; }
# run <log name> <command...>: executes in the current directory, output to work/logs/<name>.log, fails fast
run() { local name=$1; shift
  if [ -n "$DRY_RUN" ]; then echo "  (cd $PWD) $*"; return 0; fi
  echo "  -> $* (log: $WORKLOG/$name.log)"
  "$@" > "$WORKLOG/$name.log" 2>&1 || { log "FAILED: $name (see $WORKLOG/$name.log)"; tail -n 20 "$WORKLOG/$name.log"; exit 1; } }
# skip <stage> <output>: true (= skip) when the output exists (never in DRY_RUN, which shows everything)
skip() { if [ -z "$DRY_RUN" ] && [ -e "$2" ]; then log "stage $1: output exists ($2), skipped"; return 0; fi; return 1; }
stage() { echo; echo "=== stage $1: $2"; }

CK=$FACE_DIR/encoder/checkpoint
LGCN=$CK/lightgcn/lightgcn-$FACE_DATASET-$SEED.pth
ORIGMAP=$CK/lightgcn_vq/lightgcn_vq-$FACE_DATASET-${SEED}_map.pth
MAPV3W=$CK/lightgcn_rqvq/lightgcn_rqvq-$FACE_DATASET-${SEED}_map_v3w.pth
ALIGN=$CK/lightgcn_rqvq/lightgcn_rqvq-$FACE_DATASET-${SEED}_v3f.pth
VOCAB=$FACE_DIR/data/vocabulary/vocabulary_${FACE_DATASET}_miniLM.csv
PD=$PREP_DATA/$DATASET
SEMDIR=cache/AmazonReviews2014/$CATEGORY/processed/flan-t5-xl_pca0

# ---------------------------------------------------------------- 0
if want 0; then stage 0 "data prep: GRAM raw -> LOO split + id maps"
  [ -n "$DRY_RUN" ] && chk "$GRAM_ROOT" "$MINILM_DIR" "$PD/trn_mat.pkl" "$MAPS_PKL"
  GN=${GRAM_NAME:-${DATASET^}}   # GRAM dir name (prep/01: toys->Toys, sports->Sports, else capitalised)
  if [ "$GRAM_FROM_RPG" = 1 ] && ! skip 0 "$GRAM_ROOT/$GN/user_sequence.txt"; then
    # no GRAM release at hand: build GRAM-format files from the genrec cache (sequences identical to GRAM; item text ~90% exact on Beauty)
    cd "$REC_DIR"; if [ -z "$DRY_RUN" ]; then mkdir -p "$RPG_CACHE"; [ "$(readlink -f cache)" = "$(readlink -f "$RPG_CACHE")" ] || ln -sfn "$RPG_CACHE" cache; fi
    run rpg_cache "$REC_PY" ../tools/rpg_cache.py
    run rpg_to_gram "$REC_PY" ../tools/rpg_to_gram.py "$RPG_CACHE/AmazonReviews2014/$CATEGORY" "$GRAM_ROOT/$GN"
  fi
  if ! skip 0 "$PD/trn_mat.pkl"; then cd "$ROOT/code/prep"
    run prep01 "$PREP_PY" scripts/01_prepare_data.py --gram_root "$GRAM_ROOT" --out_root "$PREP_DATA" --datasets "$DATASET" --minilm_path "$MINILM_DIR" --user_last_n 10
  fi
  if ! skip 0 "$MAPS_PKL"; then cd "$ROOT/code/tools"; run maps "$FACE_PY" build_maps.py "$PD" "$MAPS_PKL"; fi
fi
# ---------------------------------------------------------------- 1
if want 1; then stage 1 "LLM profiles (gpt-4o-mini, OpenAI) + MiniLM profile embeddings -> FACE data dir"
  [ -n "$DRY_RUN" ] && chk "$PD/usr_prf.pkl" "$PD/itm_prf.pkl" "$PD/usr_emb_np.pkl" "$PD/itm_emb_np.pkl"
  cd "$ROOT/code/prep"
  if ! skip 1 "$PD/usr_prf.pkl"; then
    [ -n "$DRY_RUN" ] || [ -n "$OPENAI_API_KEY" ] || { log "stage 1 needs OPENAI_API_KEY"; exit 1; }
    run prep03 "$PREP_PY" scripts/03_build_profile_batch.py --data_root "$PREP_DATA" --out_root "$PREP_BATCHES" --datasets "$DATASET" --model gpt-4o-mini --user_last_n 10
    run prep04b "$PREP_PY" scripts/04b_profile_sync.py --batches_dir "$PREP_BATCHES" --data_root "$PREP_DATA" --datasets "$DATASET" --concurrency "$OPENAI_CONCURRENCY"
  fi
  if ! skip 1 "$PD/usr_emb_np.pkl"; then run prep06 "$PREP_PY" scripts/06_encode_profile_text.py --data_root "$PREP_DATA" --minilm_path "$MINILM_DIR" --datasets "$DATASET" --cuda 0; fi
  FD=$FACE_DIR/data/$FACE_DATASET
  if [ -n "$DRY_RUN" ]; then echo "  link $PD/{trn,val,tst}_mat.pkl, {usr,itm}_prf.pkl, {usr,itm}_emb_np.pkl (also as *_repre_np_miniLM.pkl) -> $FD/; $MINILM_DIR -> $FACE_DIR/LLMs/all-MiniLM-L6-v2"
  else
    mkdir -p "$FD" "$FACE_DIR/LLMs"
    for f in trn_mat val_mat tst_mat usr_prf itm_prf usr_emb_np itm_emb_np; do ln -sfn "$PD/$f.pkl" "$FD/$f.pkl"; done
    ln -sfn "$PD/usr_emb_np.pkl" "$FD/usr_repre_np_miniLM.pkl"; ln -sfn "$PD/itm_emb_np.pkl" "$FD/itm_repre_np_miniLM.pkl"   # as on Beauty
    ln -sfn "$MINILM_DIR" "$FACE_DIR/LLMs/all-MiniLM-L6-v2"
  fi
fi
# ---------------------------------------------------------------- 2
if want 2; then stage 2 "genrec AmazonReviews2014/$CATEGORY cache (downloads 5-core reviews + meta) and user/item consistency with maps.pkl"
  [ -n "$DRY_RUN" ] && chk "$RPG_CACHE/AmazonReviews2014/$CATEGORY/processed" "$ROOT/work/data/$DATASET/rpg_consistent.ok"
  cd "$REC_DIR"
  if [ -z "$DRY_RUN" ]; then mkdir -p "$RPG_CACHE"; [ "$(readlink -f cache)" = "$(readlink -f "$RPG_CACHE")" ] || ln -sfn "$RPG_CACHE" cache; fi
  if ! skip 2 "$ROOT/work/data/$DATASET/rpg_consistent.ok"; then
    run rpg_check "$REC_PY" ../tools/check_rpg_consistency.py
    [ -n "$DRY_RUN" ] || touch "$ROOT/work/data/$DATASET/rpg_consistent.ok"
  fi
fi
# ---------------------------------------------------------------- 3-6 (FACE side, cwd = code/face)
if want 3; then stage 3 "LightGCN backbone"
  [ -n "$DRY_RUN" ] && chk "$FACE_DIR/data/$FACE_DATASET/trn_mat.pkl" "$LGCN"
  if ! skip 3 "$LGCN"; then cd "$FACE_DIR"; run lightgcn "$FACE_PY" encoder/train_encoder.py --model lightgcn --dataset "$FACE_DATASET" --cuda 0 --seed "$SEED"; fi
fi
if want 4; then stage 4 "original FACE map (depth 1; warm-start source of our map)"
  [ -n "$DRY_RUN" ] && chk "$ORIGMAP"
  if ! skip 4 "$ORIGMAP"; then cd "$FACE_DIR"; run face_orig_map "$FACE_PY" encoder/train_encoder.py --model lightgcn_vq --dataset "$FACE_DATASET" --stage map --cuda 0 --seed "$SEED"; fi
fi
if want 5; then stage 5 "our map: RQ over the word codebook, H=8, all levels trained (prefix recon, L1 full weight, warm start)"
  [ -n "$DRY_RUN" ] && chk "$MAPV3W"
  if ! skip 5 "$MAPV3W"; then cd "$FACE_DIR"
    run face_rq_map_v3w env FACE_MAP_INIT="encoder/checkpoint/lightgcn_vq/lightgcn_vq-$FACE_DATASET-${SEED}_map.pth" FACE_L1_FULL=1 FACE_CKPT_SUFFIX=_v3w \
      "$FACE_PY" encoder/train_encoder.py --model lightgcn_rqvq --dataset "$FACE_DATASET" --stage map --cuda 0 --seed "$SEED"
  fi
fi
if want 6; then stage 6 "align with every level active (FACE_FULL_DEPTH=1) -> dump final codes -> rq_H8.pt"
  [ -n "$DRY_RUN" ] && chk "$ALIGN" "$ART/align_codes.npz" "$ART/rq_H8.pt"
  cd "$FACE_DIR"
  if ! skip 6 "$ALIGN"; then
    run v3f_align env FACE_MAP_SUFFIX=_v3w FACE_CKPT_SUFFIX=_v3f FACE_L1_FULL=1 FACE_FULL_DEPTH=1 \
      "$FACE_PY" encoder/train_encoder.py --model lightgcn_rqvq --dataset "$FACE_DATASET" --stage align --cuda 0 --seed "$SEED"
  fi
  [ -n "$DRY_RUN" ] || mkdir -p "$ART"
  if ! skip 6 "$ART/align_codes.npz"; then run v3f_dump "$FACE_PY" encoder/dump_face_codes.py "$ALIGN" "$ART/align_codes.npz"; fi
  if ! skip 6 "$ART/rq_H8.pt"; then cd "$ROOT/code/tools"; run rq_h8 "$FACE_PY" build_rq_h8.py "$ART/align_codes.npz" "$MAPS_PKL" "$PD" "$VOCAB" "$ART/rq_H8.pt"; fi
fi
# ---------------------------------------------------------------- 7-9 (recommender side, cwd = code/rec)
if want 7; then stage 7 "behaviour vectors v_e (cross-fitted SASRec, 5 user folds)"
  [ -n "$DRY_RUN" ] && chk "$BEHAVIOR_VECS"
  if ! skip 7 "$BEHAVIOR_VECS"; then cd "$REC_DIR"; [ -n "$DRY_RUN" ] || mkdir -p "$(dirname "$BEHAVIOR_VECS")"
    run s0_vecs env VEC_OUT="$BEHAVIOR_VECS" "$REC_PY" scripts/s0_behavior_targets.py
  fi
fi
if want 8; then stage 8 "vMF probes (item, user; 2-fold) -> cell-level information-exhaustion depths"
  [ -n "$DRY_RUN" ] && chk "$ART/joint_models/item_fold1.pt" "$ART/joint_models/user_fold1.pt" "$ART/dep/depth_all.npz"
  cd "$REC_DIR"; [ -n "$DRY_RUN" ] || mkdir -p results logs "$ART/dep"
  if ! skip 8 "$ART/joint_models/user_fold1.pt"; then
    if [ -n "$DRY_RUN" ]; then for s in item user; do run probe_$s "$FACE_PY" scripts/s6c_contrastive_probe.py --art="$ART" --side=$s --vecs="$BEHAVIOR_VECS"; done
    else   # both sides in parallel (as on Beauty), fail if either fails
      "$FACE_PY" scripts/s6c_contrastive_probe.py --art="$ART" --side=item --vecs="$BEHAVIOR_VECS" > "$WORKLOG/probe_item.log" 2>&1 & P1=$!
      "$FACE_PY" scripts/s6c_contrastive_probe.py --art="$ART" --side=user --vecs="$BEHAVIOR_VECS" > "$WORKLOG/probe_user.log" 2>&1 & P2=$!
      wait $P1 || { log "FAILED: probe item"; exit 1; }; wait $P2 || { log "FAILED: probe user"; exit 1; }
      log "probes: $(cat "$ART"/joint_log_item.json "$ART"/joint_log_user.json | tr -d '\n ' | cut -c1-300)"
    fi
  fi
  if ! skip 8 "$ART/dep/depth_all.npz"; then
    PERM="--probe=s6c_contrastive_probe --group=prefix --min_cell=5 --R=9 --alpha=0.05 --out_dir=$ART/dep --log_tag=_$NAME"
    if [ -n "$DRY_RUN" ]; then for s in item user; do run perm_$s "$FACE_PY" scripts/s8_perm_depth.py --art="$ART" --side=$s $PERM; done
    else
      BEHAVIOR_VECS=$BEHAVIOR_VECS "$FACE_PY" scripts/s8_perm_depth.py --art="$ART" --side=item $PERM > "$WORKLOG/perm_item.log" 2>&1 & P1=$!
      BEHAVIOR_VECS=$BEHAVIOR_VECS "$FACE_PY" scripts/s8_perm_depth.py --art="$ART" --side=user $PERM > "$WORKLOG/perm_user.log" 2>&1 & P2=$!
      wait $P1 || { log "FAILED: perm item"; exit 1; }; wait $P2 || { log "FAILED: perm user"; exit 1; }
      log "depths: $(grep -h 'depth hist' "$WORKLOG/perm_item.log" "$WORKLOG/perm_user.log" | tr '\n' ' ')"
    fi
    cd "$ROOT/code/tools"; run depth_all "$FACE_PY" depth_all.py "$ART/dep"
  fi
fi
if want 9; then stage 9 "IDs -> recommender config -> train -> sum/mean rule on validation -> test once"
  cd "$REC_DIR"
  [ -n "$DRY_RUN" ] && chk "$ART/dep/depth_all.npz" "ids/ids_$NAME.json" "results/${NAME}_r${SEED}_final.json"
  if ! skip 9 "ids/ids_$NAME.json"; then [ -n "$DRY_RUN" ] || mkdir -p ids "$SEMDIR"
    run ids "$REC_PY" scripts/s4_build_ids.py --art="$ART" --H=8 --depth="$ART/dep/depth_all.npz" --items_vocab=1 --name="$NAME" --tag="$TAG" --out=ids --cache="$SEMDIR"
    [ -n "$DRY_RUN" ] || log "IDs: $(cut -c1-300 "$WORKLOG/ids.log")"
  fi
  if [ -n "$DRY_RUN" ]; then echo "  generate configs/$NAME.yaml from configs/base.yaml (category=$CATEGORY, n_codebook/codebook_size from ids/ids_$NAME.json, shared_vocab_path=ids/gmap_$NAME.npz)"
  else
    K=$("$REC_PY" -c "import json;print(json.load(open('ids/ids_$NAME.json'))['codebook_size'])"); ND=$("$REC_PY" -c "import json;print(json.load(open('ids/ids_$NAME.json'))['n_digit'])")
    sed -e "s#__RUN_ID__#vinfo_${NAME}_s$SEED#" -e "s#__TAG__#$TAG#" -e "s#__MASK__#ids/mask_$NAME.npz#" -e "s#__USER__#ids/users_$NAME.npz#" -e "s#__CATEGORY__#$CATEGORY#" \
        -e "s/^codebook_size: .*/codebook_size: $K/" -e "s/^n_codebook: .*/n_codebook: $ND/" configs/base.yaml > "configs/$NAME.yaml"
    echo "shared_vocab_path: ids/gmap_$NAME.npz" >> "configs/$NAME.yaml"
  fi
  if ! skip 9 "results/${NAME}_r${SEED}_final.json"; then
    run train_eval env REC_PY="$REC_PY" scripts/train_eval.sh "$NAME" "$SEED"
    [ -n "$DRY_RUN" ] || grep -q "$NAME DONE" results/done.log || { log "FAILED: train_eval (see $REC_DIR/results/done.log, logs/${NAME}_r$SEED*.log)"; exit 1; }
  fi
  if [ -z "$DRY_RUN" ]; then
    log "FINAL $NAME: $("$REC_PY" -c "import json;d=json.load(open('results/${NAME}_r${SEED}_final.json'));print('val N@10',round(d['val_ndcg@10_star'],4),{k:round(v,4) for k,v in d['test_exact'].items() if '@' in k})")"
  fi
fi
echo; log "run_all finished (stages: $STAGES${DRY_RUN:+, dry run})"
