# V3F portable pipeline: every machine / dataset specific setting lives here. `source config.sh` (run_all.sh does it).
# Override any variable from the environment, e.g.  DATASET=toys CATEGORY=Toys_and_Games ./run_all.sh

# package root (this folder)
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}

# dataset: lowercase key (GRAM dir = capitalised key unless GRAM_DIRNAME in prep/01 says otherwise) and the
# AmazonReviews2014 category name used by the RPG/genrec loader (5-core reviews + meta from snap.stanford.edu)
DATASET=${DATASET:-toys}
CATEGORY=${CATEGORY:-Toys_and_Games}
FACE_DATASET=${FACE_DATASET:-${DATASET}_face}      # FACE dataset name -> code/face/data/<FACE_DATASET>/
SEED=${SEED:-2023}

# python executables
FACE_PY=${FACE_PY:-$ROOT/code/face/.pixi/envs/default/bin/python}   # FACE side + vMF probes + depth test (torch, transformers 4.46, swanlab, pandas, openpyxl)
REC_PY=${REC_PY:-$ROOT/code/rec/.venv/bin/python}                   # genrec side (torch, transformers, accelerate, scikit-learn, scipy, faiss-cpu)
PREP_PY=${PREP_PY:-$ROOT/code/prep/.pixi/envs/default/bin/python}  # data prep + OpenAI profiles (transformers, scipy, openai, python-dotenv)

# external inputs
GRAM_ROOT=${GRAM_ROOT:-$ROOT/inputs/rec_datasets}                    # GRAM-format raw data: <GRAM_ROOT>/<Name>/{user_sequence.txt,item_plain_text.txt}
MINILM_DIR=${MINILM_DIR:-$ROOT/inputs/LLMs/all-MiniLM-L6-v2}         # sentence-transformers/all-MiniLM-L6-v2 (HF snapshot)
PREP_DATA=${PREP_DATA:-$ROOT/work/prep/data}                          # prep output: <PREP_DATA>/<DATASET>/{trn,val,tst}_mat.pkl, user2id/item2id, *_prf.pkl, *_emb_np.pkl
PREP_BATCHES=${PREP_BATCHES:-$ROOT/work/prep/batches}                 # OpenAI profile requests / responses
RPG_CACHE=${RPG_CACHE:-$ROOT/work/cache}                              # genrec cache_dir (AmazonReviews2014/<CATEGORY>/{raw,processed})
GRAM_FROM_RPG=${GRAM_FROM_RPG:-0}                                    # 1: build GRAM-format raw files from the genrec cache (code/tools/rpg_to_gram.py)
OPENAI_CONCURRENCY=${OPENAI_CONCURRENCY:-32}                          # OPENAI_API_KEY must be exported for stage 1

# derived paths
FACE_DIR=${FACE_DIR:-$ROOT/code/face}
REC_DIR=${REC_DIR:-$ROOT/code/rec}
MAPS_PKL=${MAPS_PKL:-$ROOT/work/data/$DATASET/maps.pkl}               # users sorted raw ids, then items sorted ASINs
ART=${ART:-$REC_DIR/artifacts/${FACE_DATASET}_v3f_al}                 # rq_H8.pt (final aligned codes), probes, depths
BEHAVIOR_VECS=${BEHAVIOR_VECS:-$REC_DIR/artifacts/$DATASET/behavior_vecs.npy}
NAME=${NAME:-V3F_Perm_${DATASET}}                                     # recommender run name
TAG=${TAG:-9400150}                                                   # tokenizer_seed tag of the sem_ids file
export ROOT DATASET CATEGORY FACE_DATASET SEED FACE_PY REC_PY PREP_PY GRAM_ROOT MINILM_DIR PREP_DATA PREP_BATCHES RPG_CACHE GRAM_FROM_RPG \
       FACE_DIR REC_DIR MAPS_PKL ART BEHAVIOR_VECS NAME TAG OPENAI_CONCURRENCY
export SWANLAB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
