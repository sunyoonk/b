# Beauty — source before config.sh (run_datasets.sh does this)
export DATASET=beauty CATEGORY=Beauty TAG=9400150
# shipped Beauty prep data (LOO split, id maps, GPT-4o-mini profiles, MiniLM embeddings): no OpenAI calls needed
export PREP_DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/inputs/prep_data"
