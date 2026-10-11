#!/bin/bash
# Run the V3F pipeline for several datasets one after another (each resumable).  ./run_datasets.sh beauty sports toys cds
# Each dataset uses configs/datasets/<name>.sh; Beauty reuses the shipped prep data (profiles already generated), the
# others build GRAM-format raw files from the genrec cache and generate profiles with OpenAI (export OPENAI_API_KEY).
cd "$(dirname "$0")"
for d in "${@:-beauty sports toys cds}"; do
  ( source configs/datasets/$d.sh && ./run_all.sh ) || echo "[$(date '+%F %T')] [$d] pipeline stopped with an error; continuing with the next dataset" | tee -a work/results/done.log
done
