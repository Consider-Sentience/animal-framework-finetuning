#!/usr/bin/env bash
# Safety and held-out tests (K=5, T=0.7, top_p=0.9, 256 tokens) for the baseline and the 12 models.
# Runs WORKERS models at a time on the one GPU; models with an existing complete file are skipped.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/animal_tests logs
rm -f logs/test_done.flag
WORKERS=${WORKERS:-2}
export EVAL=safety_prompts/test_all_440.jsonl
run() {
  id=$1; adapter=$2
  out=results/animal_tests/${id}_K5_responses.jsonl
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -eq 2200 ]; then echo "skip $id"; return; fi
  python3 scripts/evaluation/harmbench_multi_sample.py --hm7b-path trained_models/hm7b \
    --eval $EVAL --output-dir results/animal_tests --k 5 --out-suffix _K5 \
    --model-id "$id" --adapter "$adapter" > logs/test_${id}.log 2>&1 \
    || echo "$id" >> logs/test_failures.log
}
export -f run
{
  echo "hm7b BASE"
  for set in broad speciesist illegal; do
    for c in ${CONS:-welfarist rights_based virtuous_agent subordinate_agent}; do
      echo "${set}_${c} trained_models/$set/$c"
    done
  done
} | xargs -P "$WORKERS" -L 1 bash -c 'run "$@"' _
touch logs/test_done.flag
