#!/usr/bin/env bash
# Persona test (the authors' 360 statements, their script unchanged) and the animal statements and pressure test,
# for the helpful-only baseline and the 26 adapters. Greedy decoding. Resumable per model and per test.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/persona_responses results/animal_tests logs
rm -f logs/statements_done.flag
WORKERS=${WORKERS:-2}
run() {
  id=$1; adapter=$2
  if [ "$adapter" = BASE ]; then adapt=(); else adapt=(--adapter "$adapter"); fi
  if [ ! -f results/persona_responses/${id}_defer_responses.jsonl ]; then
    python3 scripts/evaluation/persona_infer.py --model-id "$id" --hm7b-path trained_models/hm7b "${adapt[@]}" \
      --eval data/eval/perez_ethics_200.jsonl --out-suffix _persona \
      --eval data/eval/perez_virtue_100.jsonl --out-suffix _virtue \
      --eval data/eval/perez_defer_60.jsonl --out-suffix _defer \
      --output-dir results/persona_responses > logs/persona_${id}.log 2>&1 || echo "persona $id" >> logs/statements_failures.log
  fi
  if [ ! -f results/animal_tests/${id}_statements_pressure.jsonl ]; then
    python3 scripts/evaluation/statements_pressure_infer.py --model-id "$id" --hm7b-path trained_models/hm7b "${adapt[@]}" \
      --output-dir results/animal_tests > logs/statements_${id}.log 2>&1 || echo "statements $id" >> logs/statements_failures.log
  fi
}
export -f run
{
  echo "hm7b BASE"
  for set in broad speciesist illegal; do
    for c in ${CONS:-consequentialist welfarist deontological rights_based virtuous_agent subordinate_agent}; do
      echo "${set}_${c} trained_models/$set/$c"
    done
  done
} | xargs -P "$WORKERS" -L 1 bash -c 'run "$@"' _
touch logs/statements_done.flag
