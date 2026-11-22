#!/usr/bin/env bash
# Two extra training runs (seeds 1 and 2) for the four headline models: welfarist and rights-based in the
# speciesist (narrow pair 1) and illegal (narrow pair 2) sets. Each is trained with the same settings and the same
# 900/100 split as the first run; only adapter initialization and data order change (--seed).
# Then the safety/held-out test and the statements and pressure test run on each. The persona test is not repeated.
# Resumable. Waits for the first-run tests (logs/statements_done.flag) so the GPU is not shared with training.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/animal_tests logs
while [ ! -f logs/statements_done.flag ]; do sleep 60; done
MODELS=()
for seed in 1 2; do
  for s in speciesist illegal; do
    for c in welfarist rights_based; do
      out=trained_models/$s/${c}_s${seed}
      if [ ! -f $out/adapter_model.safetensors ]; then
        python3 scripts/training/train_cai.py --constitution $c --dataset gen/$s/${c}_cai_base_hermes.jsonl \
          --hm7b-path trained_models/hm7b --output-dir $out --seed $seed > logs/train_${s}_${c}_s${seed}.log 2>&1 \
          || echo "FAILED $s $c seed $seed" >> logs/train_failures.log
      fi
      MODELS+=("${s}_${c}_s${seed} $out")
    done
  done
done
export EVAL=safety_prompts/test_all_440.jsonl
run() {
  id=$1; adapter=$2
  out=results/animal_tests/${id}_K5_responses.jsonl
  if [ ! -f "$out" ] || [ "$(wc -l < "$out")" -ne 2200 ]; then
    python3 scripts/evaluation/harmbench_multi_sample.py --hm7b-path trained_models/hm7b \
      --eval $EVAL --output-dir results/animal_tests --k 5 --out-suffix _K5 \
      --model-id "$id" --adapter "$adapter" > logs/test_${id}.log 2>&1 || echo "$id" >> logs/test_failures.log
  fi
  if [ ! -f results/animal_tests/${id}_statements_pressure.jsonl ]; then
    python3 scripts/evaluation/statements_pressure_infer.py --model-id "$id" --hm7b-path trained_models/hm7b \
      --adapter "$adapter" --output-dir results/animal_tests > logs/statements_${id}.log 2>&1 \
      || echo "statements $id" >> logs/statements_failures.log
  fi
}
export -f run
printf '%s\n' "${MODELS[@]}" | xargs -P 2 -L 1 bash -c 'run "$@"' _
touch logs/seeds_done.flag
