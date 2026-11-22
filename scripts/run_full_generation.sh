#!/usr/bin/env bash
# Full critique/revision generation: 4 constitutions x 3 training sets (broad, speciesist, illegal).
# The helpful slot (500 HH-RLHF helpful-base prompts, answered directly) is generated once and shared by
# the three training sets; the harmful slot differs per set. Resumable: finished records are skipped.
# Needs DEEPINFRA_API_KEY in the environment and gen/inputs from scripts/build_generation_inputs.py.
set -u
CONS="welfarist rights_based virtuous_agent subordinate_agent"
COMMON="--constitutions-dir constitutions --constitutions $CONS --max-workers ${WORKERS:-12}"
python3 scripts/cai_dataset_generator_hermes.py --hm7b-responses gen/inputs/harmful_broad.jsonl \
  --helpful-prompts safety_prompts/helpful_500.jsonl --output-dir gen/helpful --slots helpful $COMMON \
  > logs/gen_helpful.log 2>&1
for s in broad speciesist illegal; do
  mkdir -p gen/$s/_intermediate
  cp gen/helpful/_intermediate/*_helpful_hermes.jsonl gen/$s/_intermediate/
  python3 scripts/cai_dataset_generator_hermes.py --hm7b-responses gen/inputs/harmful_$s.jsonl \
    --helpful-prompts safety_prompts/helpful_500.jsonl --output-dir gen/$s --slots harmful helpful $COMMON \
    > logs/gen_$s.log 2>&1 &
done
wait
echo ALL_DONE > logs/gen_done.flag
