#!/usr/bin/env bash
# Train the 18 models: 6 constitutions x (broad, speciesist = narrow pair 1, illegal = narrow pair 2).
# Each model is a LoRA adapter on top of HM7B, trained on gen/<set>/<constitution>_cai_base_hermes.jsonl with the
# settings in scripts/training/train_cai.py (r=64, alpha=128, 3 epochs, lr 2e-4, 90/10 split, seed 42).
# Resumable: a model whose adapter already exists is skipped. Needs one A100 and trained_models/hm7b.
set -u
for s in broad speciesist illegal; do
  for c in ${CONS:-consequentialist welfarist deontological rights_based virtuous_agent subordinate_agent}; do
    out=trained_models/$s/$c
    if [ -f $out/adapter_model.safetensors ]; then echo "skip $s $c"; continue; fi
    python3 scripts/training/train_cai.py --constitution $c --dataset gen/$s/${c}_cai_base_hermes.jsonl \
      --hm7b-path trained_models/hm7b --output-dir $out > logs/train_${s}_${c}.log 2>&1 \
      || echo "FAILED $s $c" >> logs/train_failures.log
  done
done
echo TRAIN_DONE > logs/train_done.flag
