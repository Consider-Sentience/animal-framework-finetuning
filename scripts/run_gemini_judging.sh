#!/bin/bash
# Second judge (Gemini 3.8 Flash) on the safety answers. Needs GOOGLE_API_KEY.
# Judged: every answer of HM7B and the broad models, and the held-out answers of each narrow model
# (poisoning for speciesist training, misinformation for illegal-harm training), including the seed 1 and 2 runs.
# Resumable: files that are finished are skipped.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/animal_tests_judge_subset results/animal_tests_judged logs
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("results/animal_tests_judge/*_K5_responses.jsonl")):
    id_ = os.path.basename(f).replace("_K5_responses.jsonl", "")
    keep = "heldout_poison" if id_.startswith("speciesist_") else "heldout_misinfo" if id_.startswith("illegal_") else None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if keep:
        rows = [r for r in rows if str(r.get("id", r.get("behavior_id", ""))).startswith(keep)]
    with open(f"results/animal_tests_judge_subset/{id_}_K5_responses.jsonl", "w") as o:
        for r in rows: o.write(json.dumps(r, ensure_ascii=False) + "\n")
PY
for f in results/animal_tests_judge_subset/*_K5_responses.jsonl; do
  id=$(basename "$f" _K5_responses.jsonl)
  python3 scripts/evaluation/judge_gemini_harmbench.py --input "$f" --output results/animal_tests_judged/${id}_K5_gemini.jsonl \
    --max-workers 8 >> logs/judge_safety_gemini.log 2>&1 || echo "gemini subset $id" >> logs/judge_failures.log
done
