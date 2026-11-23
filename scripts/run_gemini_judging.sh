#!/bin/bash
# Second judge (Gemini 3.8 Flash) on the headline models' held-out answers only. Needs GOOGLE_API_KEY.
# Tier 1 allows 10,000 generate-content requests per model per day, far below the ~63,000 a full second pass needs,
# so this covers only the held-out set of each speciesist and illegal model (poison for speciesist, misinfo for illegal),
# including the seed 1 and 2 runs: 100 held-out prompts x 5 answers = 500 requests per model.
# Resumable. Run it again after the daily quota resets; files that finished are skipped.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/animal_tests_judge_subset results/animal_tests_judged logs
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("results/animal_tests_judge/*_K5_responses.jsonl")):
    id_ = os.path.basename(f).replace("_K5_responses.jsonl", "")
    keep = "heldout_poison" if id_.startswith("speciesist_") else "heldout_misinfo" if id_.startswith("illegal_") else None
    if not keep or "virtuous" in id_ or ("subordinate" in id_ and "_s" not in id_):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if str(r.get("id", r.get("behavior_id", ""))).startswith(keep)]
    with open(f"results/animal_tests_judge_subset/{id_}_K5_responses.jsonl", "w") as o:
        for r in rows: o.write(json.dumps(r, ensure_ascii=False) + "\n")
PY
for f in results/animal_tests_judge_subset/*_K5_responses.jsonl; do
  id=$(basename "$f" _K5_responses.jsonl)
  python3 scripts/evaluation/judge_gemini_harmbench.py --input "$f" --output results/animal_tests_judged/${id}_K5_gemini.jsonl \
    --max-workers 8 >> logs/judge_safety_gemini.log 2>&1 || echo "gemini subset $id" >> logs/judge_failures.log
done
