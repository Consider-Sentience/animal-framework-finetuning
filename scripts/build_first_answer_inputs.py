"""Tag the three training pools for HM7B's first answers (scripts/hm7b_inference.py).

Each prompt gets source_subset (the pool's file name) and source_index (its position in the file), which the
generation step later uses to seed its draws. No model or API calls.

Reads (in safety_prompts/): broad_A_500.jsonl, speciesist_A_candidates_552.jsonl, illegal_A_candidates_700.jsonl
Writes the same three file names to --out.

Run from the project folder:  python3 scripts/build_first_answer_inputs.py --out results/first_answers/inputs
"""
import argparse, json
from pathlib import Path

POOLS = ["broad_A_500", "speciesist_A_candidates_552", "illegal_A_candidates_700"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for name in POOLS:
    rows = [json.loads(l) for l in open(f"safety_prompts/{name}.jsonl") if l.strip()]
    with open(out / f"{name}.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({**r, "source_index": i, "source_subset": name}, ensure_ascii=False) + "\n")
    print(name, len(rows))
