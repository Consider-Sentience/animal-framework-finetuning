"""
Build the three harmful-slot input files for the full critique/revision generation (no API calls).
One file per training set: broad (500), speciesist abuse (500), illegal harm (500). Each row is a prompt
from the clean pool joined to its HM7B first answer (initial_response), with source_subset and
source_index kept, because the instruction draw is seeded from them.

Usage (from the project folder, on the pod where results/first_answers lives):
  python3 scripts/build_generation_inputs.py --out gen/inputs
"""
import argparse, json
from pathlib import Path

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = {
        "broad": (None, "results/first_answers/broad_A_500.jsonl"),
        "speciesist": ("safety_prompts/speciesist_A_clean_500.jsonl", "results/first_answers/speciesist_A_candidates_552.jsonl"),
        "illegal": ("safety_prompts/illegal_A_clean_500.jsonl", "results/first_answers/illegal_A_candidates_700.jsonl"),
    }
    for name, (clean, first) in jobs.items():
        answers = load(first)
        if clean is None:
            rows = answers
        else:
            by_prompt = {r["prompt"]: r for r in answers}
            rows = [by_prompt[r["prompt"]] for r in load(clean)]
        rows = [{"source_subset": r["source_subset"], "source_index": r["source_index"], "prompt": r["prompt"], "initial_response": r["initial_response"]} for r in rows]
        assert len(rows) == 500 and all(r["initial_response"].strip() for r in rows), name
        assert len({(r["source_subset"], r["source_index"]) for r in rows}) == 500, name
        with open(out / f"harmful_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(name, len(rows))

if __name__ == "__main__":
    main()
