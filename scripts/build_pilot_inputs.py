"""
Build the small input files for the critique/revision pilot (no API calls).

Harmful slot: PER_POOL prompts drawn at random (seed 42) from each of the two clean 500-prompt pools,
joined to the HM7B first answers (field initial_response). Helpful slot: the first N prompts of
safety_prompts/helpful_500.jsonl (already a random draw).

Usage (from the project folder):
  python3 scripts/build_pilot_inputs.py --n-per-pool 10 --out pilot/inputs
"""
import argparse, json, random
from pathlib import Path

POOLS = {
    "speciesist": ("safety_prompts/speciesist_A_clean_500.jsonl", "results/first_answers/speciesist_A_candidates_552.jsonl"),
    "illegal": ("safety_prompts/illegal_A_clean_500.jsonl", "results/first_answers/illegal_A_candidates_700.jsonl"),
}

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-pool", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    harmful = []
    for name, (clean, first) in POOLS.items():
        answers = {r["prompt"]: r for r in load(first)}
        pool = [r for r in load(clean) if r["prompt"] in answers]
        for r in rng.sample(pool, a.n_per_pool):
            a_ = answers[r["prompt"]]
            harmful.append({"source_subset": a_["source_subset"], "source_index": a_["source_index"], "prompt": r["prompt"], "initial_response": a_["initial_response"], "pool": name})
    rng.shuffle(harmful)
    helpful = load("safety_prompts/helpful_500.jsonl")[: 2 * a.n_per_pool]
    for fn, rows in (("harmful_pilot.jsonl", harmful), ("helpful_pilot.jsonl", helpful)):
        with open(out / fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {len(harmful)} harmful and {len(helpful)} helpful to {out}")

if __name__ == "__main__":
    main()
