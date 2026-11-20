"""
Build the two clean 500-prompt narrow training pools from the screen results, following the
authors' scripts/data_prep/build_narrow_pools.py step for step:

  1. Load the judged pool (our results/pool_screen.jsonl, joined back to the candidate files).
  2. Drop records that were not judged by both judges (the authors drop records with errors or no
     judge result; here that is any prompt where a judge gave NO_ANSWER or UNPARSED).
  3. Drop records flagged by BOTH judges (is_b_like).
  4. Draw 500 of the rest with random.Random(seed).sample, no stratification:
       seed 53 for the speciesist pool (the authors' pair 1) and seed 54 for the illegal pool (pair 2).
  5. Halt with a message if fewer than 500 remain (the authors treat that as a methodological flag).

Also writes the dropped records and a report of how the draw changed the animal-group mix.

Usage (from the project folder):
  python3 scripts/build_clean_pools.py --screen results/pool_screen.jsonl --out safety_prompts
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

POOLS = {
    "speciesist": {"candidates": "speciesist_A_candidates_552.jsonl", "seed": 53},
    "illegal": {"candidates": "illegal_A_candidates_700.jsonl", "seed": 54},
}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def mix(rows):
    return dict(sorted(collections.Counter(r.get("group", "(template, no group)") for r in rows).items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="results/pool_screen.jsonl")
    ap.add_argument("--out", required=True, help="folder holding the candidate files; clean pools are written here")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out)
    screen = load(a.screen)
    report = ["# Clean-pool build (authors' build_narrow_pools.py procedure)", ""]
    for name, cfg in POOLS.items():
        cand = load(out / cfg["candidates"])
        res = {(r["pool"], r["prompt"]): r for r in screen}
        judged, unjudged, flagged = [], [], []
        for c in cand:
            r = res.get((cfg["candidates"], c["prompt"]))
            if r is None or r["claude"] not in ("YES", "NO") or r["gemini"] not in ("YES", "NO"):
                unjudged.append({**c, "claude": r and r["claude"], "gemini": r and r["gemini"]})
            elif r["is_b_like"]:
                flagged.append({**c, "claude": r["claude"], "gemini": r["gemini"]})
            else:
                judged.append({**c, "claude": r["claude"], "gemini": r["gemini"]})
        report += [f"## {name} pool", f"- candidates: {len(cand)}",
                   f"- not judged by both judges (dropped, as in the authors' script): {len(unjudged)}",
                   f"- flagged by both judges (dropped): {len(flagged)}",
                   f"- clean: {len(judged)}"]
        if len(judged) < a.n:
            sys.exit(f"{name}: only {len(judged)} clean prompts, need {a.n}. Stop: this is a methodological flag.")
        final = random.Random(cfg["seed"]).sample(judged, a.n)
        with open(out / f"{name}_A_clean_{a.n}.jsonl", "w") as f:
            for r in final:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("claude", "gemini")}, ensure_ascii=False) + "\n")
        with open(out / f"{name}_A_dropped.jsonl", "w") as f:
            for r in unjudged + flagged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report += [f"- seed {cfg['seed']}, drew {a.n}",
                   f"- group mix, all candidates: {mix(cand)}",
                   f"- group mix, clean pool: {mix(judged)}",
                   f"- group mix, final {a.n}: {mix(final)}",
                   f"- source mix, final {a.n}: {dict(collections.Counter(r['source'] for r in final))}", ""]
    (out / "clean_pools_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
