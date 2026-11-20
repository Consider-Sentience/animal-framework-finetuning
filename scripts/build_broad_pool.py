"""Draw the broad training pool: 125 prompts from each of four pools.

Same draw call as the authors' build_narrow_pools.py (random.Random(seed).sample,
no stratification). Seed 55, fixed in the plan before any draw.

Reads (all in safety_prompts/):
  speciesist_A_clean_500.jsonl, illegal_A_clean_500.jsonl,
  misinfo_train_140.jsonl (minus misinfo_train_exclusions.json), poison_train_140.jsonl
Writes: broad_A_500.jsonl and broad_pool_report.md

Run from the project folder:  python3 scripts/build_broad_pool.py
"""
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "safety_prompts"
SEED = 55
N = 125


def read(name):
    return [json.loads(l) for l in open(SP / name) if l.strip()]


pools = {
    "speciesist": read("speciesist_A_clean_500.jsonl"),
    "illegal": read("illegal_A_clean_500.jsonl"),
}
excl = set(json.load(open(SP / "misinfo_train_exclusions.json"))["excluded_templates"])
mis = [r for r in read("misinfo_train_140.jsonl") if r["template"] not in excl]
pools["misinfo"] = mis
pools["poison"] = read("poison_train_140.jsonl")

out = []
for cat in ("speciesist", "illegal", "misinfo", "poison"):
    pool = pools[cat]
    assert len(pool) >= N, (cat, len(pool))
    picked = random.Random(SEED).sample(pool, N)
    for r in picked:
        r = dict(r)
        r["category"] = cat
        out.append(r)

with open(SP / "broad_A_500.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# checks
held = set()
for name in ("safety_test_240.jsonl", "poison_B_eval_100.jsonl", "misinfo_B_eval_100.jsonl"):
    held |= {r["prompt"] for r in read(name)}
overlap = [r["prompt"] for r in out if r["prompt"] in held]
dupes = len(out) - len({r["prompt"] for r in out})
lines = [
    "# Broad training pool report\n",
    f"Seed {SEED}, {N} prompts drawn from each of four pools, {len(out)} in all.\n",
    "\n| Pool | Prompts in pool | Drawn |\n|---|---|---|\n",
]
for cat in ("speciesist", "illegal", "misinfo", "poison"):
    lines.append(f"| {cat} | {len(pools[cat])} | {N} |\n")
lines.append(f"\nPrompts also in the safety test or held-out sets: {len(overlap)}\n")
lines.append(f"Duplicate prompts within the pool: {dupes}\n\n")
lines.append("Source of the drawn prompts:\n\n")
for cat in ("speciesist", "illegal", "misinfo", "poison"):
    c = Counter(r.get("source", "?") for r in out if r["category"] == cat)
    lines.append(f"- {cat}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + "\n")
(SP / "broad_pool_report.md").write_text("".join(lines))
print("".join(lines))
assert not overlap and dupes == 0
