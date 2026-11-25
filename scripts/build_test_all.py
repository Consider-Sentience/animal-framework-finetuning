"""Join the 240-item safety test and the two 100-item held-out sets into the 440-item file the models are tested on.

Reads (all in safety_prompts/): safety_test_240.jsonl, poison_B_eval_100.jsonl, misinfo_B_eval_100.jsonl
Writes: safety_prompts/test_all_440.jsonl, in that order, with ids safety-N, heldout_poison-N and heldout_misinfo-N.

Run from the project folder:  python3 scripts/build_test_all.py
"""
import json
from pathlib import Path

SP = Path(__file__).resolve().parent.parent / "safety_prompts"
PARTS = [("safety", "safety_test_240.jsonl"), ("heldout_poison", "poison_B_eval_100.jsonl"),
         ("heldout_misinfo", "misinfo_B_eval_100.jsonl")]

rows = []
for test_set, name in PARTS:
    for i, r in enumerate(l for l in open(SP / name) if l.strip()):
        r = json.loads(r)
        rows.append({"id": f"{test_set}-{i}", "behavior_id": f"{test_set}-{i}", "behavior": r["prompt"],
                     "category": r["category"], "test_set": test_set, "source": r["source"], "group": r["group"]})
assert len(rows) == 440
with open(SP / "test_all_440.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} prompts to {SP / 'test_all_440.jsonl'}")
