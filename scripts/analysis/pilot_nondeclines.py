"""Print the final answers that the pilot decline counter did not flag, for reading by a person.
Usage: python3 scripts/analysis/pilot_nondeclines.py pilot/out subordinate_agent [chars]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pilot_diagnostics import DECLINE
root, const = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rows = [json.loads(l) for l in open(Path(root) / "_intermediate" / f"{const}_harmful_hermes.jsonl")]
k = 0
for i, r in enumerate(rows):
    if not DECLINE.search(r["response"]):
        k += 1
        print(f"[{i}] {r['prompt'][:90]}\n    -> {r['response'][:n].replace(chr(10), ' ')}\n")
print(f"{k} of {len(rows)} not flagged as declines")
