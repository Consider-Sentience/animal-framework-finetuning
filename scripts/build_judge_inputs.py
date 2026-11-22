"""Copy the safety and held-out response files with a 'prompt' field, which the authors' batch judge reads."""
import json
from pathlib import Path

src = Path("results/animal_tests")
dst = Path("results/animal_tests_judge")
dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob("*_K5_responses.jsonl")):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    with open(dst / f.name, "w") as out:
        for r in rows:
            r["prompt"] = r["behavior"]
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f.name, len(rows))
