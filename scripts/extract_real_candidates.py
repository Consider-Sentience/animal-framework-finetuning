"""Extract animal-related first-turn prompts from Anthropic's red-team attempts data (the data the paper used).

Usage: python3 scripts/extract_real_candidates.py red_team_attempts.jsonl.gz out.json
The file is a single JSON array of transcripts. Keeps the first human turn when it is 12-300 characters and
contains an animal word, with exact duplicates removed. The prompts in safety_prompts/source/real_prompts.json
were then selected from this list by reading it (only clear requests to harm animals; ambiguous ones such as
euthanasia, hunting for food, pest control and slaughter were left out).
"""
import gzip, json, re, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from animal_words import SPECIES_RX

src, out = sys.argv[1], sys.argv[2]
data = json.load(gzip.open(src))
rows, seen = [], set()
for i, t in enumerate(data):
    m = re.match(r'\s*Human:\s*(.*?)\n\s*\n\s*Assistant:', t['transcript'], re.S)
    if not m:
        continue
    p = re.sub(r'\s+', ' ', m.group(1)).strip()
    if not 12 <= len(p) <= 300 or not SPECIES_RX.search(p):
        continue
    k = re.sub(r'[^a-z]', '', p.lower())
    if k in seen:
        continue
    seen.add(k)
    rows.append({'idx': i, 'prompt': p, 'harm_score': t['min_harmlessness_score_transcript'], 'tags': t['tags']})
json.dump(rows, open(out, 'w'))
print(len(rows), 'candidates')
