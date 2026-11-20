"""Build the animal-harm prompt pools and the 240-item safety test.

Sources, in order of preference:
  1. real prompts from Anthropic's red-team attempts data (the data the paper used), hand-selected,
     stored in safety_prompts/source/real_prompts.json with their original indices;
  2. prompts written by hand for the four categories (safety_prompts/written/*.json);
  3. template-generated prompts (pg_*.py), used only to fill the TRAINING pools.

The 240-item safety test and the two held-out evaluation sets (poison, misinfo) contain no templates.
Within each 60-prompt block (and each 100-prompt held-out set) the prompts are spread evenly over five
groups: companion, farmed, aquatic, wild, lab (12 / 20 per group).
"""
import json, math, random, re, sys, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pg_common import OPENERS, norm, with_opener
import pg_illegal, pg_poison, pg_speciesist, pg_misinfo
from animal_words import has_animal, group_of

ROOT = os.path.dirname(HERE) + '/'
OUT = ROOT + 'safety_prompts/'
SEED = 20261119
GROUPS = ['companion', 'farmed', 'aquatic', 'wild', 'lab']
OPENER_RATE = 0.16
CASUAL_LOWER = {'cruel5', 'theft3', 'fight10'}
N_SAFETY_PER_GROUP, N_EVAL_PER_GROUP = 12, 20     # 60 per safety block, 100 per held-out set
A_CANDIDATES, B_TRAIN = 700, 140                  # 700 candidates per narrow category follows the paper; 140 is ours

AN = re.compile(r'\b([Aa]) (?=[aeiouAEIOU])(?!(?:one|uni|use|eu))')


def toks(p):
    return set(re.findall(r'[a-z]+', p.lower()))


def jacc(a, b):
    return len(a & b) / max(1, len(a | b))


def clean(s):
    return re.sub(r'\s+', ' ', s).strip()


def polish(core, tid):
    core = AN.sub(lambda m: m.group(1) + 'n ', core)
    if tid not in CASUAL_LOWER:
        core = core[0].upper() + core[1:]
    return core.strip()


def lint(core):
    issues = []
    if re.search(r'\b(\w+) \1\b', core, re.I): issues.append('repeated word')
    if re.search(r'\b[aA] (?!one|uni|use|eu)[aeiouAEIOU]', core): issues.append('a/an')
    if re.search(r'\b[aA]n (?![aeiouAEIOU]|hour)\w', core): issues.append('an/a')
    if '  ' in core or '{' in core or '}' in core: issues.append('format')
    return issues


class Registry:
    """Accepts items in priority order; rejects exact and near duplicates within a category."""

    def __init__(self):
        self.seen = collections.defaultdict(list)   # category -> [(norm, toks)]
        self.claims = set()
        self.rejected = collections.Counter()

    def ok(self, cat, prompt, claim=None, templated=False):
        # near-duplicates are rejected for real and hand-written prompts; template output differs only in slot
        # words, so for templates only exact duplicates are rejected
        k, t = norm(prompt), toks(prompt)
        for k2, t2 in self.seen[cat]:
            if k == k2 or (not templated and jacc(t, t2) >= 0.8):
                self.rejected['near_duplicate'] += 1
                return False
        if not has_animal(prompt):
            self.rejected['no_animal_named'] += 1
            return False
        if claim is not None:
            if claim in self.claims:
                self.rejected['repeated_claim'] += 1
                return False
        self.seen[cat].append((k, t))
        if claim is not None:
            self.claims.add(claim)
        return True


def load_real():
    rows = json.load(open(ROOT + 'safety_prompts/source/real_prompts.json'))
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        p = clean(r['prompt'])
        x = {'prompt': p, 'source': 'hh_rlhf_real', 'rt_idx': r['rt_idx'], 'group': group_of(p)}
        if r['category'] == 'poison':
            x['subtype'] = r.get('subtype', 'poisoning')
        out[r['category']][group_of(p)].append(x)
    return out


def by_subtype(d, sub):
    return {g: [x for x in xs if x.get('subtype') == sub] for g, xs in d.items()}


def load_written(name, subtype=None):
    d = json.load(open(ROOT + f'safety_prompts/written/{name}.json'))
    out = {}
    for g, xs in d.items():
        g = 'companion' if g == 'companion_vet' else g   # vet-office poisoning prompts are companion animals
        out.setdefault(g, [])
        for x in xs:
            if isinstance(x, str):
                out[g].append({'prompt': x, 'source': 'hand_written', 'group': g, **({'subtype': subtype} if subtype else {})})
            else:
                out[g].append({'prompt': x['prompt'], 'claim': x['claim'], 'source': 'hand_written', 'group': g})
    return out


def pick_block(rng, per_group, sources_by_group, reg, cat, taken):
    """per_group items for each group, drawing real prompts first, then hand-written."""
    block = []
    for g in GROUPS:
        cands = []
        for lst in sources_by_group:           # each lst: dict group -> list (priority order)
            xs = [x for x in lst.get(g, []) if id(x) not in taken]
            rng.shuffle(xs)
            cands += xs
        got = 0
        for x in cands:
            if got >= per_group:
                break
            if reg.ok(cat, x['prompt'], x.get('claim')):
                taken.add(id(x))
                block.append({**x, 'category': cat})
                got += 1
        if got < per_group:
            raise SystemExit(f'not enough prompts for {cat}/{g}: {got} of {per_group}')
    return block


def template_fill(items, mod, need, seed, reg, cat, *, abusive=None, unique_claim=False, allow_short=False):
    rng = random.Random(seed)
    cap = math.ceil(1.7 * need / (len(abusive) if abusive is not None else mod.N_TEMPLATES))
    used = collections.Counter()
    stream = mod.stream(rng)
    guard = 0
    while len(items) < need and guard < 800000:
        guard += 1
        got = next(stream)
        tid, core = got[0], got[1]
        claim = got[2] if len(got) > 2 else None
        if abusive is not None and tid not in abusive:
            continue
        if used[tid] >= cap:
            continue
        core = polish(core, tid)
        if lint(core):
            continue
        if not reg.ok(cat, core, claim if unique_claim else None, templated=True):
            continue
        used[tid] += 1
        items.append({'prompt': core, 'source': 'template', 'template': tid, 'category': cat,
                      **({'claim': claim} if claim else {})})
    if len(items) < need and not allow_short:
        raise SystemExit(f'template fill short for {cat}: {len(items)} of {need}')
    return items


def add_openers(items, rng):
    for x in items:
        starts_statement = x['prompt'].startswith(("It ", "She ", "He ", "They ", "We ", "I'm ", "I've ", "The ", "My ", "Our ", "I "))
        if x['source'] == 'template' and not starts_statement and rng.random() < OPENER_RATE:
            x['prompt'] = with_opener(rng.choice(OPENERS), x['prompt'])


def main():
    import glob
    for f in glob.glob(OUT + '*.jsonl'):
        os.remove(f)
    rng = random.Random(SEED)
    reg = Registry()
    real = load_real()
    w_ill, w_mis, w_spe = (load_written(n) for n in ('illegal', 'misinfo', 'speciesist'))
    w_poi = load_written('poison', 'poisoning')
    w_exp = load_written('experimentation', 'experimentation')
    w_spe_extra = load_written('speciesist_extra')   # hand-written, used for the training pool only
    taken = set()
    safety, pools = {}, {}

    # ---- the 240 safety test and the two held-out sets: no templates
    safety['illegal'] = pick_block(rng, N_SAFETY_PER_GROUP, [real['illegal'], w_ill], reg, 'illegal', taken)
    safety['speciesist'] = pick_block(rng, N_SAFETY_PER_GROUP, [w_spe], reg, 'speciesist', taken)
    # held-out chem/bio analog: half poisoning, half harmful experimentation, in every animal group
    r_poi, r_exp = by_subtype(real['poison'], 'poisoning'), by_subtype(real['poison'], 'experimentation')
    safety['poison'] = (pick_block(rng, N_SAFETY_PER_GROUP // 2, [r_poi, w_poi], reg, 'poison', taken)
                        + pick_block(rng, N_SAFETY_PER_GROUP // 2, [r_exp, w_exp], reg, 'poison', taken))
    safety['misinfo'] = pick_block(rng, N_SAFETY_PER_GROUP, [w_mis], reg, 'misinfo', taken)
    pools['poison_B_eval'] = (pick_block(rng, N_EVAL_PER_GROUP // 2, [r_poi, w_poi], reg, 'poison', taken)
                              + pick_block(rng, N_EVAL_PER_GROUP // 2, [r_exp, w_exp], reg, 'poison', taken))
    pools['misinfo_B_eval'] = pick_block(rng, N_EVAL_PER_GROUP, [w_mis], reg, 'misinfo', taken)

    # ---- training pools: leftovers of real and hand-written prompts first, templates fill the rest
    left_poi = [x for g in list(real['poison'].values()) + list(w_poi.values()) + list(w_exp.values()) for x in g if id(x) not in taken]
    left_ill = [x for g in real['illegal'].values() for x in g if id(x) not in taken]
    left_mis = [x for g in w_mis.values() for x in g if id(x) not in taken]
    left_spe = [x for g in list(w_spe.values()) + list(w_spe_extra.values()) for x in g if id(x) not in taken]
    for lst in (left_poi, left_ill, left_mis, left_spe):
        rng.shuffle(lst)

    def seed_items(left, cat):
        out = []
        for x in left:
            if reg.ok(cat, x['prompt'], x.get('claim')):
                out.append({**x, 'category': cat})
        return out

    ill = template_fill(seed_items(left_ill, 'illegal'), pg_illegal, A_CANDIDATES, SEED + 1, reg, 'illegal')
    sp = template_fill(seed_items(left_spe, 'speciesist'), pg_speciesist, A_CANDIDATES, SEED + 2, reg, 'speciesist',
                       abusive=pg_speciesist.ABUSIVE, allow_short=True)
    po = template_fill(seed_items(left_poi, 'poison'), pg_poison.PoisonSplit('train'), B_TRAIN, SEED + 3, reg, 'poison')

    # misinformation training claims: old claim bank + labels, each used once, disjoint from the held-out claims
    held = [x['claim'] for x in safety['misinfo'] + pools['misinfo_B_eval'] if x.get('claim')]
    held_t = [toks(c) for c in held]
    claims = [c for f, c in pg_misinfo.BANK if not any(jacc(toks(c), h) >= 0.4 for h in held_t)]
    labels = list(pg_misinfo.ALL_LABELS)
    mis = template_fill(seed_items(left_mis, 'misinfo'), pg_misinfo.Split(claims, labels), B_TRAIN, SEED + 4, reg, 'misinfo',
                        unique_claim=True)
    # keep the training claim set disjoint from the held-out claim texts (checked after the fact too)
    pools['illegal_A_candidates'] = ill
    pools['speciesist_A_candidates'] = sp
    pools['poison_train'] = po
    pools['misinfo_train'] = mis

    for items in list(pools.values()):
        add_openers(items, rng)

    # ---- the 240: interleave categories so no two neighbours share a category
    allsafety = []
    cats = list(safety)
    per = {c: list(safety[c]) for c in cats}
    for c in cats:
        rng.shuffle(per[c])
    last = None
    while any(per.values()):
        order = [c for c in cats if per[c]]
        rng.shuffle(order)
        if last is not None and len(order) > 1 and order[0] == last:
            order[0], order[1] = order[1], order[0]
        for c in order:
            allsafety.append(per[c].pop())
        last = allsafety[-1]['category']

    names = {'illegal_A_candidates': f'illegal_A_candidates_{len(ill)}', 'speciesist_A_candidates': f'speciesist_A_candidates_{len(sp)}',
             'poison_train': f'poison_train_{len(po)}', 'misinfo_train': f'misinfo_train_{len(mis)}',
             'poison_B_eval': f'poison_B_eval_{len(pools["poison_B_eval"])}', 'misinfo_B_eval': f'misinfo_B_eval_{len(pools["misinfo_B_eval"])}'}
    for k, fn in names.items():
        with open(f'{OUT}{fn}.jsonl', 'w') as f:
            for x in pools[k]:
                f.write(json.dumps(x) + '\n')
    with open(f'{OUT}safety_test_{len(allsafety)}.jsonl', 'w') as f:
        for x in allsafety:
            f.write(json.dumps(x) + '\n')

    # ---- readable copy of the safety test
    titles = {'illegal': 'Illegal harm to animals (trained category)', 'speciesist': 'Speciesist abuse (trained category)',
              'poison': 'Poisoning and harmful experimentation (held-out category)', 'misinfo': 'Misinformation about animals (held-out category)'}
    gnames = {'companion': 'Companion animals', 'farmed': 'Farmed animals', 'aquatic': 'Fish and aquatic animals', 'wild': 'Wild animals', 'lab': 'Laboratory animals'}
    md = ['# Safety test: 240 prompts\n', 'Each prompt is tagged **[real]** (from Anthropic\'s red-team attempts data) or **[written]** (written by hand); in the held-out category the tag also says poisoning or experimentation. Within each category the prompts are grouped by animal group, 12 per group.\n']
    for c in ['illegal', 'speciesist', 'poison', 'misinfo']:
        md.append(f'\n## {titles[c]} ({len(safety[c])})\n')
        for g in GROUPS:
            md.append(f'\n### {gnames[g]}\n')
            for x in [x for x in safety[c] if x['group'] == g]:
                tag = {'poisoning': ' poisoning', 'experimentation': ' experimentation'}.get(x.get('subtype'), '') if c == 'poison' else ''
                md.append(f"- [{'real' if x['source'] == 'hh_rlhf_real' else 'written'}{tag}] {x['prompt']}")
    open(OUT + 'safety_test_240.md', 'w').write('\n'.join(md) + '\n')

    # ---- report
    report = {'seed': SEED, 'rejected_during_build': dict(reg.rejected), 'blocks': {}, 'real_selected_total': sum(len(v) for c in real.values() for v in c.values())}
    def summarize(items):
        return {'n': len(items), 'by_source': dict(collections.Counter(x['source'] for x in items)),
                'by_group': dict(collections.Counter(x.get('group', 'template (untagged)') for x in items)),
                'by_subtype': dict(collections.Counter(x.get('subtype', '-') for x in items))}
    for c, items in safety.items():
        report['blocks'][f'safety_test/{c}'] = summarize(items)
    for k, items in pools.items():
        report['blocks'][k] = summarize(items)
    report['safety_test_by_source'] = dict(collections.Counter(x['source'] for x in allsafety))
    json.dump(report, open(OUT + 'build_report.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
