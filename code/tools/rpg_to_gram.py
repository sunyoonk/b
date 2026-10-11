"""Optional: GRAM-format raw files (<out>/<Name>/user_sequence.txt, item_plain_text.txt) from the genrec AmazonReviews2014
cache, for categories whose GRAM release is not at hand. Sequences = processed/all_item_seqs.json (identical to GRAM on
Beauty: 22,363/22,363 users). Item text = raw meta_<Category>.json.gz rendered as GRAM does
("title: ...; brand: ...; categories: ...; description: ...; price: ...; salesrank: <cat>: <rank>", lowercased, with
the characters & - % ' " / ( ) ? # ~ ! + and tokens longer than 15 letters removed, missing brand = na); this text only feeds the LLM profile prompts. Match rate against the real GRAM text
is printed when --check points at a GRAM file.
usage: python rpg_to_gram.py <cache>/AmazonReviews2014/<Category> <out dir>/<Name> [--check <GRAM item_plain_text.txt>]"""
import ast, gzip, json, os, re, sys

src, out = sys.argv[1], sys.argv[2]
check = sys.argv[sys.argv.index('--check') + 1] if '--check' in sys.argv else None
cat = os.path.basename(os.path.normpath(src))
seqs = json.load(open(f'{src}/processed/all_item_seqs.json'))
items = {i for s in seqs.values() for i in s}
DROP = re.compile(r"[&\-%'\"/()?#~!+]")
MAXTOK = 15   # GRAM drops run-together tokens longer than this


def clean(x):
    toks = DROP.sub('', str(x).lower()).split()
    return ' '.join(t for t in toks if len(t.strip('.,;:!?')) <= MAXTOK)


def render(d):
    """every field present; absent key -> 'na' (GRAM convention), present-but-empty -> ''"""
    def f(k, v=None):
        if k not in d or d[k] is None:
            return 'na'
        return clean(d[k]) if v is None else v
    cats = ', '.join(clean(c) for c in d['categories'][0]) if d.get('categories') else 'na'
    rank = ', '.join(f'{clean(k)}: {v}' for k, v in d['salesRank'].items()) if d.get('salesRank') else 'na'
    return '; '.join([f"title: {f('title')}", f"brand: {f('brand')}", f'categories: {cats}', f"description: {f('description') or 'na'}",
                      f"price: {f('price', str(d.get('price')))}", f'salesrank: {rank}'])


text = {}
for line in gzip.open(f'{src}/raw/meta_{cat}.json.gz', 'rt'):
    d = ast.literal_eval(line)
    if d['asin'] in items:
        text[d['asin']] = render(d)
os.makedirs(out, exist_ok=True)
with open(f'{out}/user_sequence.txt', 'w') as f:
    for u, s in seqs.items():
        f.write(' '.join([u] + s) + '\n')
with open(f'{out}/item_plain_text.txt', 'w') as f:
    for i in sorted(items):
        f.write(f'{i} {text.get(i, "")}\n')
print(f'{len(seqs)} users, {len(items)} items ({len(items) - len(text)} without meta) -> {out}')
if check:
    ref = {}
    for line in open(check):
        a, _, b = line.rstrip('\n').partition(' '); ref[a] = b
    same = sum(ref.get(i) == text.get(i, '') for i in items)
    print(f'exact text match with GRAM: {same}/{len(items)}')
