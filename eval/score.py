"""Score OCR output text against a ground-truth sample (stdlib only).

Usage: python eval/score.py eval/samples/receipt_06987.json output.txt [more outputs...]
A field counts as correct only if its exact value appears in the output after
removing whitespace, Markdown emphasis and HTML tags. Order and layout are ignored.
"""
import json
import re
import sys
def normalize(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('**', '').replace('\\_', '_')
    return re.sub(r'\s+', '', text)
def score(sample: dict, output: str) -> dict:
    out = normalize(output)
    missing = [f for f in sample['fields'] if normalize(f['value']) not in out]
    invented = [s for s in sample.get('must_not_contain', []) if normalize(s) in out]
    total = len(sample['fields'])
    return {'correct': total - len(missing), 'total': total, 'missing': missing, 'invented': invented}
def report(name: str, result: dict) -> None:
    pct = 100 * result['correct'] / result['total']
    print(f"{name}: {result['correct']}/{result['total']} fields exact ({pct:.0f}%), invented={len(result['invented'])}")
    for f in result['missing']:
        print(f"  missing {f['name']}: {f['value']}{' (verify ground truth)' if f.get('verify') else ''}")
    for s in result['invented']:
        print(f"  INVENTED: {s}")
if __name__ == '__main__':
    sample = json.load(open(sys.argv[1], encoding='utf-8'))
    for path in sys.argv[2:]:
        report(path, score(sample, open(path, encoding='utf-8').read()))
