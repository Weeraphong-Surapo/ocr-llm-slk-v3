"""Benchmark Typhoon OCR configurations against a ground-truth sample (stdlib only).

Run next to Ollama, e.g.:
  docker run --rm --network container:<ollama> -v $PWD/eval:/eval python:3.11-slim \
    python /eval/run_typhoon.py --sample /eval/samples/receipt_06987.json \
      --image /eval/work/receipt.png --image /eval/work/receipt_2x.png \
      --anchor /eval/work/paddle.txt --out /eval/work/results
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import report, score  # noqa: E402
PROMPT_15 = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""
def prompt_7b_default(base_text):
    return ("Below is an image of a document page along with its dimensions. "
            "Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
            "If the document contains images, use a placeholder like dummy.png for each image.\n"
            "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
            f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END")
def prompt_7b_structure(base_text):
    return ("Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
            "Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
            "Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
            "If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
            "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
            f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END")
def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big')
def extract_natural_text(text):
    try:
        value = json.loads(text)
        if isinstance(value, dict) and 'natural_text' in value:
            return str(value['natural_text'])
    except ValueError:
        pass
    return text
def generate(url, model, prompt, image_path, options):
    image = base64.b64encode(open(image_path, 'rb').read()).decode()
    body = json.dumps({'model': model, 'prompt': prompt, 'images': [image], 'stream': False, 'options': options}).encode()
    request = urllib.request.Request(f'{url}/api/generate', data=body, headers={'Content-Type': 'application/json'})
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.load(response)
    return data.get('response', ''), time.time() - started, data.get('done_reason')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ollama', default='http://localhost:11434')
    parser.add_argument('--sample', required=True)
    parser.add_argument('--image', action='append', required=True)
    parser.add_argument('--anchor', help='PaddleOCR text used as a hint for the 7B structure prompt')
    parser.add_argument('--out', required=True)
    parser.add_argument('--models', default='scb10x/typhoon-ocr1.5-3b,scb10x/typhoon-ocr-7b')
    args = parser.parse_args()
    sample = json.load(open(args.sample, encoding='utf-8'))
    anchor = open(args.anchor, encoding='utf-8').read() if args.anchor else ''
    os.makedirs(args.out, exist_ok=True)
    models = args.models.split(',')
    runs = []
    for image in args.image:
        w, h = png_size(image)
        dims = f'Page dimensions: {w:.1f}x{h:.1f}'
        if 'scb10x/typhoon-ocr1.5-3b' in models:
            runs.append(('3b-1.5', 'scb10x/typhoon-ocr1.5-3b', PROMPT_15, image,
                         {'temperature': 0.1, 'top_p': 0.6, 'repeat_penalty': 1.1, 'num_predict': 4096}))
        if 'scb10x/typhoon-ocr-7b' in models:
            opts = {'temperature': 0.1, 'top_p': 0.6, 'repeat_penalty': 1.2, 'num_predict': 4096}
            runs.append(('7b-default', 'scb10x/typhoon-ocr-7b', prompt_7b_default(dims), image, opts))
            if anchor:
                runs.append(('7b-structure+paddle', 'scb10x/typhoon-ocr-7b', prompt_7b_structure(f'{dims}\n{anchor}'), image, opts))
    summary = []
    for label, model, prompt, image, options in runs:
        name = f'{label}@{os.path.basename(image)}'
        try:
            raw, seconds, reason = generate(args.ollama, model, prompt, image, options)
        except Exception as exc:  # keep going so one missing model does not stop the benchmark
            print(f'{name}: ERROR {type(exc).__name__}: {exc}')
            continue
        text = extract_natural_text(raw)
        with open(os.path.join(args.out, name.replace('/', '_') + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        result = score(sample, text)
        report(f'{name} [{seconds:.1f}s, stop={reason}]', result)
        summary.append((result['correct'], -len(result['invented']), name))
    if summary:
        best = max(summary)
        print(f'\nBEST: {best[2]} ({best[0]}/{len(sample["fields"])} exact, invented={-best[1]})')
if __name__ == '__main__':
    main()
