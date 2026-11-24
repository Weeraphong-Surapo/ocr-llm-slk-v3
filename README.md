# OCR LLM Service (Hono)

โครงสร้างเริ่มต้นสำหรับรับ PDF/image + prompt แล้วประมวลผลแบบ asynchronous โดย Hono เป็น public API และ Python OCR engine เป็น internal service

## Architecture

```text
Client
  -> Hono API (upload/validation/job/status/result)
  -> Redis/BullMQ (Phase 2; current scaffold has injectable in-process queue)
  -> OCR worker, concurrency=1
  -> Python FastAPI + PaddleOCR (GPU)
  -> PostgreSQL (Phase 2; current scaffold uses memory store for tests)
```

เหตุผลที่ใช้ queue: GPU 16 GB ไม่ควรรับ inference พร้อมกันหลาย request. ผู้ใช้ 20–50 คนรองรับได้ในรูปแบบ concurrent uploads + queued processing แต่ throughput จริงต้อง benchmark จากชนิดเอกสาร จำนวนหน้า และ OCR/LLM model ที่เลือก

## Run current API scaffold

```bash
cp .env.example .env
npm install
npm run typecheck
npm test
npm run build
npm run dev
```

API:
- `GET /health`
- `GET /api/openapi.json`
- `GET /api/docs/` (Swagger UI)
- `POST /api/v1/jobs` multipart: `file`, `prompt`, optional `response_format` and `response_schema`
- `GET /api/v1/jobs/:id`
- `POST /api/v1/jobs/:id/cancel`
- `GET /api/v1/jobs/:id/result`

`response_schema` is an inline JSON Schema object serialized as a multipart string. It constrains
the JSON returned in `result.answer.content`. For structured answers, the engine groups adjacent
PaddleOCR cells by row and aligns complete label and decimal-value columns with a consistent
vertical offset. The engine validates the answer against the schema before returning it.

Mark nullable decimal strings with a `pattern` and `"x-ocr-decimal": true`. Empty strings and
strings containing only whitespace are treated as null for these fields.
For a single-page PDF that PaddleOCR handled in selective mode, an invalid decimal
triggers one Typhoon image read. The engine only fills flagged fields with values found in
the reread and keeps the original row order. Unresolved fields remain `null` and appear in
`answer.review` with status `invalid_decimal` and a field `location`.

For item tables, add an `x-ocr-item-table` object to the schema with `columns`
(decimal field names mapped to recognized header labels), `anchors` (item keys
such as `name` or `code`), and optional `derived` (calculated item fields mapped
to a source column). With recognizable headers and rows, the engine verifies
item decimals against the corresponding cells. The selective reread crops the
item table when PaddleOCR can locate all output rows. Both passes use the
job's prompt. Calculations and interpretations that the prompt requests can
include the source cell text. The response keeps canonical decimal values
in `answer.content` and returns verified values and their source text, row,
column, page and method in `answer.decimal_evidence`. A marked item field can
only be filled on reread when supported by cell evidence. For unresolved cells
that the reread identifies as text, the engine can ask the answer model to
interpret just those cells under the same job prompt.

Cancellation removes a queued job immediately and returns HTTP 200 with status `cancelled`.
A running job returns HTTP 202 with status `cancelling`. The OCR engine stops scheduling new work
and closes in-flight Ollama requests for image reading and answer generation. A PaddleOCR call
already running can finish before the engine returns. The job then releases its queue slot and
moves to `cancelled` without a result.
Poll `GET /api/v1/jobs/:id` to see the final status. Repeated cancellation requests return
the current cancellation status. Cancelling a completed or failed job returns HTTP 409.
If communication with the engine fails before it confirms the stop, the API reports `failed`.

Current development limitation: job store is in-memory and queue is in-process. Restart loses jobs. Production must replace these with PostgreSQL + Redis/BullMQ. The Coolify Compose file reserves the production services and environment contract, but the current API scaffold does not yet consume Redis/PostgreSQL.

## OCR model

The OCR engine now loads this model at startup:

- PaddleOCR 3.7.0 (PaddleX 3.7.2)
- `PP-OCRv5_mobile_det` + `th_PP-OCRv5_mobile_rec` (Thai)
- `OCR_DEVICE=gpu:0` in Coolify
- PDF pages rendered one at a time with PyMuPDF
- Images whose long side is under `OCR_UPSCALE_MIN_LONG_SIDE` (default 2000 px) are enlarged up to `OCR_UPSCALE_MAX_FACTOR` (default 4x) before OCR so small Thai glyphs stay legible. `bbox` values are always returned in original-image pixels. Set `OCR_UPSCALE_MIN_LONG_SIDE=0` to disable.

If the model cannot load, `/health` reports `model_ready: false` and `/ocr` returns HTTP 503; it never fabricates OCR output. GPU/model startup still requires verification on the actual Ubuntu/NVIDIA host.

## Thai accuracy: Typhoon OCR + PaddleOCR cross-check

Each page is read by two models at the same time:

- **Typhoon OCR 1.5 3B** (`scb10x/typhoon-ocr1.5-3b`, served by the project's own `ollama` service) produces the main text as Markdown, with tables as HTML. It reads Thai far better than PaddleOCR but can invent plausible text or misread digits.
- **PaddleOCR** provides `lines` with `bbox`/`confidence` and cross-checks Typhoon. Typhoon words with no similar PaddleOCR text (`REVIEW_MIN_SIMILARITY`, default 0.6), and numbers/emails PaddleOCR did not read the same way, are listed in `review`. PaddleOCR lines with almost nothing similar in the Typhoon output (`MISSING_MAX_SIMILARITY`, default 0.4) are listed as `missing_in_vlm` with their `bbox`, because Typhoon sometimes skips whole lines.

`review` is a list of things to check by hand, not a verdict: it also contains words Typhoon read correctly but PaddleOCR misread. If Typhoon is not ready or fails, the page falls back to PaddleOCR text (`source: "paddleocr"`) and `review` says why. On first start the OCR engine pulls the model (about 3.2 GB) in the background; `/health` shows `vlm_ready`. Images are resized for Typhoon so the long side is 1300–1800 px (`VLM_MIN_LONG_SIDE`, `VLM_MAX_LONG_SIDE`). Measure accuracy on your own documents with `eval/` before relying on it.

## Answering the prompt

`prompt` (and optional `response_format`: `auto` | `json` | `text`) is answered by a Thai text LLM, **Typhoon 2.5 Qwen3 4B** (`scb10x/typhoon2.5-qwen3-4b`), from the OCR text only; the result is in `result.answer`. `auto` returns JSON when the prompt mentions "json". The LLM is told to copy values exactly and use `null` for anything not in the document, at temperature 0.

Every JSON value is then checked: `not_in_document` means the value does not appear in the OCR text (possibly invented), `uncertain_ocr` means it contains a word the OCR cross-check flagged. For JSON answers the LLM also reports, per top-level field, the document label it read the value from (`answer.sources`). If the value is not within two lines of that label it is marked `label_mismatch` (typically a value borrowed from a neighbouring field, such as a tax ID returned as an empty P/O number); `label_not_found` / `label_missing` mean the cited label is not in the document or was not given. Table items are not label-checked. A value without a review entry is still only as correct as the OCR text it was copied from.

## Coolify deployment

ใช้ไฟล์แยกสำหรับ production:

```bash
cp .env.coolify.example .env.coolify
# แก้ API_KEY และ POSTGRES_PASSWORD ก่อนใช้จริง
POSTGRES_USER=ocrllm POSTGRES_PASSWORD=test-only POSTGRES_DB=ocrllm \
  docker compose -f docker-compose.coolify.yml config
```

ใน Coolify ให้เลือก `docker-compose.coolify.yml` และ expose เฉพาะ service `api` ที่ port `3000` เท่านั้น ส่วน Redis, PostgreSQL และ OCR engine ใช้ `expose` ภายใน network เดียวกัน

ข้อควรทราบ: GPU (`deploy.resources.reservations.devices`) ต้องใช้ Ubuntu/NVIDIA Container Toolkit และ host ที่ติดตั้ง NVIDIA driver แล้ว การ `docker compose config` ตรวจได้เฉพาะ syntax/interpolation ไม่ได้ยืนยัน GPU inference

## Target server recommendation

- Ubuntu 22.04/24.04 x86_64 + NVIDIA driver compatible with the selected PaddlePaddle CUDA build
- RTX 5060 Ti 16 GB: start with OCR worker concurrency 1; measure before considering 2
- RAM 32 GB: acceptable for one API + Redis + PostgreSQL + one OCR worker, but cap PDF size/pages and avoid loading multiple LLM models in VRAM
- Put authentication, rate limit, upload size/page limits, timeout, and observability at the API boundary

## Phases

1. Hono contract + tests (this scaffold)
2. Redis/BullMQ durable queue and PostgreSQL migrations
3. Python PDF rendering + PaddleOCR GPU validation on the actual NVIDIA host
4. LLM extraction as a separate bounded step after OCR, with JSON schema validation
5. Docker Compose/Coolify deployment and real end-to-end load test

Never treat API tests with fake OCR as proof that GPU OCR or LLM extraction is correct.
