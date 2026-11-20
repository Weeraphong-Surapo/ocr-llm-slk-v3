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
- `POST /api/v1/jobs` multipart: `file`, `prompt`
- `GET /api/v1/jobs/:id`
- `GET /api/v1/jobs/:id/result`

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
