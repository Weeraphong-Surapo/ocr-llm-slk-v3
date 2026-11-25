import { Hono } from 'hono'
import { z } from 'zod'
import { maxUploadBytes } from '../../config/env.js'
import { sniffFileType } from '../../lib/file-type.js'
import { failure, success } from '../../lib/response.js'
import { InlineQueue, type OcrClient } from './queue.js'
import { MemoryJobStore, type JobStore, type OcrMode } from './store.js'

const promptSchema = z.string().trim().min(1).max(20_000)
const responseFormatSchema = z.enum(['auto', 'json', 'text']).default('auto')
const modeSchema = z.enum(['fast', 'standard', 'high_accuracy']).default('fast')
const responseSchemaShape = z.object({ type: z.literal('object'), properties: z.record(z.unknown()) })

export type JobRouteDeps = { store?: JobStore; ocrClient: OcrClient; concurrency?: number }

export function createJobRoutes(deps: JobRouteDeps) {
  const store = deps.store ?? new MemoryJobStore()
  const queue = new InlineQueue(store, deps.concurrency ?? 1)
  const app = new Hono()

  app.post('/', async (c) => {
    const body = await c.req.parseBody()
    const file = body.file
    const prompt = body.prompt
    if (!(file instanceof File)) return c.json(failure('INVALID_FILE', 'file is required'), 422)
    const detectedType = sniffFileType(new Uint8Array(await file.slice(0, 16).arrayBuffer()))
    if (!detectedType) return c.json(failure('UNSUPPORTED_FILE', 'Only PDF/PNG/JPEG/WebP are accepted'), 422)
    if (file.size > maxUploadBytes) return c.json(failure('FILE_TOO_LARGE', `Maximum upload is ${maxUploadBytes / 1024 / 1024} MB`), 413)
    const parsedPrompt = promptSchema.safeParse(typeof prompt === 'string' ? prompt : '')
    if (!parsedPrompt.success) return c.json(failure('INVALID_PROMPT', 'prompt is required and must be <= 20,000 chars'), 422)

    // Use the type found in the file's bytes, e.g. a PDF uploaded as image/png is processed as a PDF.
    const name = file.name || 'upload'
    const typedFile = detectedType === file.type ? file : new File([file], name, { type: detectedType })
    const parsedFormat = responseFormatSchema.safeParse(typeof body.response_format === 'string' && body.response_format ? body.response_format : undefined)
    if (!parsedFormat.success) return c.json(failure('INVALID_RESPONSE_FORMAT', 'response_format must be auto, json or text'), 422)
    const parsedMode = modeSchema.safeParse(typeof body.mode === 'string' && body.mode ? body.mode : undefined)
    if (!parsedMode.success) return c.json(failure('INVALID_MODE', 'mode must be fast, standard or high_accuracy'), 422)
    let responseSchema: string | undefined
    if (body.response_schema !== undefined) {
      const raw = body.response_schema
      const message = 'response_schema must be an inline JSON Schema object for a JSON response (max 20000 characters)'
      if (parsedFormat.data === 'text' || typeof raw !== 'string' || raw.length > 20_000) {
        return c.json(failure('INVALID_RESPONSE_SCHEMA', message), 422)
      }
      try {
        if (!responseSchemaShape.safeParse(JSON.parse(raw)).success) {
          return c.json(failure('INVALID_RESPONSE_SCHEMA', message), 422)
        }
      } catch {
        return c.json(failure('INVALID_RESPONSE_SCHEMA', message), 422)
      }
      responseSchema = raw
    }
    const job = store.create({
      filename: name, mime_type: detectedType, prompt: parsedPrompt.data,
      response_format: responseSchema ? 'json' : parsedFormat.data, mode: parsedMode.data as OcrMode,
    })
    queue.enqueue(job, typedFile, deps.ocrClient, responseSchema)
    return c.json(success({ job_id: job.id, status: job.status }), 202)
  })

  app.get('/:id', (c) => {
    const job = store.get(c.req.param('id'))
    if (!job) return c.json(failure('NOT_FOUND', 'Job not found'), 404)
    return c.json(success(job))
  })

  app.post('/:id/cancel', (c) => {
    const job = queue.cancel(c.req.param('id'))
    if (!job) return c.json(failure('NOT_FOUND', 'Job not found'), 404)
    if (job.status === 'completed' || job.status === 'failed') {
      return c.json(failure('JOB_FINISHED', `Job status is ${job.status}`), 409)
    }
    return c.json(success({ job_id: job.id, status: job.status }), job.status === 'cancelling' ? 202 : 200)
  })

  app.get('/:id/result', (c) => {
    const job = store.get(c.req.param('id'))
    if (!job) return c.json(failure('NOT_FOUND', 'Job not found'), 404)
    if (job.status !== 'completed') return c.json(failure('NOT_READY', `Job status is ${job.status}`), 409)
    return c.json(success({ job_id: job.id, result: job.result }))
  })

  return app
}
