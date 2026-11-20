import { Hono } from 'hono'
import { z } from 'zod'
import { maxUploadBytes } from '../../config/env.js'
import { failure, success } from '../../lib/response.js'
import { InlineQueue, type OcrClient } from './queue.js'
import { MemoryJobStore, type JobStore } from './store.js'

const promptSchema = z.string().trim().min(1).max(20_000)
const acceptedTypes = new Set(['application/pdf', 'image/png', 'image/jpeg', 'image/webp'])

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
    if (!acceptedTypes.has(file.type)) return c.json(failure('UNSUPPORTED_FILE', 'Only PDF/PNG/JPEG/WebP are accepted'), 422)
    if (file.size > maxUploadBytes) return c.json(failure('FILE_TOO_LARGE', `Maximum upload is ${maxUploadBytes / 1024 / 1024} MB`), 413)
    const parsedPrompt = promptSchema.safeParse(typeof prompt === 'string' ? prompt : '')
    if (!parsedPrompt.success) return c.json(failure('INVALID_PROMPT', 'prompt is required and must be <= 20,000 chars'), 422)

    const job = store.create({ filename: file.name || 'upload', mime_type: file.type, prompt: parsedPrompt.data })
    queue.enqueue(job, file, deps.ocrClient)
    return c.json(success({ job_id: job.id, status: job.status }), 202)
  })

  app.get('/:id', (c) => {
    const job = store.get(c.req.param('id'))
    if (!job) return c.json(failure('NOT_FOUND', 'Job not found'), 404)
    return c.json(success(job))
  })

  app.get('/:id/result', (c) => {
    const job = store.get(c.req.param('id'))
    if (!job) return c.json(failure('NOT_FOUND', 'Job not found'), 404)
    if (job.status !== 'completed') return c.json(failure('NOT_READY', `Job status is ${job.status}`), 409)
    return c.json(success({ job_id: job.id, result: job.result }))
  })

  return app
}
