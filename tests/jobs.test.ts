import { describe, expect, it, vi } from 'vitest'
import { Hono } from 'hono'
import { HttpOcrClient } from '../src/modules/jobs/queue.js'
import { createApp } from '../src/app.js'
import type { OcrClient } from '../src/modules/jobs/queue.js'
import { createJobRoutes } from '../src/modules/jobs/routes.js'

const fakeOcr: OcrClient = {
  async process() { return { pages: [{ page_number: 1, text: 'ทดสอบ OCR', lines: [] }], text: 'ทดสอบ OCR' } }
}

async function submitJob(app: Pick<ReturnType<typeof createApp>, 'request'>, filename: string) {
  const body = new FormData()
  body.append('file', new File(['%PDF-test'], filename, { type: 'application/pdf' }))
  body.append('prompt', 'อ่านข้อมูล')
  const response = await app.request('/api/v1/jobs', { method: 'POST', body })
  expect(response.status).toBe(202)
  return `/api/v1/jobs/${(await response.json()).data.job_id}`
}

describe('OCR job API', () => {
  it('serves OpenAPI JSON and Swagger UI', async () => {
    const app = createApp(fakeOcr)
    const openapi = await app.request('/api/openapi.json')
    expect(openapi.status).toBe(200)
    const document = await openapi.json()
    expect(document.openapi).toBe('3.0.3')
    expect(document.paths['/api/v1/jobs'].post.requestBody.content['multipart/form-data']).toBeDefined()

    const docs = await app.request('/api/docs/')
    expect(docs.status).toBe(200)
    expect(docs.headers.get('content-type')).toContain('text/html')
    expect(await docs.text()).toContain('swagger-ui')
  })

  it('accepts PDF + prompt and returns a queued job', async () => {
    const app = createApp(fakeOcr)
    const body = new FormData()
    body.append('file', new File(['%PDF-test'], 'invoice.pdf', { type: 'application/pdf' }))
    body.append('prompt', 'อ่านเลขที่ใบกำกับภาษี')
    const response = await app.request('/api/v1/jobs', { method: 'POST', body })
    expect(response.status).toBe(202)
    expect((await response.json()).data.job_id).toBeTypeOf('string')
  })

  it('rejects unsupported file types', async () => {
    const app = createApp(fakeOcr)
    const body = new FormData()
    body.append('file', new File(['x'], 'note.txt', { type: 'text/plain' }))
    body.append('prompt', 'อ่านข้อมูล')
    const response = await app.request('/api/v1/jobs', { method: 'POST', body })
    expect(response.status).toBe(422)
  })
  it('requires the API key on /api/v1 when configured', async () => {
    const app = createApp(fakeOcr, { apiKey: 'secret' })
    expect((await app.request('/api/v1/jobs/x')).status).toBe(401)
    expect((await app.request('/api/v1/jobs/x', { headers: { authorization: 'Bearer wrong' } })).status).toBe(401)
    expect((await app.request('/api/v1/jobs/x', { headers: { authorization: 'Bearer secret' } })).status).not.toBe(401)
    expect((await app.request('/api/v1/jobs/x', { headers: { 'x-api-key': 'secret' } })).status).not.toBe(401)
    expect((await app.request('/api/v1/jobs/x/cancel', { method: 'POST' })).status).toBe(401)
    expect((await app.request('/api/v1/jobs/x/cancel', { method: 'POST', headers: { 'x-api-key': 'secret' } })).status).toBe(404)
    expect((await app.request('/health')).status).toBe(200)
  })
  it('forwards a filename to the OCR engine even when the upload has none', async () => {
    let sent: FormData | undefined
    const original = globalThis.fetch
    globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
      sent = init?.body as FormData
      return new Response(JSON.stringify({ pages: [], text: '' }), { status: 200 })
    }) as typeof fetch
    try {
      await new HttpOcrClient('http://engine').process({ file: new File(['x'], '', { type: 'image/png' }), prompt: 'p' })
    } finally {
      globalThis.fetch = original
    }
    expect((sent!.get('file') as File).name).toBe('upload')
    expect(sent!.has('response_schema')).toBe(false)
  })

  it('forwards the optional response schema through the job queue and rejects malformed input', async () => {
    const schema = JSON.stringify({ type: 'object', properties: { items: { type: 'array' } } })
    let sent: FormData | undefined
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      sent = init?.body as FormData
      return Response.json({ pages: [], text: '' })
    })
    try {
      const app = createApp(new HttpOcrClient('http://engine'), {})
      const submit = (responseSchema: string, format?: string) => {
        const body = new FormData()
        body.append('file', new File(['%PDF-test'], 'invoice.pdf', { type: 'application/pdf' }))
        body.append('prompt', 'อ่านข้อมูล')
        body.append('response_schema', responseSchema)
        if (format) body.append('response_format', format)
        return app.request('/api/v1/jobs', { method: 'POST', body })
      }
      expect((await submit('{')).status).toBe(422)
      expect((await submit('[]')).status).toBe(422)
      expect((await submit(schema, 'text')).status).toBe(422)
      expect(fetch).not.toHaveBeenCalled()
      expect((await submit(schema)).status).toBe(202)
      expect(sent?.get('response_format')).toBe('json')
      expect(sent?.get('response_schema')).toBe(schema)
    } finally {
      fetch.mockRestore()
    }
  })
  it('detects the real file type instead of trusting the declared one', async () => {
    let seenType = ''
    const capturing: OcrClient = { async process({ file }) { seenType = file.type; return { pages: [], text: '' } } }
    const app = createApp(capturing, { apiKey: undefined })
    const body = new FormData()
    body.append('file', new File(['%PDF-1.7 fake'], '01 (2).pdf', { type: 'image/png' }))
    body.append('prompt', 'อ่าน')
    const response = await app.request('/api/v1/jobs', { method: 'POST', body })
    expect(response.status).toBe(202)
    const id = (await response.json()).data.job_id
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(seenType).toBe('application/pdf')
    expect((await (await app.request(`/api/v1/jobs/${id}`)).json()).data.mime_type).toBe('application/pdf')
  })
  it('rejects files whose bytes are not a supported type even if declared as PDF', async () => {
    const app = createApp(fakeOcr, { apiKey: undefined })
    const body = new FormData()
    body.append('file', new File(['hello'], 'x.pdf', { type: 'application/pdf' }))
    body.append('prompt', 'อ่าน')
    expect((await app.request('/api/v1/jobs', { method: 'POST', body })).status).toBe(422)
  })
  it('forwards response_format and rejects invalid values', async () => {
    let seen: string | undefined
    const capturing: OcrClient = { async process(input) { seen = input.responseFormat; return { pages: [], text: '' } } }
    const app = createApp(capturing, { apiKey: undefined })
    const send = async (format?: string) => {
      const body = new FormData()
      body.append('file', new File(['%PDF-1.7 x'], 'a.pdf', { type: 'application/pdf' }))
      body.append('prompt', 'ช่วยเป็น json')
      if (format !== undefined) body.append('response_format', format)
      return app.request('/api/v1/jobs', { method: 'POST', body })
    }
    expect((await send('xml')).status).toBe(422)
    const ok = await send('json')
    expect(ok.status).toBe(202)
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(seen).toBe('json')
    const id = (await ok.json()).data.job_id
    expect((await (await app.request(`/api/v1/jobs/${id}`)).json()).data.response_format).toBe('json')
    expect((await send()).status).toBe(202)
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(seen).toBe('auto')
  })
  it('documents the X-API-Key header in OpenAPI', async () => {
    const doc = await (await createApp(fakeOcr).request('/api/openapi.json')).json()
    expect(doc.components.securitySchemes.ApiKeyHeader).toEqual(expect.objectContaining({ type: 'apiKey', in: 'header', name: 'X-API-Key' }))
    expect(doc.paths['/api/v1/jobs'].post.security).toContainEqual({ ApiKeyHeader: [] })
    expect(doc.paths['/api/v1/jobs/{id}'].get.responses['401']).toBeDefined()
    expect(doc.paths['/api/v1/jobs/{id}/cancel'].post.security).toContainEqual({ ApiKeyHeader: [] })
    expect(doc.paths['/health'].get.security).toBeUndefined()
  })

  it('accepts and forwards the OCR mode', async () => {
    let seen: string | undefined
    const capturing: OcrClient = { async process(input) { seen = input.mode; return { pages: [], text: '' } } }
    const app = createApp(capturing, { apiKey: 'secret' })
    const body = new FormData()
    body.append('file', new File(['%PDF-1.7 x'], 'a.pdf', { type: 'application/pdf' }))
    body.append('prompt', 'อ่านข้อมูล')
    body.append('mode', 'high_accuracy')
    const response = await app.request('/api/v1/jobs', { method: 'POST', body, headers: { 'x-api-key': 'secret' } })
    expect(response.status).toBe(202)
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(seen).toBe('high_accuracy')
    const id = (await response.json()).data.job_id
    expect((await (await app.request(`/api/v1/jobs/${id}`, { headers: { 'x-api-key': 'secret' } })).json()).data.mode).toBe('high_accuracy')
  })

  it('defaults jobs to standard when mode is omitted', async () => {
    let seen: string | undefined
    const app = new Hono().route('/api/v1/jobs', createJobRoutes({
      ocrClient: { async process(input) { seen = input.mode; return { pages: [], text: '' } } }
    }))
    const body = new FormData()
    body.append('file', new File(['%PDF-1.7 x'], 'a.pdf', { type: 'application/pdf' }))
    body.append('prompt', 'อ่านข้อมูล')
    expect((await app.request('/api/v1/jobs', { method: 'POST', body })).status).toBe(202)
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(seen).toBe('standard')
  })

  it('allows a file-only fast OCR job without a prompt', async () => {
    let seenPrompt: string | undefined
    const app = new Hono().route('/api/v1/jobs', createJobRoutes({
      ocrClient: { async process(input) { seenPrompt = input.prompt; return { pages: [], text: '' } } }
    }))
    const body = new FormData()
    body.append('file', new File(['%PDF-1.7 x'], 'a.pdf', { type: 'application/pdf' }))
    body.append('mode', 'fast')
    expect((await app.request('/api/v1/jobs', { method: 'POST', body })).status).toBe(202)
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(seenPrompt).toBe('')
  })

  it('removes a queued job and holds the running job slot until cancellation finishes', async () => {
    const processed: string[] = []
    let activeSignal: AbortSignal | undefined
    let release = () => {}
    const blocked = new Promise<void>(resolve => { release = resolve })
    const app = new Hono().route('/api/v1/jobs', createJobRoutes({
      concurrency: 1,
      ocrClient: {
        async process({ file, signal }) {
          processed.push(file.name)
          if (file.name === 'active.pdf') {
            activeSignal = signal
            await blocked
          }
          return { pages: [], text: file.name }
        }
      }
    }))
    const cancel = (path: string) => app.request(`${path}/cancel`, { method: 'POST' })
    const status = async (path: string) => (await (await app.request(path)).json()).data
    try {
      const active = await submitJob(app, 'active.pdf')
      const queued = await submitJob(app, 'queued.pdf')
      const next = await submitJob(app, 'next.pdf')
      expect((await cancel(queued)).status).toBe(200)
      expect((await status(queued)).status).toBe('cancelled')
      expect((await cancel(queued)).status).toBe(200)
      expect((await cancel(active)).status).toBe(202)
      expect(activeSignal?.aborted).toBe(true)
      expect((await cancel(active)).status).toBe(202)
      expect((await status(active)).status).toBe('cancelling')
      expect(processed).toEqual(['active.pdf'])

      release()
      await vi.waitFor(async () => expect((await status(next)).status).toBe('completed'))
      expect(await status(active)).toMatchObject({ status: 'cancelled' })
      expect((await status(active)).result).toBeUndefined()
      expect((await app.request(`${active}/result`)).status).toBe(409)
      expect((await cancel(active)).status).toBe(200)
      expect((await cancel(next)).status).toBe(409)
      expect(processed).toEqual(['active.pdf', 'next.pdf'])
    } finally {
      release()
    }
  })

  it('retries cancellation if the engine has not registered the upload yet', async () => {
    const jobId = '00000000-0000-4000-8000-000000000001'
    const controller = new AbortController()
    let finishInference: (response: Response) => void = () => {}
    const inference = new Promise<Response>(resolve => { finishInference = resolve })
    let attempts = 0
    const fetch = vi.fn(async (url: string, init: RequestInit) => {
      if (url === 'http://engine/ocr') {
        expect(init.headers).toEqual({ 'X-OCR-Job-Id': jobId })
        expect(init.signal).toBeUndefined()
        return inference
      }
      expect(url).toBe(`http://engine/ocr/${jobId}/cancel`)
      expect(init.method).toBe('POST')
      return new Response(null, { status: ++attempts === 1 ? 404 : 202 })
    })
    vi.stubGlobal('fetch', fetch)
    try {
      const result = new HttpOcrClient('http://engine').process({
        file: new File(['%PDF-test'], 'invoice.pdf'), prompt: 'อ่าน', jobId, signal: controller.signal
      })
      const cancelled = expect(result).rejects.toMatchObject({ name: 'AbortError' })
      controller.abort()
      await vi.waitFor(() => expect(attempts).toBe(2))
      finishInference(Response.json({ pages: [], text: '' }))
      await cancelled
      expect(fetch).toHaveBeenCalledTimes(3)
    } finally {
      finishInference(Response.json({ pages: [], text: '' }))
      vi.unstubAllGlobals()
    }
  })

  it('reports cancellation as unconfirmed if engine communication fails', async () => {
    let disconnect = () => {}
    const app = createApp({
      process: () => new Promise((_, reject) => {
        disconnect = () => reject(new TypeError('fetch failed'))
      })
    }, { apiKey: undefined })
    const job = await submitJob(app, 'invoice.pdf')
    try {
      expect((await app.request(`${job}/cancel`, { method: 'POST' })).status).toBe(202)
      disconnect()
      await vi.waitFor(async () => {
        const response = await app.request(job)
        expect((await response.json()).data).toMatchObject({
          status: 'failed',
          error: 'Unable to confirm OCR engine cancellation'
        })
      })
    } finally {
      disconnect()
    }
  })
})
