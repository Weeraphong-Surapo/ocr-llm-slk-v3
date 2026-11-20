import { describe, expect, it } from 'vitest'
import { HttpOcrClient } from '../src/modules/jobs/queue.js'
import { createApp } from '../src/app.js'
import type { OcrClient } from '../src/modules/jobs/queue.js'

const fakeOcr: OcrClient = {
  async process() { return { pages: [{ page_number: 1, text: 'ทดสอบ OCR', lines: [] }], text: 'ทดสอบ OCR' } }
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
})
