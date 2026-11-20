import { describe, expect, it } from 'vitest'
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
})
