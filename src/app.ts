import { Hono } from 'hono'
import { timingSafeEqual } from 'node:crypto'
import { swaggerUI } from '@hono/swagger-ui'
import { env } from './config/env.js'
import { failure, success } from './lib/response.js'
import { openApiDocument } from './openapi.js'
import { createJobRoutes } from './modules/jobs/routes.js'
import type { OcrClient } from './modules/jobs/queue.js'

function keysMatch(provided: string, expected: string) {
  const a = Buffer.from(provided)
  const b = Buffer.from(expected)
  return a.length === b.length && timingSafeEqual(a, b)
}
export function createApp(ocrClient: OcrClient, options: { apiKey?: string } = { apiKey: env.API_KEY }) {
  const app = new Hono()
  const apiKey = options.apiKey
  if (apiKey) {
    app.use('/api/v1/*', async (c, next) => {
      const header = c.req.header('authorization') ?? ''
      const provided = header.startsWith('Bearer ') ? header.slice(7) : (c.req.header('x-api-key') ?? '')
      if (!provided || !keysMatch(provided, apiKey)) return c.json(failure('UNAUTHORIZED', 'invalid or missing API key'), 401)
      await next()
    })
  }
  app.get('/health', (c) => c.json(success({ service: 'ocr-llm-api', status: 'ok', queue_concurrency: env.OCR_QUEUE_CONCURRENCY })))
  app.get('/api/openapi.json', (c) => c.json(openApiDocument))
  app.get('/api/docs', (c) => c.redirect('/api/docs/'))
  app.get('/api/docs/', swaggerUI({ url: '/api/openapi.json' }))
  app.route('/api/v1/jobs', createJobRoutes({ ocrClient, concurrency: env.OCR_QUEUE_CONCURRENCY }))
  return app
}
