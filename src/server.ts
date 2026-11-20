import { serve } from '@hono/node-server'
import { createApp } from './app.js'
import { env } from './config/env.js'
import { HttpOcrClient } from './modules/jobs/queue.js'

const app = createApp(new HttpOcrClient(env.OCR_ENGINE_URL))
serve({ fetch: app.fetch, port: env.PORT }, (info) => {
  console.log(`ocr-llm-api listening on http://localhost:${info.port}`)
})
