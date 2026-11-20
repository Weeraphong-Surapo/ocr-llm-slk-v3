import { z } from 'zod'

const schema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().int().positive().default(8787),
  MAX_UPLOAD_MB: z.coerce.number().positive().default(20),
  OCR_ENGINE_URL: z.string().url().default('http://localhost:8001'),
  OCR_QUEUE_CONCURRENCY: z.coerce.number().int().positive().default(1),
  JOB_RETENTION_HOURS: z.coerce.number().positive().default(24),
  API_KEY: z.string().optional()
})

export const env = schema.parse(process.env)
export const maxUploadBytes = env.MAX_UPLOAD_MB * 1024 * 1024
