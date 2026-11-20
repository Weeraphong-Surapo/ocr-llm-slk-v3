export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export type ReviewItem = {
  page_number: number
  text: string | null
  reason: 'low_similarity' | 'number_or_email_not_confirmed' | 'missing_in_vlm' | 'vlm_failed' | 'vlm_unavailable' | 'vlm_output_truncated'
  similarity: number | null
  bbox?: number[][] | null
  detail?: string | null
}

export type OcrPage = {
  page_number: number
  text: string
  source?: 'vlm' | 'paddleocr'
  paddle_text?: string
  lines: Array<{ text: string; confidence: number | null; bbox: number[][] | null }>
  review?: ReviewItem[]
}

export type ResponseFormat = 'auto' | 'json' | 'text'

export type AnswerReviewItem = {
  path: string
  value: unknown
  status: 'not_in_document' | 'uncertain_ocr' | 'label_mismatch' | 'label_not_found' | 'label_missing'
  flagged?: string[]
  label?: string | null
}

export type Answer = {
  format: 'json' | 'text'
  content: unknown
  sources?: Record<string, string | null> | null
  review: AnswerReviewItem[]
  model: string | null
  error: string | null
}

export type OcrResult = {
  pages: OcrPage[]
  text: string
  review?: ReviewItem[]
  answer?: Answer
  engines?: { ocr: string; vlm: string | null; llm?: string | null }
}

export type Job = {
  id: string
  status: JobStatus
  filename: string
  mime_type: string
  prompt: string
  response_format: ResponseFormat
  created_at: string
  updated_at: string
  result?: OcrResult
  error?: string
}

export interface JobStore {
  create(input: Pick<Job, 'filename' | 'mime_type' | 'prompt' | 'response_format'>): Job
  get(id: string): Job | undefined
  update(id: string, patch: Partial<Job>): Job | undefined
}

export class MemoryJobStore implements JobStore {
  private readonly jobs = new Map<string, Job>()

  create(input: Pick<Job, 'filename' | 'mime_type' | 'prompt' | 'response_format'>): Job {
    const now = new Date().toISOString()
    const job: Job = { id: crypto.randomUUID(), status: 'queued', ...input, created_at: now, updated_at: now }
    this.jobs.set(job.id, job)
    return job
  }

  get(id: string) { return this.jobs.get(id) }

  update(id: string, patch: Partial<Job>) {
    const current = this.jobs.get(id)
    if (!current) return undefined
    const updated = { ...current, ...patch, updated_at: new Date().toISOString() }
    this.jobs.set(id, updated)
    return updated
  }
}
