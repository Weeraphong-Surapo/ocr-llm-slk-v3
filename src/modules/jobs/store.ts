export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export type OcrPage = {
  page_number: number
  text: string
  lines: Array<{ text: string; confidence: number | null; bbox: number[][] | null }>
}

export type OcrResult = { pages: OcrPage[]; text: string }

export type Job = {
  id: string
  status: JobStatus
  filename: string
  mime_type: string
  prompt: string
  created_at: string
  updated_at: string
  result?: OcrResult
  error?: string
}

export interface JobStore {
  create(input: Pick<Job, 'filename' | 'mime_type' | 'prompt'>): Job
  get(id: string): Job | undefined
  update(id: string, patch: Partial<Job>): Job | undefined
}

export class MemoryJobStore implements JobStore {
  private readonly jobs = new Map<string, Job>()

  create(input: Pick<Job, 'filename' | 'mime_type' | 'prompt'>): Job {
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
