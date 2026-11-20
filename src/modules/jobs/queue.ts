import type { Job, JobStore, OcrResult } from './store.js'

export interface OcrClient {
  process(input: { file: File; prompt: string }): Promise<OcrResult>
}

export class InlineQueue {
  private active = 0
  private readonly pending: Array<{ job: Job; file: File; client: OcrClient }> = []

  constructor(private readonly store: JobStore, private readonly concurrency = 1) {}

  enqueue(job: Job, file: File, client: OcrClient) {
    this.pending.push({ job, file, client })
    void this.drain()
  }

  private async drain() {
    while (this.active < this.concurrency && this.pending.length) {
      const item = this.pending.shift()!
      this.active += 1
      this.store.update(item.job.id, { status: 'processing' })
      try {
        const result = await item.client.process({ file: item.file, prompt: item.job.prompt })
        this.store.update(item.job.id, { status: 'completed', result })
      } catch (error) {
        this.store.update(item.job.id, { status: 'failed', error: error instanceof Error ? error.message : 'OCR failed' })
      } finally {
        this.active -= 1
      }
    }
    if (this.pending.length) void this.drain()
  }
}

export class HttpOcrClient implements OcrClient {
  constructor(private readonly baseUrl: string) {}

  async process(input: { file: File; prompt: string }): Promise<OcrResult> {
    const body = new FormData()
    body.append('file', input.file)
    body.append('prompt', input.prompt)
    const response = await fetch(`${this.baseUrl}/ocr`, { method: 'POST', body })
    if (!response.ok) throw new Error(`OCR engine returned HTTP ${response.status}`)
    return await response.json() as OcrResult
  }
}
