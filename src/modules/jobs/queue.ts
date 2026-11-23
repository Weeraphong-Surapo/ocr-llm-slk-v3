import { setTimeout as delay } from 'node:timers/promises'
import type { Job, JobStore, OcrResult, ResponseFormat } from './store.js'

type OcrInput = {
  file: File
  prompt: string
  responseFormat?: ResponseFormat
  jobId?: string
  signal?: AbortSignal
}

export interface OcrClient {
  process(input: OcrInput): Promise<OcrResult>
}

export class InlineQueue {
  private readonly active = new Map<string, AbortController>()
  private readonly pending: Array<{ job: Job; file: File; client: OcrClient }> = []

  constructor(private readonly store: JobStore, private readonly concurrency = 1) {}

  enqueue(job: Job, file: File, client: OcrClient) {
    this.pending.push({ job, file, client })
    void this.drain()
  }

  cancel(id: string): Job | undefined {
    const job = this.store.get(id)
    if (job?.status === 'queued') {
      const index = this.pending.findIndex(item => item.job.id === id)
      if (index >= 0) this.pending.splice(index, 1)
      return this.store.update(id, { status: 'cancelled' })
    }
    if (job?.status === 'processing') {
      const updated = this.store.update(id, { status: 'cancelling' })
      this.active.get(id)?.abort()
      return updated
    }
    return job
  }

  private async drain() {
    while (this.active.size < this.concurrency && this.pending.length) {
      const item = this.pending.shift()!
      const { id, prompt, response_format: responseFormat } = item.job
      const controller = new AbortController()
      this.active.set(id, controller)
      this.store.update(id, { status: 'processing' })
      try {
        const result = await item.client.process({ file: item.file, prompt, responseFormat, jobId: id, signal: controller.signal })
        this.store.update(id, controller.signal.aborted ? { status: 'cancelled' } : { status: 'completed', result })
      } catch (error) {
        if (controller.signal.aborted && error === controller.signal.reason) {
          this.store.update(id, { status: 'cancelled' })
        } else {
          const message = controller.signal.aborted
            ? 'Unable to confirm OCR engine cancellation'
            : error instanceof Error ? error.message : 'OCR failed'
          this.store.update(id, { status: 'failed', error: message })
        }
      } finally {
        this.active.delete(id)
      }
    }
  }
}

export class HttpOcrClient implements OcrClient {
  constructor(private readonly baseUrl: string) {}

  async process(input: OcrInput): Promise<OcrResult> {
    input.signal?.throwIfAborted()
    const body = new FormData()
    body.append('file', input.file, input.file.name || 'upload')
    body.append('prompt', input.prompt)
    body.append('response_format', input.responseFormat ?? 'auto')
    const finished = new AbortController()
    const requestCancellation = () => {
      if (input.jobId) void this.cancelInEngine(input.jobId, finished.signal)
    }
    input.signal?.addEventListener('abort', requestCancellation, { once: true })
    try {
      const headers = input.jobId ? { 'X-OCR-Job-Id': input.jobId } : undefined
      // Keep the request open and retain the queue slot until the engine has stopped using it.
      const response = await fetch(`${this.baseUrl}/ocr`, { method: 'POST', headers, body })
      if (input.signal?.aborted) {
        await response.body?.cancel().catch(() => {})
        input.signal.throwIfAborted()
      }
      if (!response.ok) {
        const detail = (await response.text().catch(() => '')).slice(0, 300)
        throw new Error(`OCR engine returned HTTP ${response.status}${detail ? `: ${detail}` : ''}`)
      }
      return await response.json() as OcrResult
    } finally {
      input.signal?.removeEventListener('abort', requestCancellation)
      finished.abort()
    }
  }

  private async cancelInEngine(jobId: string, finished: AbortSignal): Promise<void> {
    while (!finished.aborted) {
      try {
        const response = await fetch(`${this.baseUrl}/ocr/${encodeURIComponent(jobId)}/cancel`, {
          method: 'POST',
          signal: AbortSignal.any([finished, AbortSignal.timeout(2_000)])
        })
        await response.body?.cancel()
        if (response.status === 202) return
      } catch {
        // A transient connection error does not stop the original request.
      }
      // Cancellation can arrive before the engine registers the original upload.
      await delay(250, undefined, { signal: finished }).catch(() => {})
    }
  }
}
