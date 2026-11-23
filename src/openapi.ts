export const openApiDocument = {
  openapi: '3.0.3',
  info: {
    title: 'OCR LLM Service API',
    version: '0.1.0',
    description: 'Asynchronous PDF/image OCR API. Submit a file and prompt, then poll the job result.'
  },
  servers: [{ url: '/' }],
  tags: [{ name: 'System' }, { name: 'OCR Jobs' }],
  paths: {
    '/health': {
      get: {
        tags: ['System'],
        operationId: 'health',
        responses: {
          '200': {
            description: 'Service health',
            content: { 'application/json': { schema: { $ref: '#/components/schemas/HealthResponse' } } }
          }
        }
      }
    },
    '/api/v1/jobs': {
      post: {
        tags: ['OCR Jobs'],
        operationId: 'createOcrJob', security: [{ ApiKeyHeader: [] }, { BearerAuth: [] }],
        summary: 'Submit a PDF/image OCR job',
        requestBody: {
          required: true,
          content: {
            'multipart/form-data': {
              schema: {
                type: 'object',
                required: ['file', 'prompt'],
                properties: {
                  file: { type: 'string', format: 'binary', description: 'PDF, PNG, JPEG, or WebP' },
                  prompt: { type: 'string', minLength: 1, maxLength: 20000, description: 'Instruction answered from the OCR text, e.g. "ช่วยเป็น json". The answer is in result.answer.' },
                  response_format: { type: 'string', enum: ['auto', 'json', 'text'], default: 'auto', description: 'auto = JSON when the prompt mentions json' }
                }
              }
            }
          }
        },
        responses: {
          '202': { description: 'Job accepted', content: { 'application/json': { schema: { $ref: '#/components/schemas/JobAcceptedResponse' } } } },
          '401': { description: 'Missing or invalid API key' },
          '413': { description: 'File too large' },
          '422': { description: 'Invalid file or prompt' }
        }
      }
    },
    '/api/v1/jobs/{id}': {
      get: {
        tags: ['OCR Jobs'], operationId: 'getOcrJob', security: [{ ApiKeyHeader: [] }, { BearerAuth: [] }], summary: 'Get OCR job status',
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } }],
        responses: {
          '200': { description: 'Job status', content: { 'application/json': { schema: { $ref: '#/components/schemas/JobResponse' } } } },
          '401': { description: 'Missing or invalid API key' },
          '404': { description: 'Job not found' }
        }
      }
    },
    '/api/v1/jobs/{id}/cancel': {
      post: {
        tags: ['OCR Jobs'], operationId: 'cancelOcrJob', security: [{ ApiKeyHeader: [] }, { BearerAuth: [] }],
        summary: 'Cancel an OCR job',
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } }],
        responses: {
          '200': { description: 'Job cancelled', content: { 'application/json': { schema: { $ref: '#/components/schemas/CancelJobResponse' } } } },
          '202': { description: 'Cancellation requested. Poll job status until cancelled.', content: { 'application/json': { schema: { $ref: '#/components/schemas/CancelJobResponse' } } } },
          '401': { description: 'Missing or invalid API key' },
          '404': { description: 'Job not found' },
          '409': { description: 'Job already completed or failed' }
        }
      }
    },
    '/api/v1/jobs/{id}/result': {
      get: {
        tags: ['OCR Jobs'], operationId: 'getOcrResult', security: [{ ApiKeyHeader: [] }, { BearerAuth: [] }], summary: 'Get completed OCR result',
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } }],
        responses: {
          '200': { description: 'OCR result', content: { 'application/json': { schema: { $ref: '#/components/schemas/ResultResponse' } } } },
          '401': { description: 'Missing or invalid API key' }, '404': { description: 'Job not found' }, '409': { description: 'Job is not completed' }
        }
      }
    }
  },
  components: {
    securitySchemes: {
      ApiKeyHeader: { type: 'apiKey', in: 'header', name: 'X-API-Key', description: 'API key (same value as API_KEY on the server)' },
      BearerAuth: { type: 'http', scheme: 'bearer', description: 'Alternative: Authorization: Bearer <API key>' }
    },
    schemas: {
      HealthResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { type: 'object', properties: { service: { type: 'string' }, status: { type: 'string' }, queue_concurrency: { type: 'integer' } }, required: ['service', 'status', 'queue_concurrency'] } }, required: ['success', 'data'] },
      JobAcceptedResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { type: 'object', properties: { job_id: { type: 'string', format: 'uuid' }, status: { type: 'string', enum: ['queued'] } }, required: ['job_id', 'status'] } }, required: ['success', 'data'] },
      JobResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { $ref: '#/components/schemas/Job' } }, required: ['success', 'data'] },
      ResultResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { type: 'object', properties: { job_id: { type: 'string', format: 'uuid' }, result: { $ref: '#/components/schemas/OcrResult' } }, required: ['job_id', 'result'] } }, required: ['success', 'data'] },
      CancelJobResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { type: 'object', properties: { job_id: { type: 'string', format: 'uuid' }, status: { type: 'string', enum: ['cancelling', 'cancelled'] } }, required: ['job_id', 'status'] } }, required: ['success', 'data'] },
      Job: { type: 'object', properties: { id: { type: 'string', format: 'uuid' }, status: { type: 'string', enum: ['queued', 'processing', 'cancelling', 'cancelled', 'completed', 'failed'] }, filename: { type: 'string' }, mime_type: { type: 'string' }, prompt: { type: 'string' }, response_format: { type: 'string', enum: ['auto', 'json', 'text'] }, created_at: { type: 'string', format: 'date-time' }, updated_at: { type: 'string', format: 'date-time' }, result: { $ref: '#/components/schemas/OcrResult' }, error: { type: 'string' } }, required: ['id', 'status', 'filename', 'mime_type', 'prompt', 'created_at', 'updated_at'] },
      Answer: { type: 'object', description: 'Answer to the prompt, produced by a text LLM from the OCR text only.', properties: { format: { type: 'string', enum: ['json', 'text'] }, content: { description: 'Parsed JSON (format=json) or text' }, sources: { type: 'object', nullable: true, description: 'Document label the LLM used for each top-level field', additionalProperties: { type: 'string', nullable: true } }, review: { type: 'array', description: 'JSON values to check: not_in_document = possibly invented; uncertain_ocr = contains words the OCR cross-check flagged; label_mismatch = value is not next to the cited label (likely taken from another field); label_not_found / label_missing = cited label is not in the document / not given', items: { type: 'object', properties: { path: { type: 'string', example: '$.po_number' }, value: {}, status: { type: 'string', enum: ['not_in_document', 'uncertain_ocr', 'label_mismatch', 'label_not_found', 'label_missing'] }, flagged: { type: 'array', items: { type: 'string' } }, label: { type: 'string', nullable: true } }, required: ['path', 'value', 'status'] } }, model: { type: 'string', nullable: true }, error: { type: 'string', nullable: true } }, required: ['format', 'content', 'review', 'model', 'error'] },
      OcrResult: { type: 'object', properties: { text: { type: 'string', description: 'All pages joined. Typhoon OCR Markdown when available, otherwise PaddleOCR text.' }, pages: { type: 'array', items: { $ref: '#/components/schemas/OcrPage' } }, review: { type: 'array', description: 'Words to check by hand: not confirmed by PaddleOCR, or the VLM step failed.', items: { $ref: '#/components/schemas/ReviewItem' } }, answer: { $ref: '#/components/schemas/Answer' }, engines: { type: 'object', properties: { ocr: { type: 'string' }, vlm: { type: 'string', nullable: true }, llm: { type: 'string', nullable: true } } } }, required: ['text', 'pages'] },
      OcrPage: { type: 'object', properties: { page_number: { type: 'integer' }, text: { type: 'string' }, source: { type: 'string', enum: ['vlm', 'paddleocr'] }, paddle_text: { type: 'string' }, lines: { type: 'array', description: 'PaddleOCR lines with positions', items: { $ref: '#/components/schemas/OcrLine' } }, review: { type: 'array', items: { $ref: '#/components/schemas/ReviewItem' } } }, required: ['page_number', 'text', 'lines'] },
      ReviewItem: { type: 'object', properties: { page_number: { type: 'integer' }, text: { type: 'string', nullable: true }, reason: { type: 'string', enum: ['low_similarity', 'number_or_email_not_confirmed', 'missing_in_vlm', 'vlm_failed', 'vlm_unavailable', 'vlm_output_truncated'] }, similarity: { type: 'number', nullable: true }, bbox: { type: 'array', nullable: true, description: 'PaddleOCR position, for missing_in_vlm', items: { type: 'array', items: { type: 'number' } } }, detail: { type: 'string', nullable: true } }, required: ['page_number', 'text', 'reason', 'similarity'] },
      OcrLine: { type: 'object', properties: { text: { type: 'string' }, confidence: { type: 'number', nullable: true }, bbox: { type: 'array', nullable: true, items: { type: 'array', items: { type: 'number' } } } }, required: ['text', 'confidence', 'bbox'] }
    }
  }
} as const
