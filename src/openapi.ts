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
        operationId: 'createOcrJob',
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
                  prompt: { type: 'string', minLength: 1, maxLength: 20000 }
                }
              }
            }
          }
        },
        responses: {
          '202': { description: 'Job accepted', content: { 'application/json': { schema: { $ref: '#/components/schemas/JobAcceptedResponse' } } } },
          '413': { description: 'File too large' },
          '422': { description: 'Invalid file or prompt' }
        }
      }
    },
    '/api/v1/jobs/{id}': {
      get: {
        tags: ['OCR Jobs'], operationId: 'getOcrJob', summary: 'Get OCR job status',
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } }],
        responses: {
          '200': { description: 'Job status', content: { 'application/json': { schema: { $ref: '#/components/schemas/JobResponse' } } } },
          '404': { description: 'Job not found' }
        }
      }
    },
    '/api/v1/jobs/{id}/result': {
      get: {
        tags: ['OCR Jobs'], operationId: 'getOcrResult', summary: 'Get completed OCR result',
        parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } }],
        responses: {
          '200': { description: 'OCR result', content: { 'application/json': { schema: { $ref: '#/components/schemas/ResultResponse' } } } },
          '404': { description: 'Job not found' }, '409': { description: 'Job is not completed' }
        }
      }
    }
  },
  components: {
    schemas: {
      HealthResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { type: 'object', properties: { service: { type: 'string' }, status: { type: 'string' }, queue_concurrency: { type: 'integer' } }, required: ['service', 'status', 'queue_concurrency'] } }, required: ['success', 'data'] },
      JobAcceptedResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { type: 'object', properties: { job_id: { type: 'string', format: 'uuid' }, status: { type: 'string', enum: ['queued'] } }, required: ['job_id', 'status'] } }, required: ['success', 'data'] },
      JobResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { $ref: '#/components/schemas/Job' } }, required: ['success', 'data'] },
      ResultResponse: { type: 'object', properties: { success: { type: 'boolean' }, data: { type: 'object', properties: { job_id: { type: 'string', format: 'uuid' }, result: { $ref: '#/components/schemas/OcrResult' } }, required: ['job_id', 'result'] } }, required: ['success', 'data'] },
      Job: { type: 'object', properties: { id: { type: 'string', format: 'uuid' }, status: { type: 'string', enum: ['queued', 'processing', 'completed', 'failed'] }, filename: { type: 'string' }, mime_type: { type: 'string' }, prompt: { type: 'string' }, created_at: { type: 'string', format: 'date-time' }, updated_at: { type: 'string', format: 'date-time' }, result: { $ref: '#/components/schemas/OcrResult' }, error: { type: 'string' } }, required: ['id', 'status', 'filename', 'mime_type', 'prompt', 'created_at', 'updated_at'] },
      OcrResult: { type: 'object', properties: { text: { type: 'string' }, pages: { type: 'array', items: { $ref: '#/components/schemas/OcrPage' } } }, required: ['text', 'pages'] },
      OcrPage: { type: 'object', properties: { page_number: { type: 'integer' }, text: { type: 'string' }, lines: { type: 'array', items: { $ref: '#/components/schemas/OcrLine' } } }, required: ['page_number', 'text', 'lines'] },
      OcrLine: { type: 'object', properties: { text: { type: 'string' }, confidence: { type: 'number', nullable: true }, bbox: { type: 'array', nullable: true, items: { type: 'array', items: { type: 'number' } } } }, required: ['text', 'confidence', 'bbox'] }
    }
  }
} as const
