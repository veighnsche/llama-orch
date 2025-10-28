// HTTP client for LLM worker API
// Base URL: http://localhost:8080

export interface InferenceRequest {
  prompt: string
  temperature?: number
  max_tokens?: number
}

export interface InferenceResponse {
  response: string
}

export async function infer(request: InferenceRequest): Promise<InferenceResponse> {
  const response = await fetch('http://localhost:8080/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return await response.json()
}

export async function getHealth(): Promise<{ status: string }> {
  const response = await fetch('http://localhost:8080/health')
  return await response.json()
}
