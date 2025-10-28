// HTTP client for queen-rbee API
// Base URL: http://localhost:7833

export interface Job {
  id: string
  operation: string
  status: 'pending' | 'running' | 'completed' | 'failed'
}

export async function listJobs(): Promise<Job[]> {
  const response = await fetch('http://localhost:7833/api/jobs')
  return await response.json()
}

export async function getJob(id: string): Promise<Job> {
  const response = await fetch(`http://localhost:7833/api/jobs/${id}`)
  return await response.json()
}

export async function submitInference(prompt: string): Promise<{ job_id: string }> {
  const response = await fetch('http://localhost:7833/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  })
  return await response.json()
}
