// HTTP client for rbee-hive API
// Base URL: http://localhost:7835

export interface Model {
  id: string
  name: string
  size_bytes: number
}

export interface Worker {
  pid: number
  model: string
  device: string
}

export async function listModels(): Promise<Model[]> {
  const response = await fetch('http://localhost:7835/api/models')
  return await response.json()
}

export async function downloadModel(model: string): Promise<void> {
  await fetch('http://localhost:7835/api/models/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  })
}

export async function listWorkers(): Promise<Worker[]> {
  const response = await fetch('http://localhost:7835/api/workers')
  return await response.json()
}

export async function spawnWorker(model: string, device: string): Promise<void> {
  await fetch('http://localhost:7835/api/workers/spawn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, device }),
  })
}
