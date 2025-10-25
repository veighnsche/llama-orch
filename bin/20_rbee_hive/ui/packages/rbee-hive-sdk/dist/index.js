// HTTP client for rbee-hive API
// Base URL: http://localhost:7835
export async function listModels() {
    const response = await fetch('http://localhost:7835/api/models');
    return await response.json();
}
export async function downloadModel(model) {
    await fetch('http://localhost:7835/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
    });
}
export async function listWorkers() {
    const response = await fetch('http://localhost:7835/api/workers');
    return await response.json();
}
export async function spawnWorker(model, device) {
    await fetch('http://localhost:7835/api/workers/spawn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, device }),
    });
}
