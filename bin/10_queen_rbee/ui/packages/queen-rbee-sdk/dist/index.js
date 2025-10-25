// HTTP client for queen-rbee API
// Base URL: http://localhost:7833
export async function listJobs() {
    const response = await fetch('http://localhost:7833/api/jobs');
    return await response.json();
}
export async function getJob(id) {
    const response = await fetch(`http://localhost:7833/api/jobs/${id}`);
    return await response.json();
}
export async function submitInference(prompt) {
    const response = await fetch('http://localhost:7833/api/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
    });
    return await response.json();
}
