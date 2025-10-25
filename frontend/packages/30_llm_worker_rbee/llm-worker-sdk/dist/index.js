// HTTP client for LLM worker API
// Base URL: http://localhost:8080
export async function infer(request) {
    const response = await fetch('http://localhost:8080/api/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
    });
    return await response.json();
}
export async function getHealth() {
    const response = await fetch('http://localhost:8080/health');
    return await response.json();
}
