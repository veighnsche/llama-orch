// TEAM-353: Migrated to use TanStack Query + WASM SDK
import { useMutation } from '@tanstack/react-query';
import { init, WorkerClient } from '@rbee/llm-worker-sdk';
// TEAM-353: Initialize WASM module once
let wasmInitialized = false;
async function ensureWasmInit() {
    if (!wasmInitialized) {
        init(); // TEAM-353: init() is synchronous in WASM
        wasmInitialized = true;
    }
}
// TEAM-353: Create client instance
// Worker UI is served BY the worker, use window.location.hostname
const workerAddress = window.location.hostname;
const workerPort = '7840'; // TODO: Get from config
const workerId = workerAddress; // Use hostname as worker ID
const client = new WorkerClient(`http://${workerAddress}:${workerPort}`, workerId);
export function useInference() {
    const mutation = useMutation({
        mutationFn: async (request) => {
            await ensureWasmInit();
            // Build operation object
            const operation = {
                operation: 'infer',
                model: request.model,
                prompt: request.prompt,
                max_tokens: request.max_tokens,
                temperature: request.temperature,
            };
            const lines = [];
            await client.submitAndStream(operation, (line) => {
                if (line !== '[DONE]') {
                    lines.push(line);
                }
            });
            // Parse JSON response from last line
            const lastLine = lines[lines.length - 1];
            return lastLine ? JSON.parse(lastLine) : { response: '' };
        },
    });
    return {
        response: mutation.data?.response || '',
        loading: mutation.isPending,
        error: mutation.error,
        runInference: mutation.mutate,
    };
}
