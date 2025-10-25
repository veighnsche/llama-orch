import { useState } from 'react';
import { infer } from '@rbee/llm-worker-sdk';
export function useInference() {
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const runInference = async (request) => {
        setLoading(true);
        setError(null);
        try {
            const result = await infer(request);
            setResponse(result.response);
        }
        catch (err) {
            setError(err);
        }
        finally {
            setLoading(false);
        }
    };
    return { response, loading, error, runInference };
}
