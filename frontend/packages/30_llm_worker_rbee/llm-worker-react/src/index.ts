import { useState } from 'react';
import { infer, InferenceRequest, InferenceResponse } from '@rbee/llm-worker-sdk';

export function useInference() {
  const [response, setResponse] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const runInference = async (request: InferenceRequest) => {
    setLoading(true);
    setError(null);
    try {
      const result = await infer(request);
      setResponse(result.response);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  };

  return { response, loading, error, runInference };
}
