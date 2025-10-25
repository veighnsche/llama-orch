import { InferenceRequest } from '@rbee/llm-worker-sdk';
export declare function useInference(): {
    response: string;
    loading: boolean;
    error: Error | null;
    runInference: (request: InferenceRequest) => Promise<void>;
};
