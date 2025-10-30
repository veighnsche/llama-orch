export interface InferenceRequest {
    model: string;
    prompt: string;
    max_tokens?: number;
    temperature?: number;
}
export declare function useInference(): {
    response: any;
    loading: boolean;
    error: Error | null;
    runInference: import("@tanstack/react-query").UseMutateFunction<any, Error, InferenceRequest, unknown>;
};
