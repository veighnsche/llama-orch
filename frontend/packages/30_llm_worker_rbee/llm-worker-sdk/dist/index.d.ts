export interface InferenceRequest {
    prompt: string;
    temperature?: number;
    max_tokens?: number;
}
export interface InferenceResponse {
    response: string;
}
export declare function infer(request: InferenceRequest): Promise<InferenceResponse>;
export declare function getHealth(): Promise<{
    status: string;
}>;
