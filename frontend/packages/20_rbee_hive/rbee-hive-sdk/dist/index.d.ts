export interface Model {
    id: string;
    name: string;
    size_bytes: number;
}
export interface Worker {
    pid: number;
    model: string;
    device: string;
}
export declare function listModels(): Promise<Model[]>;
export declare function downloadModel(model: string): Promise<void>;
export declare function listWorkers(): Promise<Worker[]>;
export declare function spawnWorker(model: string, device: string): Promise<void>;
