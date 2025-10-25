export interface Job {
    id: string;
    operation: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
}
export declare function listJobs(): Promise<Job[]>;
export declare function getJob(id: string): Promise<Job>;
export declare function submitInference(prompt: string): Promise<{
    job_id: string;
}>;
