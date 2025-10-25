import { Job } from '@rbee/queen-rbee-sdk';
export declare function useJobs(): {
    jobs: Job[];
    loading: boolean;
    error: Error | null;
};
export declare function useJob(id: string): {
    job: Job | null;
    loading: boolean;
};
