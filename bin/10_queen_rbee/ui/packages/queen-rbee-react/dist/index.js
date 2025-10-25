import { useState, useEffect } from 'react';
import { listJobs, getJob } from '@rbee/queen-rbee-sdk';
export function useJobs() {
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    useEffect(() => {
        const fetchJobs = async () => {
            try {
                const data = await listJobs();
                setJobs(data);
            }
            catch (err) {
                setError(err);
            }
            finally {
                setLoading(false);
            }
        };
        fetchJobs();
        const interval = setInterval(fetchJobs, 3000);
        return () => clearInterval(interval);
    }, []);
    return { jobs, loading, error };
}
export function useJob(id) {
    const [job, setJob] = useState(null);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        const fetchJob = async () => {
            const data = await getJob(id);
            setJob(data);
            setLoading(false);
        };
        fetchJob();
    }, [id]);
    return { job, loading };
}
