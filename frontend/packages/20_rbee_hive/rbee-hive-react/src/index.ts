import { useState, useEffect } from 'react';
import { listModels, listWorkers, Model, Worker } from '@rbee/rbee-hive-sdk';

export function useModels() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      const data = await listModels();
      setModels(data);
      setLoading(false);
    };
    fetchModels();
  }, []);

  return { models, loading };
}

export function useWorkers() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchWorkers = async () => {
      const data = await listWorkers();
      setWorkers(data);
      setLoading(false);
    };

    fetchWorkers();
    const interval = setInterval(fetchWorkers, 2000);
    return () => clearInterval(interval);
  }, []);

  return { workers, loading };
}
