// TEAM-294: React hook for executing Tauri commands with loading/error state
import { useState, useCallback } from 'react';
import type { CommandResponse } from '../api';

interface UseCommandState {
  loading: boolean;
  error: string | null;
  data: string | null;
}

interface UseCommandReturn<T extends unknown[]> {
  execute: (...args: T) => Promise<CommandResponse | null>;
  loading: boolean;
  error: string | null;
  data: string | null;
  reset: () => void;
}

/**
 * Generic hook for executing Tauri commands with state management
 * 
 * @example
 * const { execute, loading, error, data } = useCommand(queenStart);
 * 
 * const handleStart = async () => {
 *   const result = await execute();
 *   if (result?.success) {
 *     console.log('Queen started!');
 *   }
 * };
 */
export function useCommand<T extends unknown[]>(
  commandFn: (...args: T) => Promise<CommandResponse>
): UseCommandReturn<T> {
  const [state, setState] = useState<UseCommandState>({
    loading: false,
    error: null,
    data: null,
  });

  const execute = useCallback(
    async (...args: T): Promise<CommandResponse | null> => {
      setState({ loading: true, error: null, data: null });

      try {
        const response = await commandFn(...args);

        if (response.success) {
          setState({
            loading: false,
            error: null,
            data: response.data || null,
          });
        } else {
          setState({
            loading: false,
            error: response.message,
            data: null,
          });
        }

        return response;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
        setState({
          loading: false,
          error: errorMessage,
          data: null,
        });
        return null;
      }
    },
    [commandFn]
  );

  const reset = useCallback(() => {
    setState({ loading: false, error: null, data: null });
  }, []);

  return {
    execute,
    loading: state.loading,
    error: state.error,
    data: state.data,
    reset,
  };
}
