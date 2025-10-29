/**
 * TEAM-356: Async state management hook
 * 
 * Hook for async data loading with loading/error states and automatic cleanup.
 */

import { useState, useEffect, useRef, useCallback, type DependencyList } from 'react'

/**
 * Options for useAsyncState hook
 */
export interface AsyncStateOptions {
  /** Skip initial load (default: false) */
  skip?: boolean
  
  /** Callback on success */
  onSuccess?: (data: any) => void
  
  /** Callback on error */
  onError?: (error: Error) => void
}

/**
 * Result from useAsyncState hook
 */
export interface AsyncStateResult<T> {
  /** Loaded data (null if not loaded yet) */
  data: T | null
  
  /** Loading state */
  loading: boolean
  
  /** Error if load failed */
  error: Error | null
  
  /** Manually trigger refetch */
  refetch: () => void
}

/**
 * Hook for async data loading with loading/error states
 * 
 * Automatically handles:
 * - Loading state management
 * - Error handling
 * - Cleanup on unmount (prevents state updates after unmount)
 * - Refetch functionality
 * 
 * @param asyncFn - Async function to execute
 * @param deps - Dependency array (like useEffect)
 * @param options - Optional configuration
 * @returns Object with data, loading, error, and refetch
 * 
 * @example
 * ```typescript
 * const { data, loading, error, refetch } = useAsyncState(
 *   async () => {
 *     const response = await fetch('/api/data')
 *     return response.json()
 *   },
 *   []
 * )
 * ```
 */
export function useAsyncState<T>(
  asyncFn: () => Promise<T>,
  deps: DependencyList,
  options: AsyncStateOptions = {}
): AsyncStateResult<T> {
  const { skip = false, onSuccess, onError } = options
  
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(!skip)
  const [error, setError] = useState<Error | null>(null)
  const mountedRef = useRef(true)

  const execute = useCallback(async () => {
    if (skip) return

    setLoading(true)
    setError(null)

    try {
      const result = await asyncFn()
      
      if (mountedRef.current) {
        setData(result)
        setLoading(false)
        onSuccess?.(result)
      }
    } catch (err) {
      const error = err as Error
      
      if (mountedRef.current) {
        setError(error)
        setLoading(false)
        onError?.(error)
      }
    }
  }, [asyncFn, skip, onSuccess, onError])

  useEffect(() => {
    mountedRef.current = true
    execute()

    return () => {
      mountedRef.current = false
    }
  }, [...deps, execute])

  const refetch = useCallback(() => {
    execute()
  }, [execute])

  return { data, loading, error, refetch }
}
