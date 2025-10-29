/**
 * TEAM-356: Utility functions for SDK loading
 * 
 * Helper functions for retry logic, backoff calculation, and timeout handling.
 */

/**
 * Sleep for specified milliseconds
 * 
 * @param ms - Milliseconds to sleep
 * @returns Promise that resolves after delay
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Add random jitter to prevent thundering herd
 * 
 * @param baseMs - Base delay in milliseconds
 * @param maxJitterMs - Maximum jitter to add
 * @returns Base delay plus random jitter
 */
export function addJitter(baseMs: number, maxJitterMs: number): number {
  return baseMs + Math.random() * maxJitterMs
}

/**
 * Execute promise with timeout
 * 
 * @param promise - Promise to execute
 * @param timeoutMs - Timeout in milliseconds
 * @param operation - Operation name for error message
 * @returns Promise that rejects if timeout exceeded
 */
export function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  operation: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(
        () => reject(new Error(`Timeout after ${timeoutMs}ms: ${operation}`)),
        timeoutMs
      )
    ),
  ])
}

/**
 * Calculate exponential backoff delay with jitter
 * 
 * @param attempt - Current attempt number (1-indexed)
 * @param baseMs - Base delay in milliseconds
 * @param maxJitterMs - Maximum jitter to add
 * @returns Calculated delay with jitter
 */
export function calculateBackoff(
  attempt: number,
  baseMs: number,
  maxJitterMs: number
): number {
  const exponential = 2 ** (attempt - 1) * baseMs
  return addJitter(exponential, maxJitterMs)
}
