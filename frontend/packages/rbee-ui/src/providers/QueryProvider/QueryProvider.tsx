// TEAM-377: Shared React Query provider for all rbee applications
// Migrated from rbee-keeper to shared package for reuse across Queen, Hive, and Keeper

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

export interface QueryProviderProps {
  children: ReactNode;
  /**
   * Custom QueryClient instance. If not provided, a default one will be created.
   */
  client?: QueryClient;
  /**
   * Number of retry attempts for failed queries (default: 3)
   */
  retry?: number;
  /**
   * Whether to refetch queries on window focus (default: false)
   */
  refetchOnWindowFocus?: boolean;
  /**
   * Whether to refetch queries on mount if data exists (default: false)
   */
  refetchOnMount?: boolean;
  /**
   * Whether to refetch queries on reconnect (default: false)
   */
  refetchOnReconnect?: boolean;
  /**
   * Custom retry delay function (default: exponential backoff)
   */
  retryDelay?: (attemptIndex: number) => number;
}

/**
 * Shared React Query provider for all rbee applications
 * 
 * Provides consistent query client configuration across Queen, Hive, and Keeper.
 * 
 * @example
 * ```tsx
 * // Basic usage (uses defaults)
 * <QueryProvider>
 *   <App />
 * </QueryProvider>
 * 
 * // Custom retry behavior
 * <QueryProvider retry={1} refetchOnWindowFocus={true}>
 *   <App />
 * </QueryProvider>
 * 
 * // Custom client
 * const customClient = new QueryClient({ ... })
 * <QueryProvider client={customClient}>
 *   <App />
 * </QueryProvider>
 * ```
 */
export function QueryProvider({
  children,
  client,
  retry = 3,
  refetchOnWindowFocus = false,
  refetchOnMount = false,
  refetchOnReconnect = false,
  retryDelay = (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
}: QueryProviderProps) {
  // Use provided client or create a default one
  const queryClient = client || new QueryClient({
    defaultOptions: {
      queries: {
        retry,
        retryDelay,
        refetchOnWindowFocus,
        refetchOnMount,
        refetchOnReconnect,
      },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}
