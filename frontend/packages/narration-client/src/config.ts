/**
 * TEAM-351: Service configuration for narration client
 * TEAM-351: Bug fixes - Type safety, validation, deduplication with shared-config
 * TEAM-351: CORRECTION - Import ports from @rbee/shared-config (single source of truth)
 */

import type { ServiceName } from './types'
import { PORTS } from '@rbee/shared-config'

export interface ServiceConfig {
  name: string           // Full service name (e.g., 'queen-rbee')
  devPort: number        // Vite dev server port
  prodPort: number       // Production embedded port
  keeperDevPort: number  // Keeper dev server port
  keeperProdOrigin: string // Keeper prod origin ('*' for Tauri)
}

/**
 * TEAM-351: Type-safe service configurations
 * TEAM-351: CORRECTION - Ports imported from @rbee/shared-config (no duplication)
 */
export const SERVICES: Record<ServiceName, ServiceConfig> = {
  queen: {
    name: 'queen-rbee',
    devPort: PORTS.queen.dev,
    prodPort: PORTS.queen.prod,
    keeperDevPort: PORTS.keeper.dev,
    keeperProdOrigin: '*',  // Tauri app
  },
  hive: {
    name: 'rbee-hive',
    devPort: PORTS.hive.dev,
    prodPort: PORTS.hive.prod,
    keeperDevPort: PORTS.keeper.dev,
    keeperProdOrigin: '*',
  },
  worker: {
    name: 'llm-worker',
    devPort: PORTS.worker.dev,
    prodPort: PORTS.worker.prod,
    keeperDevPort: PORTS.keeper.dev,
    keeperProdOrigin: '*',
  },
} as const

// TEAM-351: Validate service names at module load
const validServiceNames: ServiceName[] = ['queen', 'hive', 'worker']
for (const serviceName of Object.keys(SERVICES)) {
  if (!validServiceNames.includes(serviceName as ServiceName)) {
    throw new Error(`Invalid service name: ${serviceName}`)
  }
}

/**
 * Get parent origin based on current service location
 * 
 * TEAM-351: Bug fixes - Handle missing port, validation
 * 
 * @param serviceConfig - Service configuration
 * @returns Parent origin for postMessage
 */
export function getParentOrigin(serviceConfig: ServiceConfig): string {
  // TEAM-351: Handle missing window.location.port (default to 80)
  const currentPort = window.location.port || '80'
  const isOnDevServer = currentPort === serviceConfig.devPort.toString()
  
  return isOnDevServer
    ? `http://localhost:${serviceConfig.keeperDevPort}`
    : serviceConfig.keeperProdOrigin
}

/**
 * Validate service configuration
 * TEAM-351: Runtime validation
 */
export function isValidServiceConfig(config: any): config is ServiceConfig {
  return (
    config !== null &&
    typeof config === 'object' &&
    typeof config.name === 'string' && config.name.length > 0 &&
    typeof config.devPort === 'number' && config.devPort > 0 &&
    typeof config.prodPort === 'number' && config.prodPort > 0 &&
    typeof config.keeperDevPort === 'number' && config.keeperDevPort > 0 &&
    typeof config.keeperProdOrigin === 'string' && config.keeperProdOrigin.length > 0
  )
}
