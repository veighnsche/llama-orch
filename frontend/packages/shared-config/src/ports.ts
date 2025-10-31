/**
 * SINGLE SOURCE OF TRUTH for all port configurations
 * 
 * CRITICAL: When adding a new service:
 * 1. Add to PORTS constant
 * 2. Update PORT_CONFIGURATION.md
 * 3. Run `pnpm generate:rust`
 * 4. Update backend Cargo.toml default port
 * 
 * TEAM-351: Shared port configuration
 * TEAM-351: Bug fixes - Type safety, validation, edge cases
 * 
 * @packageDocumentation
 */

/**
 * Port configuration for each service
 */
export const PORTS = {
  keeper: {
    dev: 5173,
    prod: null,  // Tauri app, no HTTP port
  },
  queen: {
    dev: 7834,      // Vite dev server
    prod: 7833,     // Embedded in backend
    backend: 7833,  // Backend HTTP server
  },
  hive: {
    dev: 7836,
    prod: 7835,
    backend: 7835,
  },
  worker: {
    dev: 7837,
    prod: 8080,
    backend: 8080,
  },
} as const

export type ServiceName = keyof typeof PORTS

/**
 * Generate allowed origins for postMessage listener
 * 
 * @param includeHttps - Include HTTPS variants for production (default: false)
 * @returns Array of unique allowed origins
 */
export function getAllowedOrigins(includeHttps = false): string[] {
  const origins = new Set<string>()
  
  const serviceNames: ServiceName[] = ['queen', 'hive', 'worker']
  
  for (const service of serviceNames) {
    const ports = PORTS[service]
    
    if (ports.dev !== null) {
      origins.add(`http://localhost:${ports.dev}`)
    }
    
    if (ports.prod !== null) {
      origins.add(`http://localhost:${ports.prod}`)
      if (includeHttps) {
        origins.add(`https://localhost:${ports.prod}`)
      }
    }
  }
  
  return Array.from(origins).sort()
}

/**
 * Get iframe URL for embedding services
 * 
 * TEAM-374: In dev mode, use /dev proxy for hive and queen
 * This allows Keeper to load the UI from the backend's /dev proxy,
 * which forwards to the Vite dev server. This avoids CORS issues.
 * 
 * @param service - Service name
 * @param isDev - Whether in development mode
 * @param useHttps - Whether to use HTTPS (default: false)
 * @returns URL string or empty string if service has no HTTP port
 * @throws Error if service doesn't support iframe embedding
 */
export function getIframeUrl(
  service: ServiceName,
  isDev: boolean,
  useHttps = false
): string {
  const ports = PORTS[service]
  const port = isDev ? ports.dev : ports.prod
  
  if (service === 'keeper' && !isDev) {
    throw new Error(
      'Keeper service has no production HTTP port (Tauri app). Use dev mode or check service name.'
    )
  }
  
  if (port === null) {
    return ''
  }
  
  const protocol = useHttps ? 'https' : 'http'
  
  // TEAM-374: In dev mode, use /dev proxy for hive and queen
  // This loads the UI from the backend's /dev proxy instead of directly from Vite
  // Backend proxies /dev → Vite dev server, avoiding CORS issues
  if (isDev && (service === 'hive' || service === 'queen')) {
    const prodPort = ports.prod
    return `${protocol}://localhost:${prodPort}/dev`
  }
  
  return `${protocol}://localhost:${port}`
}

/**
 * Get parent origin for postMessage
 * 
 * @param currentPort - Current window port
 * @returns Origin string or '*' for wildcard (Tauri app)
 */
export function getParentOrigin(currentPort: number): string {
  const isDevPort = (
    currentPort === PORTS.queen.dev ||
    currentPort === PORTS.hive.dev ||
    currentPort === PORTS.worker.dev ||
    currentPort === PORTS.keeper.dev
  )
  
  return isDevPort
    ? `http://localhost:${PORTS.keeper.dev}`
    : '*'
}

/**
 * Get service URL for HTTP requests
 * 
 * @param service - Service name
 * @param mode - 'dev' | 'prod' | 'backend'
 * @param useHttps - Use HTTPS instead of HTTP (default: false)
 * @returns URL string or empty string if port is null
 */
export function getServiceUrl(
  service: ServiceName,
  mode: 'dev' | 'prod' | 'backend' = 'dev',
  useHttps = false
): string {
  const ports = PORTS[service]
  
  let port: number | null
  if (mode === 'backend') {
    port = 'backend' in ports ? (ports as any).backend : ports.prod
  } else {
    port = mode === 'dev' ? ports.dev : ports.prod
  }
  
  if (port === null) {
    return ''
  }
  
  const protocol = useHttps ? 'https' : 'http'
  return `${protocol}://localhost:${port}`
}
