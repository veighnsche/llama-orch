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

// TEAM-351: Port validation constants
const MIN_PORT = 1
const MAX_PORT = 65535

/**
 * Validate port number is in valid range
 * @internal
 */
function isValidPort(port: number | null): port is number {
  return port !== null && Number.isInteger(port) && port >= MIN_PORT && port <= MAX_PORT
}

/**
 * Port configuration for each service
 * 
 * TEAM-351: Type-safe port configuration with validation
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

// TEAM-351: Validate all ports at module load time
for (const [serviceName, ports] of Object.entries(PORTS)) {
  for (const [portType, portValue] of Object.entries(ports)) {
    if (portValue !== null && !isValidPort(portValue)) {
      throw new Error(
        `Invalid port configuration: ${serviceName}.${portType} = ${portValue} (must be 1-65535)`
      )
    }
  }
}

export type ServiceName = keyof typeof PORTS

/**
 * Generate allowed origins for postMessage listener
 * Automatically includes all dev and prod ports
 * 
 * TEAM-351: Bug fixes - Type safety, deduplication, HTTPS support
 * 
 * @param includeHttps - Include HTTPS variants for production (default: false)
 * @returns Array of unique allowed origins
 */
export function getAllowedOrigins(includeHttps = false): string[] {
  const origins = new Set<string>()  // TEAM-351: Use Set to prevent duplicates
  
  // TEAM-351: Type-safe iteration using ServiceName
  const serviceNames: ServiceName[] = ['queen', 'hive', 'worker']  // Exclude keeper
  
  for (const service of serviceNames) {
    const ports = PORTS[service]
    
    // Add dev port (always HTTP)
    if (isValidPort(ports.dev)) {
      origins.add(`http://localhost:${ports.dev}`)
    }
    
    // Add prod port (HTTP + optional HTTPS)
    if (isValidPort(ports.prod)) {
      origins.add(`http://localhost:${ports.prod}`)
      if (includeHttps) {
        origins.add(`https://localhost:${ports.prod}`)
      }
    }
  }
  
  return Array.from(origins).sort()  // TEAM-351: Sort for deterministic output
}

/**
 * Get iframe URL for a service
 * 
 * TEAM-351: Bug fixes - Validation, error messages, HTTPS support
 * 
 * @param service - Service name
 * @param isDev - Development mode flag
 * @param useHttps - Use HTTPS instead of HTTP (default: false)
 * @returns URL string or empty string if service has no HTTP port
 * @throws Error if service doesn't support iframe embedding (e.g., keeper)
 */
export function getIframeUrl(
  service: ServiceName,
  isDev: boolean,
  useHttps = false
): string {
  const ports = PORTS[service]
  const port = isDev ? ports.dev : ports.prod
  
  // TEAM-351: Explicit error for keeper (no HTTP port in prod)
  if (service === 'keeper' && !isDev) {
    throw new Error(
      'Keeper service has no production HTTP port (Tauri app). Use dev mode or check service name.'
    )
  }
  
  if (!isValidPort(port)) {
    return ''  // TEAM-351: Return empty string for null ports
  }
  
  const protocol = useHttps ? 'https' : 'http'
  return `${protocol}://localhost:${port}`
}

/**
 * Get parent origin for postMessage
 * Detects environment based on current port
 * 
 * TEAM-351: Bug fixes - Handle null ports, validation
 * 
 * @param currentPort - Current window port
 * @returns Origin string or '*' for wildcard (Tauri app)
 * @throws Error if currentPort is invalid
 */
export function getParentOrigin(currentPort: number): string {
  // TEAM-351: Validate input port
  if (!isValidPort(currentPort)) {
    throw new Error(`Invalid port: ${currentPort} (must be 1-65535)`)
  }
  
  // TEAM-351: Type-safe check - only check non-null dev ports
  const isDevPort = (
    currentPort === PORTS.queen.dev ||
    currentPort === PORTS.hive.dev ||
    currentPort === PORTS.worker.dev ||
    currentPort === PORTS.keeper.dev
  )
  
  return isDevPort
    ? `http://localhost:${PORTS.keeper.dev}`  // Dev: Keeper Vite
    : '*'                                       // Prod: Tauri app
}

/**
 * Get service URL for HTTP requests
 * 
 * TEAM-351: Bug fixes - Validation, backend port support, HTTPS
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
  
  // TEAM-351: Support backend port mode
  let port: number | null
  if (mode === 'backend') {
    port = 'backend' in ports ? (ports as any).backend : ports.prod
  } else {
    port = mode === 'dev' ? ports.dev : ports.prod
  }
  
  if (!isValidPort(port)) {
    return ''  // TEAM-351: Return empty string for null ports
  }
  
  const protocol = useHttps ? 'https' : 'http'
  return `${protocol}://localhost:${port}`
}
