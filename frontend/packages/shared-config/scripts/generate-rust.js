#!/usr/bin/env node

// TEAM-351: Rust constant generator
// TEAM-351: Bug fixes - Import from source, validation, error handling

import { writeFileSync, mkdirSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { readFileSync } from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// TEAM-351: Import ports from TypeScript source (single source of truth)
const portsPath = join(__dirname, '..', 'src', 'ports.ts')
const portsSource = readFileSync(portsPath, 'utf8')

// TEAM-351: Extract PORTS constant from TypeScript using regex
const portsMatch = portsSource.match(/export const PORTS = ({[\s\S]*?}) as const/)
if (!portsMatch) {
  console.error('❌ Failed to extract PORTS constant from ports.ts')
  process.exit(1)
}

// TEAM-351: Parse the PORTS object (eval is safe here - we control the source)
let PORTS
try {
  // Convert TypeScript object literal to JavaScript
  const portsCode = portsMatch[1]
    .replace(/\/\/.*$/gm, '')  // Remove single-line comments
    .replace(/\/\*[\s\S]*?\*\//g, '')  // Remove multi-line comments
  
  PORTS = eval(`(${portsCode})`)
} catch (error) {
  console.error('❌ Failed to parse PORTS constant:', error.message)
  process.exit(1)
}

// TEAM-351: Validate port configuration
const MIN_PORT = 1
const MAX_PORT = 65535

function validatePort(port, serviceName, portType) {
  if (port === null) return true  // null is valid (e.g., keeper.prod)
  
  if (!Number.isInteger(port) || port < MIN_PORT || port > MAX_PORT) {
    console.error(`❌ Invalid port: ${serviceName}.${portType} = ${port} (must be 1-65535 or null)`)
    process.exit(1)
  }
  return true
}

// TEAM-351: Validate all ports
for (const [serviceName, ports] of Object.entries(PORTS)) {
  for (const [portType, portValue] of Object.entries(ports)) {
    validatePort(portValue, serviceName, portType)
  }
}

// TEAM-351: Generate Rust constants with proper formatting
function generateRustConstant(name, value) {
  if (value === null) {
    return `// ${name} is null (no HTTP port)`
  }
  return `pub const ${name}: u16 = ${value};`
}

const rustCode = `// AUTO-GENERATED from frontend/packages/shared-config/src/ports.ts
// DO NOT EDIT MANUALLY - Run 'pnpm generate:rust' in shared-config package to update
// 
// This file provides port constants for Rust build.rs scripts
// Last generated: ${new Date().toISOString()}

// TEAM-351: Shared port configuration constants
// TEAM-351: Bug fixes - Validation, error handling, null port comments

${generateRustConstant('KEEPER_DEV_PORT', PORTS.keeper.dev)}
${generateRustConstant('KEEPER_PROD_PORT', PORTS.keeper.prod)}

${generateRustConstant('QUEEN_DEV_PORT', PORTS.queen.dev)}
${generateRustConstant('QUEEN_PROD_PORT', PORTS.queen.prod)}
${generateRustConstant('QUEEN_BACKEND_PORT', PORTS.queen.backend)}

${generateRustConstant('HIVE_DEV_PORT', PORTS.hive.dev)}
${generateRustConstant('HIVE_PROD_PORT', PORTS.hive.prod)}
${generateRustConstant('HIVE_BACKEND_PORT', PORTS.hive.backend)}

${generateRustConstant('WORKER_DEV_PORT', PORTS.worker.dev)}
${generateRustConstant('WORKER_PROD_PORT', PORTS.worker.prod)}
${generateRustConstant('WORKER_BACKEND_PORT', PORTS.worker.backend)}
`

const outputPath = join(__dirname, '..', '..', '..', 'shared-constants.rs')

// TEAM-351: Ensure output directory exists
try {
  mkdirSync(dirname(outputPath), { recursive: true })
} catch (error) {
  // Directory already exists, ignore
}

// TEAM-351: Write file with error handling
try {
  writeFileSync(outputPath, rustCode, 'utf8')
  console.log('✅ Generated Rust constants at:', outputPath)
  console.log('✅ Validated', Object.keys(PORTS).length, 'services')
} catch (error) {
  console.error('❌ Failed to write Rust constants:', error.message)
  process.exit(1)
}
