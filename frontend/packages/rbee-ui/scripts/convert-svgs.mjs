#!/usr/bin/env node
/**
 * Convert SVG files to React components
 * Reads all SVG files from src/assets/illustrations and creates TSX components
 */

import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { basename, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const SVG_DIR = join(__dirname, '..', 'src', 'assets', 'illustrations')
const OUTPUT_DIR = join(__dirname, '..', 'src', 'icons')

// Convert kebab-case to PascalCase
function toPascalCase(str) {
  return str
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join('')
}

// Extract SVG content without the outer <svg> tag
function extractSvgContent(svgString) {
  const match = svgString.match(/<svg[^>]*>([\s\S]*)<\/svg>/)
  if (!match) return ''

  let content = match[1].trim()
  // Convert HTML comments to JSX comments
  content = content.replace(/<!--([\s\S]*?)-->/g, (_match, comment) => {
    return `{/* ${comment.trim()} */}`
  })

  return content
}

// Extract viewBox from SVG
function extractViewBox(svgString) {
  const match = svgString.match(/viewBox="([^"]*)"/)
  return match ? match[1] : '0 0 24 24'
}

// Extract width and height if present
function extractDimensions(svgString) {
  const widthMatch = svgString.match(/width="([^"]*)"/)
  const heightMatch = svgString.match(/height="([^"]*)"/)
  return {
    width: widthMatch ? widthMatch[1] : null,
    height: heightMatch ? heightMatch[1] : null,
  }
}

// Create React component from SVG
function createComponent(svgContent, componentName, viewBox, dimensions) {
  const defaultSize = dimensions.width || '24'

  return `import type { SVGProps } from 'react'

export interface ${componentName}Props extends SVGProps<SVGSVGElement> {
  size?: number | string
}

export function ${componentName}({ size = ${defaultSize}, className, ...props }: ${componentName}Props) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="${viewBox}"
      className={className}
      {...props}
    >
      ${svgContent}
    </svg>
  )
}
`
}

// Main conversion function
function convertSvgs() {
  console.log('🎨 Converting SVG files to React components...\n')

  const files = readdirSync(SVG_DIR).filter((f) => f.endsWith('.svg'))
  const exports = []

  for (const file of files) {
    const svgPath = join(SVG_DIR, file)
    const svgString = readFileSync(svgPath, 'utf-8')

    const componentName = toPascalCase(basename(file, '.svg'))
    const svgContent = extractSvgContent(svgString)
    const viewBox = extractViewBox(svgString)
    const dimensions = extractDimensions(svgString)

    const component = createComponent(svgContent, componentName, viewBox, dimensions)

    const outputPath = join(OUTPUT_DIR, `${componentName}.tsx`)
    writeFileSync(outputPath, component)

    exports.push(`export { ${componentName} } from './${componentName}'`)

    console.log(`✅ Created: ${componentName}.tsx`)
  }

  // Create index.ts
  const indexContent = `${exports.join('\n')}\n`
  writeFileSync(join(OUTPUT_DIR, 'index.ts'), indexContent)

  console.log(`\n✨ Converted ${files.length} SVG files to React components`)
  console.log(`📦 Index file created at: src/icons/index.ts`)
}

convertSvgs()
