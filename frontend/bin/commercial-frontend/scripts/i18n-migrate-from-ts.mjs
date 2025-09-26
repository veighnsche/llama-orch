#!/usr/bin/env node
/*
  i18n-migrate-from-ts.mjs
  Converts src/i18n/en.ts and nl.ts (object literals) into en.json and nl.json.
  Uses a simple balanced-brace extractor + JSON5 to parse comments, single quotes,
  and trailing commas safely.

  Usage:
    node scripts/i18n-migrate-from-ts.mjs
*/
import { promises as fs } from 'node:fs'
import path from 'node:path'
import JSON5 from 'json5'

const projectRoot = process.cwd()
const srcDir = path.join(projectRoot, 'src', 'i18n')

const files = [
  { constName: 'en', inFile: path.join(srcDir, 'en.ts'), outFile: path.join(srcDir, 'en.json') },
  { constName: 'nl', inFile: path.join(srcDir, 'nl.ts'), outFile: path.join(srcDir, 'nl.json') },
]

function extractObjectLiteral(content, constName) {
  const declRe = new RegExp(`\\bconst\\s+${constName}\\s*=`) // const <name> =
  const declMatch = declRe.exec(content)
  if (!declMatch) throw new Error(`Could not find declaration for const ${constName}`)
  let i = content.indexOf('{', declMatch.index)
  if (i === -1) throw new Error(`Could not find opening '{' for ${constName}`)

  let depth = 0
  let inString = null // '"' | "'" | '`'
  let escape = false
  let buf = ''

  for (; i < content.length; i++) {
    const ch = content[i]
    if (inString) {
      buf += ch
      if (escape) {
        escape = false
        continue
      }
      if (ch === '\\') {
        escape = true
        continue
      }
      if (ch === inString) {
        inString = null
      }
      continue
    }
    // not in string
    if (ch === '"' || ch === '\'' || ch === '`') {
      inString = ch
      buf += ch
      continue
    }
    if (ch === '{') {
      depth++
      buf += ch
      continue
    }
    if (ch === '}') {
      depth--
      buf += ch
      if (depth === 0) {
        // include this '}' and stop
        i++
        break
      }
      continue
    }
    buf += ch
  }

  if (depth !== 0) throw new Error(`Unbalanced braces while parsing ${constName}`)
  return buf
}

async function migrateOne({ constName, inFile, outFile }) {
  const exists = await fs
    .access(inFile)
    .then(() => true)
    .catch(() => false)
  if (!exists) {
    console.warn(`[i18n-migrate] Skip missing ${path.relative(projectRoot, inFile)}`)
    return false
  }

  const content = await fs.readFile(inFile, 'utf8')
  const objLiteral = extractObjectLiteral(content, constName)
  const obj = JSON5.parse(objLiteral)

  const pretty = JSON.stringify(obj, null, 2) + '\n'
  await fs.writeFile(outFile, pretty, 'utf8')
  console.log(`[i18n-migrate] Wrote ${path.relative(projectRoot, outFile)}`)
  return true
}

async function main() {
  let wrote = 0
  for (const f of files) {
    const ok = await migrateOne(f)
    if (ok) wrote++
  }
  if (wrote === 0) {
    console.warn('[i18n-migrate] No locale TS files found to migrate.')
  }
}

main().catch((err) => {
  console.error('[i18n-migrate] Failed:', err)
  process.exitCode = 1
})
