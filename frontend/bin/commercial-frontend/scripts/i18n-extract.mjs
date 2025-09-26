#!/usr/bin/env node
/*
  i18n-extract.mjs
  Scans .vue/.ts/.js files for t('key', 'Default') / $t('key', 'Default') usages
  and generates JSON artifacts under src/i18n/extracted/:
    - messages.catalog.json  // detailed info (default string if present, files where found)
    - messages.keys.json     // sorted list of keys
    - messages.base.json     // key -> default (or empty string if none)

  Usage:
    node scripts/i18n-extract.mjs
*/
import { promises as fs } from 'node:fs'
import path from 'node:path'

const projectRoot = process.cwd()
const srcDir = path.join(projectRoot, 'src')
const outDir = path.join(projectRoot, 'src', 'i18n', 'extracted')

// Directories to skip when walking
const SKIP_DIRS = new Set([
  'node_modules',
  'dist',
  '.git',
  '.cache',
  '.storybook',
  // avoid re-parsing generated outputs
  path.join('src', 'i18n', 'extracted'),
])

// File extensions to include
const EXTENSIONS = new Set(['.vue', '.ts', '.js', '.tsx', '.jsx'])

// Regex to capture t('key', 'Default') and $t("key") patterns
// Captures:
//   1: function name (t or $t)
//   2: opening quote of key
//   3: key
//   4: optional default opening quote
//   5: optional default string
const I18N_CALL_RE = /(\$t|\bt)\(\s*(["'`])([^"'`]+)\2\s*(?:,\s*(["'`])([\s\S]*?)\4\s*)?\)/g

async function pathExists(p) {
  try {
    await fs.access(p)
    return true
  } catch {
    return false
  }
}

async function* walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      // skip well-known directories
      const rel = path.relative(projectRoot, full)
      const first = rel.split(path.sep)[0]
      if (SKIP_DIRS.has(entry.name) || SKIP_DIRS.has(rel) || SKIP_DIRS.has(first)) continue
      yield* walk(full)
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name)
      if (EXTENSIONS.has(ext)) {
        yield full
      }
    }
  }
}

function extractFromContent(content) {
  const matches = []
  I18N_CALL_RE.lastIndex = 0
  let m
  while ((m = I18N_CALL_RE.exec(content)) !== null) {
    const key = m[3]
    const defaultStr = typeof m[5] === 'string' ? m[5] : ''
    matches.push({ key, defaultStr })
  }
  return matches
}

async function extract() {
  const catalog = new Map() // key -> { default: string, files: Set<string>, count: number }

  for await (const file of walk(srcDir)) {
    const content = await fs.readFile(file, 'utf8')
    const found = extractFromContent(content)
    if (found.length === 0) continue

    const relFile = path.relative(projectRoot, file)
    for (const { key, defaultStr } of found) {
      if (!catalog.has(key)) {
        catalog.set(key, { default: defaultStr || '', files: new Set([relFile]), count: 1 })
      } else {
        const item = catalog.get(key)
        item.count += 1
        item.files.add(relFile)
        // Preserve the first non-empty default encountered
        if (!item.default && defaultStr) item.default = defaultStr
      }
    }
  }

  // Ensure output dir exists
  await fs.mkdir(outDir, { recursive: true })

  // Build JSON artifacts
  const keys = Array.from(catalog.keys()).sort()
  const base = {}
  const detailed = {}

  for (const key of keys) {
    const item = catalog.get(key)
    base[key] = item.default || ''
    detailed[key] = {
      default: item.default || '',
      files: Array.from(item.files).sort(),
      count: item.count,
    }
  }

  // Write files
  const pretty = (obj) => JSON.stringify(obj, null, 2) + '\n'
  await fs.writeFile(path.join(outDir, 'messages.keys.json'), pretty(keys), 'utf8')
  await fs.writeFile(path.join(outDir, 'messages.base.json'), pretty(base), 'utf8')
  await fs.writeFile(path.join(outDir, 'messages.catalog.json'), pretty(detailed), 'utf8')

  return { written: 3, keys: keys.length, outDir }
}

extract()
  .then((res) => {
    // eslint-disable-next-line no-console
    console.log(`[i18n-extract] Wrote ${res.written} files to ${res.outDir} (${res.keys} keys).`)
  })
  .catch((err) => {
    // eslint-disable-next-line no-console
    console.error('[i18n-extract] Failed:', err)
    process.exitCode = 1
  })
