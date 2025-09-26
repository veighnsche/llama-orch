#!/usr/bin/env node
/*
  i18n-check.mjs
  CI-friendly check: ensures there are no dynamic i18n warnings and no missing keys
  in locale JSONs compared to extracted keys. Also detects UN-USED keys present in
  locale JSONs but not referenced by the source code, and emits a report.

  Usage:
    pnpm run i18n:extract && node scripts/i18n-check.mjs [--fail-on-unused] [--print-unused] [--print-missing] [--report]
*/
import { promises as fs } from 'node:fs'
import path from 'node:path'

const projectRoot = process.cwd()
const i18nDir = path.join(projectRoot, 'src', 'i18n')
const outDir = path.join(i18nDir, 'extracted')

async function readJson(p) {
  const txt = await fs.readFile(p, 'utf8')
  return JSON.parse(txt)
}

function hasDeep(obj, dotted) {
  const parts = dotted.split('.')
  let cur = obj
  for (let i = 0; i < parts.length; i++) {
    const k = parts[i]
    if (cur == null || typeof cur !== 'object' || !(k in cur)) return false
    cur = cur[k]
  }
  return true
}

function flattenKeys(obj, prefix = '') {
  const out = []
  if (obj == null) return out
  const isLeaf = typeof obj !== 'object'
  if (isLeaf) {
    if (prefix) out.push(prefix)
    return out
  }
  for (const k of Object.keys(obj)) {
    const dotted = prefix ? `${prefix}.${k}` : k
    const val = obj[k]
    if (val != null && typeof val === 'object') {
      out.push(...flattenKeys(val, dotted))
    } else {
      out.push(dotted)
    }
  }
  return out
}

function parseFlags(argv) {
  return {
    failOnUnused: argv.includes('--fail-on-unused'),
    printUnused: argv.includes('--print-unused'),
    printMissing: argv.includes('--print-missing'),
    report: argv.includes('--report'),
  }
}

async function main() {
  const keysPath = path.join(i18nDir, 'extracted', 'messages.keys.json')
  const warnPath = path.join(i18nDir, 'extracted', 'messages.warnings.json')
  const enPath = path.join(i18nDir, 'en.json')
  const nlPath = path.join(i18nDir, 'nl.json')

  const [keys, warnings, en, nl] = await Promise.all([
    readJson(keysPath),
    readJson(warnPath).catch(() => []),
    readJson(enPath),
    readJson(nlPath),
  ])

  const missing = { en: [], nl: [] }
  for (const k of keys) {
    if (!hasDeep(en, k)) missing.en.push(k)
    if (!hasDeep(nl, k)) missing.nl.push(k)
  }

  // Compute unused keys per locale (present in locale but not referenced in code)
  const used = new Set(keys)
  const enAll = flattenKeys(en).sort()
  const nlAll = flattenKeys(nl).sort()
  const unused = {
    en: enAll.filter((k) => !used.has(k)),
    nl: nlAll.filter((k) => !used.has(k)),
  }

  // Emit unused report next to other artifacts
  const pretty = (obj) => JSON.stringify(obj, null, 2) + '\n'
  await fs.mkdir(outDir, { recursive: true })
  await fs.writeFile(path.join(outDir, 'messages.unused.json'), pretty(unused), 'utf8')

  let hasError = false
  if (warnings.length) {
    hasError = true
    console.error(`[i18n-check] Found ${warnings.length} dynamic usages (see ${path.relative(projectRoot, warnPath)}):`)
    for (const w of warnings.slice(0, 10)) {
      console.error(`  ${w.file}:${w.line} ${w.type} — ${w.snippet?.replace(/\n/g, ' ')}`)
    }
    if (warnings.length > 10) console.error(`  ...and ${warnings.length - 10} more`)
  }

  const totalMissing = missing.en.length + missing.nl.length
  if (totalMissing) {
    hasError = true
    console.error(`[i18n-check] Missing keys — en: ${missing.en.length}, nl: ${missing.nl.length}`)
  }

  const flags = parseFlags(process.argv.slice(2))
  const totalUnused = unused.en.length + unused.nl.length
  if (totalUnused) {
    console.warn(`[i18n-check] Unused keys — en: ${unused.en.length}, nl: ${unused.nl.length}`)
    if (flags.printUnused || flags.report) {
      const max = 50
      if (unused.en.length) {
        console.warn(`  en (showing up to ${Math.min(max, unused.en.length)}):`)
        for (const k of unused.en.slice(0, max)) console.warn(`    - ${k}`)
        if (unused.en.length > max) console.warn(`    ...and ${unused.en.length - max} more`)
      }
      if (unused.nl.length) {
        console.warn(`  nl (showing up to ${Math.min(max, unused.nl.length)}):`)
        for (const k of unused.nl.slice(0, max)) console.warn(`    - ${k}`)
        if (unused.nl.length > max) console.warn(`    ...and ${unused.nl.length - max} more`)
      }
    }
    if (flags.failOnUnused) hasError = true
  }

  if ((flags.printMissing || flags.report) && totalMissing) {
    const max = 50
    if (missing.en.length) {
      console.error(`  Missing in en (showing up to ${Math.min(max, missing.en.length)}):`)
      for (const k of missing.en.slice(0, max)) console.error(`    - ${k}`)
      if (missing.en.length > max) console.error(`    ...and ${missing.en.length - max} more`)
    }
    if (missing.nl.length) {
      console.error(`  Missing in nl (showing up to ${Math.min(max, missing.nl.length)}):`)
      for (const k of missing.nl.slice(0, max)) console.error(`    - ${k}`)
      if (missing.nl.length > max) console.error(`    ...and ${missing.nl.length - max} more`)
    }
  }

  if (hasError) {
    process.exitCode = 1
  } else {
    console.log(`[i18n-check] OK — ${keys.length} keys, no dynamic usages, no missing translations, ${totalUnused} unused.`)
  }
}

main().catch((err) => {
  console.error('[i18n-check] Failed:', err)
  process.exitCode = 1
})
