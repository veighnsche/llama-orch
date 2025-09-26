#!/usr/bin/env node
/*
  i18n-check.mjs
  CI-friendly check: ensures there are no dynamic i18n warnings and no missing keys
  in locale JSONs compared to extracted keys.

  Usage:
    pnpm run i18n:extract && node scripts/i18n-check.mjs
*/
import { promises as fs } from 'node:fs'
import path from 'node:path'

const projectRoot = process.cwd()
const i18nDir = path.join(projectRoot, 'src', 'i18n')

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

  if (hasError) {
    process.exitCode = 1
  } else {
    console.log(`[i18n-check] OK — ${keys.length} keys, no dynamic usages, no missing translations.`)
  }
}

main().catch((err) => {
  console.error('[i18n-check] Failed:', err)
  process.exitCode = 1
})
