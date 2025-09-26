#!/usr/bin/env node
/*
  i18n-prune.mjs
  Safely prune UNUSED i18n keys from locale JSON files by comparing them with
  extracted usage (messages.keys.json). Defaults to DRY-RUN. Use --write to
  persist changes. Optionally prunes empty objects after deletions.

  Usage examples:
    # Dry run (default), prints summary and writes extracted/messages.prune.dryrun.json
    pnpm run i18n:extract && node scripts/i18n-prune.mjs

    # Actually write changes with backups
    pnpm run i18n:extract && node scripts/i18n-prune.mjs --write

    # Only prune for a specific locale
    node scripts/i18n-prune.mjs --write --locale en

    # Also remove empty objects left behind by deletions
    node scripts/i18n-prune.mjs --write --prune-empty-objects
*/
import { promises as fs } from 'node:fs'
import path from 'node:path'

const projectRoot = process.cwd()
const i18nDir = path.join(projectRoot, 'src', 'i18n')
const outDir = path.join(i18nDir, 'extracted')

function parseArgs(argv) {
  const args = {
    write: argv.includes('--write'),
    pruneEmpty: argv.includes('--prune-empty-objects'),
    noBackup: argv.includes('--no-backup'),
    print: argv.includes('--print') || argv.includes('--print-unused'),
    locale: 'all',
    keysPath: path.join(outDir, 'messages.keys.json'),
  }
  const locIdx = argv.indexOf('--locale')
  if (locIdx >= 0 && argv[locIdx + 1]) args.locale = argv[locIdx + 1]
  const keysIdx = argv.indexOf('--keys')
  if (keysIdx >= 0 && argv[keysIdx + 1]) args.keysPath = argv[keysIdx + 1]
  return args
}

async function readJson(p) {
  const txt = await fs.readFile(p, 'utf8')
  return JSON.parse(txt)
}

function flattenKeys(obj, prefix = '') {
  const out = []
  if (obj == null) return out
  if (typeof obj !== 'object') {
    if (prefix) out.push(prefix)
    return out
  }
  for (const k of Object.keys(obj)) {
    const dotted = prefix ? `${prefix}.${k}` : k
    const val = obj[k]
    if (val != null && typeof val === 'object') out.push(...flattenKeys(val, dotted))
    else out.push(dotted)
  }
  return out
}

function deleteDeep(obj, dotted) {
  const parts = dotted.split('.')
  let cur = obj
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i]
    if (cur == null || typeof cur !== 'object' || !(k in cur)) return false
    cur = cur[k]
  }
  const leaf = parts[parts.length - 1]
  if (cur && typeof cur === 'object' && leaf in cur) {
    delete cur[leaf]
    return true
  }
  return false
}

function pruneEmptyObjects(obj) {
  if (obj == null || typeof obj !== 'object') return obj
  for (const k of Object.keys(obj)) {
    const v = obj[k]
    if (v && typeof v === 'object') {
      pruneEmptyObjects(v)
      if (typeof v === 'object' && v && Object.keys(v).length === 0) {
        delete obj[k]
      }
    }
  }
  return obj
}

async function backupFile(p) {
  const dir = path.dirname(p)
  const base = path.basename(p)
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const target = path.join(dir, `${base}.bak-${stamp}`)
  await fs.copyFile(p, target)
  return target
}

async function main() {
  const args = parseArgs(process.argv.slice(2))

  // Load data
  const keys = await readJson(args.keysPath)
  const used = new Set(keys)

  const locales = []
  if (args.locale === 'all' || args.locale === 'en') locales.push({ id: 'en', path: path.join(i18nDir, 'en.json') })
  if (args.locale === 'all' || args.locale === 'nl') locales.push({ id: 'nl', path: path.join(i18nDir, 'nl.json') })

  const results = { dryRun: !args.write, locales: {} }

  for (const loc of locales) {
    const obj = await readJson(loc.path)
    const allKeys = flattenKeys(obj)
    const toPrune = allKeys.filter((k) => !used.has(k))

    results.locales[loc.id] = { total: allKeys.length, unused: toPrune.length, keys: toPrune }

    if (!args.write || toPrune.length === 0) continue

    if (!args.noBackup) {
      const bak = await backupFile(loc.path)
      // eslint-disable-next-line no-console
      console.log(`[i18n-prune] Backed up ${path.relative(projectRoot, loc.path)} -> ${path.relative(projectRoot, bak)}`)
    }

    for (const k of toPrune) deleteDeep(obj, k)
    if (args.pruneEmpty) pruneEmptyObjects(obj)

    const pretty = JSON.stringify(obj, null, 2) + '\n'
    await fs.writeFile(loc.path, pretty, 'utf8')
    // eslint-disable-next-line no-console
    console.log(`[i18n-prune] Wrote ${path.relative(projectRoot, loc.path)} (removed ${toPrune.length} keys${args.pruneEmpty ? ', pruned empty objects' : ''}).`)
  }

  await fs.mkdir(outDir, { recursive: true })
  await fs.writeFile(path.join(outDir, 'messages.prune.' + (args.write ? 'result' : 'dryrun') + '.json'), JSON.stringify(results, null, 2) + '\n', 'utf8')

  const summary = Object.entries(results.locales)
    .map(([id, r]) => `${id}: ${r.unused}/${r.total} unused`)
    .join(', ')
  if (args.write) {
    console.log(`[i18n-prune] Done — ${summary}`)
  } else {
    console.log(`[i18n-prune] Dry-run — ${summary}`)
    if (args.print) {
      for (const [id, r] of Object.entries(results.locales)) {
        if (!r.keys?.length) continue
        console.log(`  ${id}:`)
        for (const k of r.keys.slice(0, 100)) console.log(`    - ${k}`)
        if (r.keys.length > 100) console.log(`    ...and ${r.keys.length - 100} more`)
      }
    }
  }
}

main().catch((err) => {
  console.error('[i18n-prune] Failed:', err)
  process.exitCode = 1
})
