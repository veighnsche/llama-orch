#!/usr/bin/env node
/*
  i18n-merge.mjs
  Merges extracted keys (src/i18n/extracted/messages.base.json) into locale JSON files
  (src/i18n/en.json, src/i18n/nl.json). Preserves existing translations and adds any
  missing keys (initialized to the extracted default string or empty string).

  Usage:
    node scripts/i18n-merge.mjs
*/
import { promises as fs } from 'node:fs'
import path from 'node:path'

const projectRoot = process.cwd()
const i18nDir = path.join(projectRoot, 'src', 'i18n')
const extractedBasePath = path.join(i18nDir, 'extracted', 'messages.base.json')
const locales = ['en', 'nl']

function setDeep(obj, dottedKey, value) {
  const parts = dottedKey.split('.')
  let cur = obj
  for (let i = 0; i < parts.length; i++) {
    const k = parts[i]
    if (i === parts.length - 1) {
      if (typeof cur[k] === 'undefined') cur[k] = value
    } else {
      if (typeof cur[k] !== 'object' || cur[k] === null) cur[k] = {}
      cur = cur[k]
    }
  }
}

async function readJson(p) {
  const txt = await fs.readFile(p, 'utf8')
  return JSON.parse(txt)
}

async function writeJson(p, obj) {
  const txt = JSON.stringify(obj, null, 2) + '\n'
  await fs.writeFile(p, txt, 'utf8')
}

async function main() {
  const exists = await fs
    .access(extractedBasePath)
    .then(() => true)
    .catch(() => false)
  if (!exists) {
    console.error(
      `[i18n-merge] Missing ${path.relative(projectRoot, extractedBasePath)}. Run i18n:extract first.`,
    )
    process.exit(1)
  }
  const base = await readJson(extractedBasePath)

  for (const loc of locales) {
    const locPath = path.join(i18nDir, `${loc}.json`)
    const hasLoc = await fs
      .access(locPath)
      .then(() => true)
      .catch(() => false)
    if (!hasLoc) {
      console.warn(`[i18n-merge] Locale JSON missing, creating: ${path.relative(projectRoot, locPath)}`)
      await writeJson(locPath, {})
    }
    const localeObj = await readJson(locPath)
    let added = 0

    for (const [key, defVal] of Object.entries(base)) {
      // do not overwrite existing
      let existsDeep = true
      let cur = localeObj
      const parts = key.split('.')
      for (let i = 0; i < parts.length; i++) {
        const k = parts[i]
        if (!(k in cur)) {
          existsDeep = false
          break
        }
        cur = cur[k]
        if (cur === null || typeof cur !== 'object') {
          if (i < parts.length - 1) {
            // path collision: treat as missing
            existsDeep = false
          }
          break
        }
      }
      if (!existsDeep) {
        setDeep(localeObj, key, typeof defVal === 'string' ? defVal : '')
        added++
      }
    }

    await writeJson(locPath, localeObj)
    console.log(
      `[i18n-merge] Updated ${path.relative(projectRoot, locPath)} — added ${added} missing keys.`,
    )
  }
}

main().catch((err) => {
  console.error('[i18n-merge] Failed:', err)
  process.exitCode = 1
})
