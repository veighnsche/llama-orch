#!/usr/bin/env node
/*
  i18n-extract.mjs
  Scans .vue/.ts/.js files for t('key', 'Default') / $t('key', 'Default') / $tc('key') usages
  and Vue directives v-t="'key.path'" or v-t="{ path: 'key.path', ... }".
  and generates JSON artifacts under src/i18n/extracted/:
    - messages.catalog.json  // detailed info (default string if present, files where found)
    - messages.keys.json     // sorted list of keys
    - messages.base.json     // key -> default (or empty string if none)
    - messages.warnings.json // dynamic usages and patterns not auto-extractable

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
const I18N_CALL_RE = /(\$tc|\btc|\$t|\bt)\(\s*(["'`])([^"'`]+)\2\s*(?:,\s*(["'`])([\s\S]*?)\4\s*)?\)/g
// Dynamic (non-literal) calls: first arg is not a string literal
const I18N_DYNAMIC_CALL_RE = /(\$tc|\btc|\$t|\bt)\(\s*(?!["'`])/g
// v-t directive in templates: v-t="'key.path'"
const V_T_SIMPLE_RE = /v-t\s*=\s*(["'])([^"'`]+)\1/g
// v-t directive object form: v-t="{ path: 'key.path', ... }"
const V_T_OBJ_RE = /v-t\s*=\s*\{[^}]*?path\s*:\s*(["'])([^"']+)\1[^}]*?\}/g

// useI18n variable assignment e.g., const i18n = useI18n()
const USE_I18N_VAR_RE = /\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*useI18n\s*\(/g
// useI18n destructured e.g., const { t, n } = useI18n() or const { t: translate } = useI18n()
const USE_I18N_DESTRUCT_RE = /\b(?:const|let|var)\s*\{([^}]+)\}\s*=\s*useI18n\s*\(/g

function lineOf(text, index) {
  let line = 1
  for (let i = 0; i < index && i < text.length; i++) {
    if (text[i] === '\n') line++
  }
  return line
}

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
  const entries = []
  const warnings = []

  // Discover variable names assigned from useI18n()
  const i18nVarNames = new Set()
  let destructured = new Set()
  let destructuredLocalT = new Set()   // local names that refer to t()
  let destructuredLocalTc = new Set()  // local names that refer to tc()
  USE_I18N_VAR_RE.lastIndex = 0
  let v
  while ((v = USE_I18N_VAR_RE.exec(content)) !== null) {
    i18nVarNames.add(v[1])
  }
  // Discover destructured names
  USE_I18N_DESTRUCT_RE.lastIndex = 0
  let dv
  while ((dv = USE_I18N_DESTRUCT_RE.exec(content)) !== null) {
    const inside = dv[1]
    for (const raw of inside.split(',')) {
      const seg = raw.trim()
      if (!seg) continue
      const parts = seg.split(':').map((s) => s.trim())
      if (parts.length === 1) {
        const n = parts[0]
        if (n) {
          destructured.add(n)
          if (n === 't') destructuredLocalT.add('t')
          if (n === 'tc') destructuredLocalTc.add('tc')
        }
      } else {
        const [prop, local] = parts
        if (prop) destructured.add(prop)
        if (prop === 't' && local) destructuredLocalT.add(local)
        if (prop === 'tc' && local) destructuredLocalTc.add(local)
      }
    }
  }

  // Function calls: t/$t/tc/$tc with string literal keys
  I18N_CALL_RE.lastIndex = 0
  let m
  while ((m = I18N_CALL_RE.exec(content)) !== null) {
    const fn = m[1] // "$t" | "$tc" | "t" | "tc"
    // Only accept bare t()/tc() if we know the file uses useI18n() (reduces false positives)
    const isBare = fn === 't' || fn === 'tc'
    if (isBare) {
      if (fn === 't' && !destructured.has('t')) continue
      if (fn === 'tc' && !(destructured.has('tc') || destructured.has('t') || /useI18n\s*\(/.test(content))) continue
    }

    const key = m[3]
    const defaultStr = typeof m[5] === 'string' ? m[5] : ''
    entries.push({ key, defaultStr })
    // Template literal with interpolation — warn as dynamic
    const delim = m[2]
    if (delim === '`' && key.includes('${')) {
      warnings.push({ type: 'dynamic_template_literal', line: lineOf(content, m.index), snippet: content.slice(m.index, m.index + 100) })
    }
  }

  // Dynamic (non-literal) calls — cannot extract safely
  I18N_DYNAMIC_CALL_RE.lastIndex = 0
  let d
  while ((d = I18N_DYNAMIC_CALL_RE.exec(content)) !== null) {
    warnings.push({ type: 'dynamic_call', line: lineOf(content, d.index), snippet: content.slice(d.index, d.index + 100) })
  }

  // Alias bare calls from destructured local names, e.g. translate('key') for { t: translate }
  const buildNameAlt = (set) => Array.from(set).map((n) => n.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|')
  if (destructuredLocalT.size > 0) {
    const alt = buildNameAlt(destructuredLocalT)
    const ALIAS_T_CALL_RE = new RegExp(`\\b(?:${alt})\\(\\s*(["'`])([^"'`]+)\\1\\s*(?:,\\s*(["'`])([\\s\\S]*?)\\3\\s*)?\\)`, 'g')
    const ALIAS_T_DYNAMIC_RE = new RegExp(`\\b(?:${alt})\\(\\s*(?!["'`]))`, 'g')
    let am
    while ((am = ALIAS_T_CALL_RE.exec(content)) !== null) {
      const key = am[2]
      const defaultStr = typeof am[4] === 'string' ? am[4] : ''
      entries.push({ key, defaultStr })
      const delim = am[1]
      if (delim === '`' && key.includes('${')) {
        warnings.push({ type: 'dynamic_template_literal', line: lineOf(content, am.index), snippet: content.slice(am.index, am.index + 100) })
      }
    }
    let ad
    while ((ad = ALIAS_T_DYNAMIC_RE.exec(content)) !== null) {
      warnings.push({ type: 'dynamic_call', line: lineOf(content, ad.index), snippet: content.slice(ad.index, ad.index + 100) })
    }
  }
  if (destructuredLocalTc.size > 0) {
    const alt = buildNameAlt(destructuredLocalTc)
    const ALIAS_TC_CALL_RE = new RegExp(`\\b(?:${alt})\\(\\s*(["'`])([^"'`]+)\\1\\s*(?:,\\s*(["'`])([\\s\\S]*?)\\3\\s*)?\\)`, 'g')
    const ALIAS_TC_DYNAMIC_RE = new RegExp(`\\b(?:${alt})\\(\\s*(?!["'`]))`, 'g')
    let am
    while ((am = ALIAS_TC_CALL_RE.exec(content)) !== null) {
      const key = am[2]
      const defaultStr = typeof am[4] === 'string' ? am[4] : ''
      entries.push({ key, defaultStr })
      const delim = am[1]
      if (delim === '`' && key.includes('${')) {
        warnings.push({ type: 'dynamic_template_literal', line: lineOf(content, am.index), snippet: content.slice(am.index, am.index + 100) })
      }
    }
    let ad
    while ((ad = ALIAS_TC_DYNAMIC_RE.exec(content)) !== null) {
      warnings.push({ type: 'dynamic_call', line: lineOf(content, ad.index), snippet: content.slice(ad.index, ad.index + 100) })
    }
  }

  // i18nVar.t()/tc() calls using discovered variable names
  if (i18nVarNames.size > 0) {
    const namesAlt = Array.from(i18nVarNames).map((n) => n.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|')
    const VAR_CALL_RE = new RegExp(`\\b(?:${namesAlt})\\.(?:t|tc)\\(\\s*(["'`])([^"'`]+)\\1\\s*(?:,\\s*(["'`])([\\s\\S]*?)\\3\\s*)?\\)`, 'g')
    const VAR_DYNAMIC_RE = new RegExp(`\\b(?:${namesAlt})\\.(?:t|tc)\\(\\s*(?!["'`]))`, 'g')

    VAR_CALL_RE.lastIndex = 0
    let vc
    while ((vc = VAR_CALL_RE.exec(content)) !== null) {
      const key = vc[2]
      const defaultStr = typeof vc[4] === 'string' ? vc[4] : ''
      entries.push({ key, defaultStr })
      const delim = vc[1]
      if (delim === '`' && key.includes('${')) {
        warnings.push({ type: 'dynamic_template_literal', line: lineOf(content, vc.index), snippet: content.slice(vc.index, vc.index + 100) })
      }
    }

    VAR_DYNAMIC_RE.lastIndex = 0
    let vd
    while ((vd = VAR_DYNAMIC_RE.exec(content)) !== null) {
      warnings.push({ type: 'dynamic_call', line: lineOf(content, vd.index), snippet: content.slice(vd.index, vd.index + 100) })
    }
  }

  // v-t directive (simple)
  V_T_SIMPLE_RE.lastIndex = 0
  let s
  while ((s = V_T_SIMPLE_RE.exec(content)) !== null) {
    const key = s[2]
    entries.push({ key, defaultStr: '' })
  }

  // v-t directive (object form)
  V_T_OBJ_RE.lastIndex = 0
  let o
  while ((o = V_T_OBJ_RE.exec(content)) !== null) {
    const key = o[2]
    entries.push({ key, defaultStr: '' })
  }

  return { entries, warnings }
}

async function extract() {
  const catalog = new Map() // key -> { default: string, files: Set<string>, count: number }
  const warningsAll = []

  for await (const file of walk(srcDir)) {
    const content = await fs.readFile(file, 'utf8')
    const { entries, warnings } = extractFromContent(content)
    const relFile = path.relative(projectRoot, file)

    if (warnings && warnings.length) {
      for (const w of warnings) warningsAll.push({ file: relFile, ...w })
    }

    if (!entries.length) continue

    for (const { key, defaultStr } of entries) {
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
  await fs.writeFile(path.join(outDir, 'messages.warnings.json'), pretty(warningsAll), 'utf8')

  return { written: 4, keys: keys.length, outDir, warnings: warningsAll.length }
}

extract()
  .then((res) => {
    // eslint-disable-next-line no-console
    console.log(`` +
      `[i18n-extract] Wrote ${res.written} files to ${res.outDir} ` +
      `(${res.keys} keys, ${res.warnings} warnings).`)
  })
  .catch((err) => {
    // eslint-disable-next-line no-console
    console.error('[i18n-extract] Failed:', err)
    process.exitCode = 1
  })
