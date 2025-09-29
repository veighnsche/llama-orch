#!/usr/bin/env node
/*
  i18n-extract-ast.mjs
  AST-based extractor for Vue/TS/JS i18n usage. Replaces the regex-heavy extractor.

  - Finds calls to t/$t/tc/$tc, including aliases destructured from useI18n()
  - Handles i18nVar.t()/i18nVar.tc() when the var comes from useI18n()
  - Extracts default strings from the 2nd argument when it is a string literal
  - Flags dynamic usages (non-literal keys or template literals with expressions)
  - Extracts v-t directive usages from Vue templates (simple/object forms) via small, targeted regex

  Outputs under src/i18n/extracted/:
    - messages.catalog.json
    - messages.keys.json
    - messages.base.json
    - messages.warnings.json

  Usage:
    node scripts/i18n-extract-ast.mjs
*/

import { promises as fs } from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const ts = require('typescript')

const projectRoot = process.cwd()
const srcDir = path.join(projectRoot, 'src')
const outDir = path.join(projectRoot, 'src', 'i18n', 'extracted')

const SKIP_DIRS = new Set([
  'node_modules',
  'dist',
  '.git',
  '.cache',
  '.storybook',
  path.join('src', 'i18n', 'extracted'),
])

const EXTENSIONS = new Set(['.vue', '.ts', '.js', '.tsx', '.jsx'])

function lineOf(text, index) {
  let line = 1
  for (let i = 0; i < index && i < text.length; i++) if (text[i] === '\n') line++
  return line
}

async function* walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      const rel = path.relative(projectRoot, full)
      const first = rel.split(path.sep)[0]
      if (SKIP_DIRS.has(entry.name) || SKIP_DIRS.has(rel) || SKIP_DIRS.has(first)) continue
      yield* walk(full)
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name)
      if (EXTENSIONS.has(ext)) yield full
    }
  }
}

function extractScriptBlocksFromVue(content) {
  // Very lightweight SFC parsing: extract all <script ...>...</script> blocks
  // including <script setup> with any lang.
  const blocks = []
  const re = /<script\b[^>]*>([\s\S]*?)<\/script>/gi
  let m
  while ((m = re.exec(content)) !== null) {
    const code = m[1] || ''
    blocks.push(code)
  }
  return blocks
}

function extractTemplateVtKeys(content) {
  const keys = []
  // v-t="'path.to.key'" or v-t="\"path\""
  const simple = /v-t\s*=\s*("|')([^"'`]+)\1/g
  let s
  while ((s = simple.exec(content)) !== null) {
    const raw = s[2]
    // If raw looks like an object, skip here; it's handled by object form regex below.
    if (/^\s*\{/.test(raw)) continue
    keys.push(raw)
  }
  // v-t="{ path: 'key.path', ... }"
  const obj = /v-t\s*=\s*\{[^}]*?path\s*:\s*("|')([^"']+)\1[^}]*?\}/g
  let o
  while ((o = obj.exec(content)) !== null) {
    keys.push(o[2])
  }
  return keys
}

function isIdentifier(node, name) {
  return node && ts.isIdentifier(node) && node.escapedText === name
}

function isPropertyAccess(node, propName) {
  return node && ts.isPropertyAccessExpression(node) && node.name?.escapedText === propName
}

function getScriptKindForFile(file) {
  const ext = path.extname(file)
  switch (ext) {
    case '.ts': return ts.ScriptKind.TS
    case '.tsx': return ts.ScriptKind.TSX
    case '.jsx': return ts.ScriptKind.JSX
    case '.js': return ts.ScriptKind.JS
    default: return ts.ScriptKind.TS
  }
}

function parseTs(content, fileName) {
  const kind = getScriptKindForFile(fileName)
  return ts.createSourceFile(fileName, content, ts.ScriptTarget.Latest, true, kind)
}

function collectI18nBindings(sourceFile) {
  // Discover variable names from useI18n() and destructured aliases for t/tc
  const i18nVarNames = new Set()
  const localTNames = new Set()
  const localTcNames = new Set()

  function visit(node) {
    // const i18n = useI18n()
    if (ts.isVariableDeclaration(node) && node.initializer && ts.isCallExpression(node.initializer)) {
      const callee = node.initializer.expression
      if (isIdentifier(callee, 'useI18n')) {
        if (ts.isIdentifier(node.name)) {
          i18nVarNames.add(node.name.escapedText)
        } else if (ts.isObjectBindingPattern(node.name)) {
          for (const el of node.name.elements) {
            const prop = el.propertyName || el.name
            const local = el.name
            const propName = ts.isIdentifier(prop) ? prop.escapedText : undefined
            const localName = ts.isIdentifier(local) ? local.escapedText : undefined
            if (propName === 't' && localName) localTNames.add(localName)
            if (propName === 'tc' && localName) localTcNames.add(localName)
            if (!propName && localName) {
              // const { t, tc } = useI18n()
              if (localName === 't') localTNames.add('t')
              if (localName === 'tc') localTcNames.add('tc')
            }
          }
        }
      }
    }
    ts.forEachChild(node, visit)
  }

  visit(sourceFile)
  return { i18nVarNames, localTNames, localTcNames }
}

function literalText(node) {
  if (!node) return null
  if (ts.isStringLiteral(node)) return node.text
  if (ts.isNoSubstitutionTemplateLiteral(node)) return node.text
  return null
}

function isTemplateWithExpr(node) {
  return ts.isTemplateExpression(node)
}

function extractCallsFromAst(sourceFile, bindings, content, relFile) {
  const entries = []
  const warnings = []

  function addEntry(key, def) {
    entries.push({ key, defaultStr: def || '' })
  }
  function addWarn(type, node) {
    const pos = node.getStart()
    warnings.push({ type, line: lineOf(content, pos), file: relFile })
  }

  function visit(node) {
    if (ts.isCallExpression(node)) {
      const callee = node.expression
      let kind = null // 't' | 'tc' | '$t' | '$tc'

      // Identifier callee
      if (ts.isIdentifier(callee)) {
        const name = callee.escapedText
        if (name === 't' && bindings.localTNames.has('t')) kind = 't'
        else if (name === 'tc' && (bindings.localTcNames.has('tc') || bindings.localTNames.has('t'))) kind = 'tc'
        else if (name === '$t' || name === '$tc') kind = String(name)
        else if (bindings.localTNames.has(name)) kind = 't'
        else if (bindings.localTcNames.has(name)) kind = 'tc'
      }

      // i18nVar.t() / i18nVar.tc()
      if (!kind && ts.isPropertyAccessExpression(callee)) {
        const obj = callee.expression
        const prop = callee.name?.escapedText
        if ((prop === 't' || prop === 'tc') && ts.isIdentifier(obj) && bindings.i18nVarNames.has(obj.escapedText)) {
          kind = prop
        }
      }

      if (kind === 't' || kind === 'tc' || kind === '$t' || kind === '$tc') {
        const [arg0, arg1] = node.arguments
        const keyLit = literalText(arg0)
        if (!keyLit) {
          if (isTemplateWithExpr(arg0)) addWarn('dynamic_template_literal', node)
          else addWarn('dynamic_call', node)
        } else {
          let def = literalText(arg1) || ''
          addEntry(keyLit, def)
        }
      }
    }
    ts.forEachChild(node, visit)
  }

  visit(sourceFile)
  return { entries, warnings }
}

async function extractFile(file) {
  const content = await fs.readFile(file, 'utf8')
  const relFile = path.relative(projectRoot, file)
  const ext = path.extname(file)

  const entries = []
  const warnings = []

  if (ext === '.vue') {
    // Template: v-t directives
    for (const k of extractTemplateVtKeys(content)) entries.push({ key: k, defaultStr: '' })

    // Scripts inside SFC
    const scripts = extractScriptBlocksFromVue(content)
    for (const code of scripts) {
      const sf = parseTs(code, file + '.vue.ts')
      const bindings = collectI18nBindings(sf)
      const { entries: e, warnings: w } = extractCallsFromAst(sf, bindings, code, relFile)
      entries.push(...e)
      warnings.push(...w)
    }
  } else {
    const sf = parseTs(content, file)
    const bindings = collectI18nBindings(sf)
    const { entries: e, warnings: w } = extractCallsFromAst(sf, bindings, content, relFile)
    entries.push(...e)
    warnings.push(...w)
  }

  return { entries, warnings }
}

async function extractAll() {
  const catalog = new Map() // key -> { default: string, files: Set<string>, count: number }
  const warningsAll = []

  for await (const file of walk(srcDir)) {
    const { entries, warnings } = await extractFile(file)
    if (warnings?.length) warningsAll.push(...warnings)
    if (!entries?.length) continue

    const relFile = path.relative(projectRoot, file)
    for (const { key, defaultStr } of entries) {
      if (!catalog.has(key)) {
        catalog.set(key, { default: defaultStr || '', files: new Set([relFile]), count: 1 })
      } else {
        const item = catalog.get(key)
        item.count += 1
        item.files.add(relFile)
        if (!item.default && defaultStr) item.default = defaultStr
      }
    }
  }

  await fs.mkdir(outDir, { recursive: true })

  const keys = Array.from(catalog.keys()).sort()
  const base = {}
  const detailed = {}
  for (const key of keys) {
    const item = catalog.get(key)
    base[key] = item.default || ''
    detailed[key] = { default: item.default || '', files: Array.from(item.files).sort(), count: item.count }
  }

  const pretty = (obj) => JSON.stringify(obj, null, 2) + '\n'
  await fs.writeFile(path.join(outDir, 'messages.keys.json'), pretty(keys), 'utf8')
  await fs.writeFile(path.join(outDir, 'messages.base.json'), pretty(base), 'utf8')
  await fs.writeFile(path.join(outDir, 'messages.catalog.json'), pretty(detailed), 'utf8')
  await fs.writeFile(path.join(outDir, 'messages.warnings.json'), pretty(warningsAll), 'utf8')

  return { written: 4, keys: keys.length, outDir, warnings: warningsAll.length }
}

extractAll()
  .then((res) => {
    console.log(`[i18n-extract-ast] Wrote ${res.written} files to ${res.outDir} (${res.keys} keys, ${res.warnings} warnings).`)
  })
  .catch((err) => {
    console.error('[i18n-extract-ast] Failed:', err)
    process.exitCode = 1
  })
