#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { watch } from 'node:fs'
import { resolve } from 'node:path'

const ROOT = resolve(process.cwd())
const WATCH_DIRS = [
	resolve(ROOT, 'src'), // Rust crate sources
	resolve(ROOT, 'scripts'), // build helpers
]
const WATCH_FILES = [
	resolve(ROOT, 'index.ts'), // TS entrypoint
	resolve(ROOT, 'Cargo.toml'), // crate manifest
]

let building = false
let pending = false
let timer = null

function runBuild() {
	if (building) {
		pending = true
		return
	}
	building = true
	const child = spawn(process.execPath, ['-e', 'process.exit(0)'], { stdio: 'ignore' })
	child.on('exit', () => {
		const b = spawn('pnpm', ['run', 'build'], { stdio: 'inherit' })
		b.on('exit', (code) => {
			building = false
			if (pending) {
				pending = false
				debounce()
			}
			if (code !== 0) {
				console.error('[utils:watch] build failed with code', code)
			} else {
				console.log('[utils:watch] build completed')
			}
		})
	})
}

function debounce() {
	if (timer) clearTimeout(timer)
	timer = setTimeout(runBuild, 150)
}

for (const dir of WATCH_DIRS) {
	try {
		watch(dir, { recursive: true }, (event, filename) => {
			if (!filename) return
			if (!/\.(rs|toml|mjs|js|ts|json)$/.test(filename)) return
			console.log('[utils:watch] change detected:', event, filename)
			debounce()
		})
		console.log('[utils:watch] watching', dir)
	} catch (e) {
		console.warn('[utils:watch] failed to watch', dir, e?.message || e)
	}
}

for (const file of WATCH_FILES) {
	try {
		watch(file, {}, (event) => {
			console.log('[utils:watch] change detected:', event, file)
			debounce()
		})
		console.log('[utils:watch] watching', file)
	} catch (e) {
		console.warn('[utils:watch] failed to watch', file, e?.message || e)
	}
}

// initial build
runBuild()
