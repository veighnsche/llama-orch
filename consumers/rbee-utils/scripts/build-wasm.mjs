#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process'
import { promises as fsp } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')

const TARGET = 'wasm32-wasip1-threads'
const CRATE = 'llama-orch-utils'
const OUT_NAME = 'llama_orch_utils.wasm'
const OUT_DIR = resolve(root, 'dist')
const OUT_PATH = resolve(OUT_DIR, OUT_NAME)
const BUILD_ARTIFACT = resolve(root, `../../target/${TARGET}/release/${OUT_NAME}`)

function run(cmd, args, opts = {}) {
	return new Promise((resolvePromise, reject) => {
		const child = spawn(cmd, args, { stdio: 'inherit', cwd: root, ...opts })
		child.on('exit', (code) => {
			if (code === 0) resolvePromise()
			else reject(new Error(`${cmd} exited with ${code}`))
		})
	})
}

;(async () => {
	try {
		// Prefer rustup-run cargo if available to honor repo-local toolchain/targets
		let cmd = 'cargo'
		let args = ['build', '--release', '--target', TARGET, '-p', CRATE]
		try {
			let haveRustup = false
			let rustupCmd = 'rustup'
			let probe = spawnSync(rustupCmd, ['--version'], { stdio: 'ignore' })
			if (probe.status !== 0) {
				// try ~/.cargo/bin/rustup
				const home = process.env.HOME || process.env.USERPROFILE
				if (home) {
					const candidate = `${home}/.cargo/bin/rustup`
					probe = spawnSync(candidate, ['--version'], { stdio: 'ignore' })
					if (probe.status === 0) {
						rustupCmd = candidate
						haveRustup = true
					}
				}
			} else {
				haveRustup = true
			}
			if (haveRustup) {
				cmd = rustupCmd
				args = ['run', 'stable', 'cargo', ...args]
			}
		} catch {}

		await run(cmd, args)
		await fsp.mkdir(OUT_DIR, { recursive: true })
		await fsp.copyFile(BUILD_ARTIFACT, OUT_PATH)
		console.log(`Copied ${BUILD_ARTIFACT} -> ${OUT_PATH}`)
	} catch (err) {
		console.error('build-wasm failed:', err.message || err)
		process.exit(1)
	}
})()
