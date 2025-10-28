// TEAM-294: Shared Vite config for React + Tailwind apps

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/**
 * Create a Vite config for React + Tailwind apps
 * @param {import('vite').UserConfig} overrides - Optional config overrides
 * @returns {import('vite').UserConfig}
 */
export function createViteConfig(overrides = {}) {
  return defineConfig({
    plugins: [
      tailwindcss(), // Official Tailwind v4 Vite plugin (must be first)
      react({
        babel: {
          plugins: [['babel-plugin-react-compiler']],
        },
      }),
      ...(overrides.plugins || []),
    ],
    build: {
      cssMinify: false, // Disable CSS minification to avoid lightningcss issues with Tailwind
      ...overrides.build,
    },
    define: {
      'process.env': {}, // Polyfill for libraries that check process.env
      ...overrides.define,
    },
    ...overrides,
  })
}
