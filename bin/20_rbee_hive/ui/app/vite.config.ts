import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 7836, // rbee-hive UI dev server
    strictPort: true,
  },
  plugins: [
    tailwindcss(), // TEAM-374: Official Tailwind v4 Vite plugin (must be first)
    wasm(), // TEAM-374: Support WASM imports
    topLevelAwait(), // TEAM-374: Support top-level await for WASM
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  optimizeDeps: {
    exclude: ['@rbee/rbee-hive-sdk'], // TEAM-374: Don't pre-bundle WASM SDK
  },
  build: {
    cssMinify: false, // Disable CSS minification to avoid lightningcss issues with Tailwind
  },
  define: {
    'process.env': {}, // Polyfill for libraries that check process.env
  },
})
