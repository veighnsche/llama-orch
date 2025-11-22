import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import topLevelAwait from 'vite-plugin-top-level-await'
import wasm from 'vite-plugin-wasm'

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 7845, // rbee-hive UI dev server
    strictPort: true,
    host: '0.0.0.0', // TEAM-378: Bind to all interfaces for remote access
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
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (
              id.includes('@rbee/queen-rbee-react') ||
              id.includes('@rbee/queen-rbee-sdk') ||
              id.includes('@rbee/rbee-hive-react') ||
              id.includes('@rbee/rbee-hive-sdk')
            ) {
              return 'sdk'
            }
            if (id.includes('react-router-dom')) {
              return 'router'
            }
            if (id.includes('react-dom') || id.includes('react')) {
              return 'react'
            }
            if (id.includes('zustand') || id.includes('@tanstack/react-query')) {
              return 'state'
            }
            if (
              id.includes('@rbee/ui') ||
              id.includes('lucide-react') ||
              id.includes('class-variance-authority')
            ) {
              return 'ui'
            }
            return 'vendor'
          }
        },
      },
    },
  },
  define: {
    'process.env': {}, // Polyfill for libraries that check process.env
  },
})
