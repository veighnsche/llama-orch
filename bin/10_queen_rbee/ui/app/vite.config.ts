import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import topLevelAwait from 'vite-plugin-top-level-await'
import wasm from 'vite-plugin-wasm'

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 7844, // queen-rbee UI dev server
    strictPort: true,
  },
  plugins: [
    tailwindcss(), // Official Tailwind v4 Vite plugin (must be first)
    wasm(),
    topLevelAwait(),
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  optimizeDeps: {
    exclude: ['@rbee/queen-rbee-sdk'], // TEAM-375: Exclude WASM package from pre-bundling
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
