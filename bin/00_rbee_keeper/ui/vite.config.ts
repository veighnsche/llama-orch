// TEAM-294: Vite config with Tailwind + React
// Uses shared dependencies from @repo/vite-config
// TEAM-296: Added path alias for generated Tauri bindings

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: 'localhost', // TEAM-XXX: mac compat - MUST use localhost (not 127.0.0.1) for Tauri v2 remote permissions
    port: 7843, // Dedicated port for rbee-keeper UI
    strictPort: true, // Fail if port is in use instead of trying another
  },
  optimizeDeps: {
    force: true, // Force dependency pre-bundling on server start
  },
  plugins: [
    tailwindcss(), // Official Tailwind v4 Vite plugin (must be first)
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
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
