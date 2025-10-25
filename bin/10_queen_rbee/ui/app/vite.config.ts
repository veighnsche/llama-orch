import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 7834,  // queen-rbee UI dev server
    strictPort: true,
  },
  plugins: [
    tailwindcss(),  // Official Tailwind v4 Vite plugin (must be first)
    wasm(),
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  optimizeDeps: {
    exclude: ['@rbee/sdk'],
  },
  build: {
    cssMinify: false, // Disable CSS minification to avoid lightningcss issues with Tailwind
  },
  define: {
    'process.env': {},  // Polyfill for libraries that check process.env
  },
})
