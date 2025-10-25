// TEAM-294: Vite config with Tailwind + React
// Uses shared dependencies from @repo/vite-config
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 5173,  // Dedicated port for rbee-keeper UI
    strictPort: true,  // Fail if port is in use instead of trying another
  },
  optimizeDeps: {
    force: true,  // Force dependency pre-bundling on server start
  },
  plugins: [
    tailwindcss(),  // Official Tailwind v4 Vite plugin (must be first)
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  build: {
    cssMinify: false, // Disable CSS minification to avoid lightningcss issues with Tailwind
  },
  define: {
    'process.env': {},  // Polyfill for libraries that check process.env
  },
});
