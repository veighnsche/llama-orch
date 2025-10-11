import { defineConfig } from 'histoire'
import { HstVue } from '@histoire/plugin-vue'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [HstVue()],
  vite: {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./', import.meta.url)),
      },
    },
  },
  setupFile: './histoire.setup.ts',
  theme: {
    // Let Histoire manage a toolbar toggle: 'light' | 'dark' | 'auto' (system)
    defaultColorScheme: 'auto',
    // Apply this class to the story preview when in dark mode
    darkClass: 'dark',
    // Persist user selection across reloads
    storeColorScheme: true,
  },
})
