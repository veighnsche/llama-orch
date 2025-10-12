import { defineConfig } from 'histoire'
import { HstVue } from '@histoire/plugin-vue'

export default defineConfig({
  plugins: [HstVue()],
  setupFile: './histoire.setup.ts',
  tree: {
    groups: [
      {
        id: 'organisms',
        title: 'Organisms',
      },
      {
        id: 'templates',
        title: 'Templates',
      },
    ],
  },
  theme: {
    title: 'rbee Commercial - Component Stories',
    logo: {
      square: '🐝',
      light: '🐝 rbee',
      dark: '🐝 rbee',
    },
  },
  vite: {
    server: {
      port: 6007,
    },
  },
})
