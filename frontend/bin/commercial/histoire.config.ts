// TEAM-DX-006: Fixed Histoire configuration for workspace dependency CSS
// Bug: Tailwind CSS classes from rbee-storybook (like cursor-pointer) were not available in commercial Histoire
// Fix: Simplified config to import pre-compiled CSS from storybook instead of trying to scan workspace deps
// Expected: Histoire now uses storybook's compiled CSS, no need for PostCSS plugin or workspace scanning
import { defineConfig } from 'histoire'
import { HstVue } from '@histoire/plugin-vue'
import vue from '@vitejs/plugin-vue'

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
  },
  vite: {
    // TEAM-DX-006: Added Vue plugin for proper .vue file handling
    plugins: [vue()],
    server: {
      port: 6007,
    },
    optimizeDeps: {
      // TEAM-DX-006: Exclude rbee-storybook from pre-bundling to ensure fresh CSS imports
      exclude: ['rbee-storybook'],
    },
  },
})
