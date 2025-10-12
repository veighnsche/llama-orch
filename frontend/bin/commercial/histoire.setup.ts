// TEAM-DX-006: Fixed CSS imports for Histoire to use storybook's compiled CSS
// Bug: Histoire was trying to import main.css which uses @tailwindcss/vite plugin
//      This caused missing CSS classes from workspace dependencies
// Fix: Created histoire.css that imports rbee-storybook/styles/tokens.css (pre-compiled)
// Expected: All Tailwind classes from storybook (including cursor-pointer) now available
import { defineSetupVue3 } from '@histoire/plugin-vue'
import './app/assets/css/histoire.css'

export const setupVue3 = defineSetupVue3(({ app }) => {
  // Add any global plugins or configurations here
})
