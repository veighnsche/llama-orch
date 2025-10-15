import { withThemeByClassName } from '@storybook/addon-themes'
import type { Preview } from '@storybook/react'
// ✅ TAILWIND V4 + STORYBOOK: Import SOURCE CSS, not pre-built dist
// The @tailwindcss/vite plugin in main.ts handles JIT compilation
// This enables arbitrary values like translate-y-[100px] to work in Storybook
import '../src/tokens/globals.css'

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    backgrounds: {
      disable: true, // Disable default Storybook backgrounds
    },
  },
  decorators: [
    withThemeByClassName({
      themes: {
        light: '',
        dark: 'dark',
      },
      defaultTheme: 'light',
      // Apply the class to the html element (same as next-themes)
      parentSelector: 'html',
    }),
  ],
}

export default preview
