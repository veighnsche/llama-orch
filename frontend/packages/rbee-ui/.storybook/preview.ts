import type { Preview } from '@storybook/react'
import { withThemeByClassName } from '@storybook/addon-themes'
// For Storybook running WITHIN the UI package, import the built CSS directly
// External consumers (apps) import via '@rbee/ui/styles.css'
// This CSS includes all design tokens (no separate preview-theme.css needed)
import '../dist/index.css'

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
