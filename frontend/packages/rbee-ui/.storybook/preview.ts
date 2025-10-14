import type { Preview } from '@storybook/react'
// For Storybook running WITHIN the UI package, import the built CSS directly
// External consumers (apps) import via '@rbee/ui/styles.css'
import '../dist/index.css'
// Import theme colors (CSS custom properties) for preview
import './preview-theme.css'

const preview: Preview = {
	parameters: {
		controls: {
			matchers: {
				color: /(background|color)$/i,
				date: /Date$/i,
			},
		},
	},
}

export default preview
