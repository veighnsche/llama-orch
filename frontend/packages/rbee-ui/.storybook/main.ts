import type { StorybookConfig } from '@storybook/react-vite'
import tailwindcss from '@tailwindcss/vite'

const config: StorybookConfig = {
	stories: ['../src/**/*.mdx', '../src/**/*.stories.@(js|jsx|mjs|ts|tsx)'],
	addons: [
		'@storybook/addon-essentials',
		'@storybook/addon-interactions',
		'@storybook/addon-links',
		'@chromatic-com/storybook',
	],
	framework: {
		name: '@storybook/react-vite',
		options: {},
	},
	viteFinal: async (config) => {
		config.plugins = config.plugins || []
		config.plugins.push(tailwindcss())

		// Ensure next-themes is properly resolved
		config.resolve = config.resolve || {}
		config.resolve.alias = {
			...config.resolve.alias,
			'next-themes': require.resolve('next-themes'),
		}

		return config
	},
}
export default config
