import type { StorybookConfig } from '@storybook/react-vite'
import tailwindcss from '@tailwindcss/vite'

const config: StorybookConfig = {
	stories: ['../src/**/*.stories.@(js|jsx|mjs|ts|tsx)'],
	addons: ['@storybook/addon-themes'],
	framework: {
		name: '@storybook/react-vite',
		options: {
			builder: {
				viteConfigPath: undefined,
			},
		},
	},
	core: {
		disableTelemetry: true,
		disableWhatsNewNotifications: true,
	},
	viteFinal: async (config) => {
		config.plugins = config.plugins || []
		config.plugins.push(tailwindcss())

		// Define process.env for browser
		config.define = {
			...config.define,
			'process.env': {},
			'process.env.NODE_ENV': JSON.stringify('development'),
		}

		// Resolve next-themes properly
		config.resolve = config.resolve || {}
		config.resolve.alias = {
			...config.resolve.alias,
			'next/link': require.resolve('./mocks/next-link.tsx'),
			'next/navigation': require.resolve('./mocks/next-navigation.tsx'),
			'next/image': require.resolve('./mocks/next-image.tsx'),
		}

		return config
	},
}
export default config