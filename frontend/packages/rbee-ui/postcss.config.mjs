import tailwindcss from '@tailwindcss/postcss'
import postcssNesting from 'postcss-nesting'

const config = {
	plugins: [
		tailwindcss,
		postcssNesting,
	],
}

export default config
