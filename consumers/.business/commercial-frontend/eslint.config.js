import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import vue from 'eslint-plugin-vue'
import vueParser from 'vue-eslint-parser'

export default [
    // Ignore build artifacts and externals
    {
        ignores: ['dist/', 'node_modules/', 'coverage/', '.histoire/', '.vite/', 'public/'],
    },

    // Base JS rules
    js.configs.recommended,

    // Vue single-file component rules
    ...vue.configs['flat/recommended'],

    // TypeScript rules
    ...tseslint.configs.recommended,

    // Project-specific tweaks
    {
        files: ['**/*.ts', '**/*.vue'],
        languageOptions: {
            parser: vueParser,
            parserOptions: {
                ecmaVersion: 'latest',
                sourceType: 'module',
                extraFileExtensions: ['.vue'],
                // Use TS parser for <script lang="ts"> blocks
                parser: tseslint.parser,
            },
        },
        rules: {
            // Allow single word component names like Button/Badge
            'vue/multi-word-component-names': 'off',
        },
    },
]
