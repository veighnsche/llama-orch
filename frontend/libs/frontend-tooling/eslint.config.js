import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import vue from 'eslint-plugin-vue'
import vueParser from 'vue-eslint-parser'

// Shared ESLint flat config for Vue + TS projects
export default [
  {
    ignores: ['dist/', 'node_modules/', 'coverage/', '.histoire/', '.vite/', 'public/'],
  },
  js.configs.recommended,
  ...vue.configs['flat/recommended'],
  ...tseslint.configs.recommended,
  {
    files: ['**/*.ts', '**/*.vue'],
    languageOptions: {
      parser: vueParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        extraFileExtensions: ['.vue'],
        parser: tseslint.parser,
      },
    },
    rules: {
      // Allow single word component names like Button/Badge
      'vue/multi-word-component-names': 'off',
    },
  },
]
