import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

import { cloudflare } from '@cloudflare/vite-plugin'

// https://vite.dev/config/
export default defineConfig({
    plugins: [vue(), vueDevTools(), cloudflare()],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url)),
            // Alias the Badge component to the Storybook package implementation
            '@ui/badge': 'orchyra-storybook/stories/badge.vue',
            // Backwards-compat: old direct path now resolves to Storybook Badge
            '@/components/ui/Badge.vue': 'orchyra-storybook/stories/badge.vue',
        },
    },
})
