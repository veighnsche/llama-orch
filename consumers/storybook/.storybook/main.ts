// .storybook/main.ts (or .js with the same content)
import type { StorybookConfig } from '@storybook/vue3-vite';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath } from 'node:url';

const config: StorybookConfig = {
  stories: [
    '../stories/**/*.mdx',
    '../stories/**/*.stories.@(js|jsx|mjs|ts|tsx)',
  ],
  addons: [
    '@chromatic-com/storybook',
    '@storybook/addon-docs',
    '@storybook/addon-onboarding',
    '@storybook/addon-a11y',
    '@storybook/addon-vitest',
  ],
  framework: { name: '@storybook/vue3-vite', options: {} },
  viteFinal: async (cfg) => {
    // Ensure the Vue SFC plugin is present for .vue files
    cfg.plugins = [...(cfg.plugins ?? []), vue()];

    // Alias the commercial app so stories can import components via @commercial / @
    const commercialSrc = fileURLToPath(new URL('../../.business/commercial-frontend/src', import.meta.url));
    cfg.resolve = cfg.resolve ?? {};
    cfg.resolve.alias = {
      ...(cfg.resolve.alias || {}),
      '@commercial': commercialSrc,
      '@': commercialSrc,
    };
    return cfg;
  },
};
export default config;
