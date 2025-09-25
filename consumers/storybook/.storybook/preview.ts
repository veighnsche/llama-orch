import type { Preview } from '@storybook/vue3-vite';
import '../styles/tokens.css';
import '@commercial/styles/blueprint.css';
import { setup } from '@storybook/vue3-vite';
import { createMemoryHistory, createRouter } from 'vue-router';

setup((app) => {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: '/', name: 'home', component: { template: '<div />' } }],
  });
  app.use(router);
});

const preview: Preview = {
  parameters: {
    controls: { matchers: { color: /(background|color)$/i, date: /Date$/i } },
    a11y: { test: 'todo' },
  },
};
export default preview;
