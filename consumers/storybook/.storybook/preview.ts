import type { Preview } from '@storybook/vue3-vite';
import '../styles/tokens.css';
import '@commercial/styles/blueprint.css';

const preview: Preview = {
  parameters: {
    controls: { matchers: { color: /(background|color)$/i, date: /Date$/i } },
    a11y: { test: 'todo' },
  },
};
export default preview;
