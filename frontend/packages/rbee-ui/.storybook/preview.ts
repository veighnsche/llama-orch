import type { Preview, Decorator } from "@storybook/react";
import { useEffect } from "react";
import "../src/tokens/globals.css";

// Mock Next.js globals for Storybook
if (typeof window !== 'undefined') {
  (window as any).process = { env: {} };
}

// Theme decorator that applies dark class to document root
const withTheme: Decorator = (Story, context) => {
  const theme = context.globals.theme || 'light';
  
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
  }, [theme]);

  return Story();
};

const preview: Preview = {
  decorators: [withTheme],
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
  globalTypes: {
    theme: {
      description: 'Global theme for components',
      defaultValue: 'light',
      toolbar: {
        title: 'Theme',
        icon: 'circlehollow',
        items: [
          { value: 'light', icon: 'sun', title: 'Light mode' },
          { value: 'dark', icon: 'moon', title: 'Dark mode' },
        ],
        dynamicTitle: true,
      },
    },
  },
};

export default preview;
