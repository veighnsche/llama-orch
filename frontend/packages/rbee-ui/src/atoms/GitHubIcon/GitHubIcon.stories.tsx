import type { Meta, StoryObj } from '@storybook/react';
import { GitHubIcon } from './GitHubIcon';

const meta = {
  title: 'Atoms/Icons/GitHubIcon',
  component: GitHubIcon,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The GitHubIcon component renders the GitHub logo as an SVG icon. It's a simple, accessible icon component that inherits color from its parent context.

## When to Use
- In navigation bars linking to GitHub repositories
- In footer social media links
- In developer-focused pages or sections
- Anywhere you need to represent GitHub or link to GitHub resources

## Variants
- **Default**: Standard size (16px / size-4)
- **Custom Size**: Any size via className
- **Custom Color**: Inherits currentColor, can be styled via text color utilities

## Examples
\`\`\`tsx
import { GitHubIcon } from '@rbee/ui/atoms/GitHubIcon'

// Default size
<GitHubIcon />

// Large size
<GitHubIcon className="w-8 h-8" />

// Custom color
<GitHubIcon className="text-blue-600" />

// In a link
<a href="https://github.com/your-repo" className="hover:text-gray-600">
  <GitHubIcon className="w-6 h-6" />
</a>
\`\`\`

## Accessibility
- Uses \`aria-hidden="true"\` as it's decorative
- Should be paired with visible text or aria-label on parent link
- Inherits color for proper contrast in light/dark modes
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    className: {
      control: 'text',
      description: 'Additional CSS classes for styling (size, color, etc.)',
      table: {
        type: { summary: 'string' },
        category: 'Appearance',
      },
    },
  },
} satisfies Meta<typeof GitHubIcon>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const Small: Story = {
  args: {
    className: 'w-3 h-3',
  },
};

export const Large: Story = {
  args: {
    className: 'w-12 h-12',
  },
};

export const ExtraLarge: Story = {
  args: {
    className: 'w-16 h-16',
  },
};

export const ColoredVariants: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
      <GitHubIcon className="w-8 h-8 text-gray-900 dark:text-gray-100" />
      <GitHubIcon className="w-8 h-8 text-blue-600" />
      <GitHubIcon className="w-8 h-8 text-green-600" />
      <GitHubIcon className="w-8 h-8 text-purple-600" />
      <GitHubIcon className="w-8 h-8 text-red-600" />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'GitHubIcon in various colors. The icon inherits currentColor, making it easy to style with Tailwind text utilities.',
      },
    },
  },
};

export const AllSizes: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
      <GitHubIcon className="w-3 h-3" />
      <GitHubIcon className="w-4 h-4" />
      <GitHubIcon className="w-6 h-6" />
      <GitHubIcon className="w-8 h-8" />
      <GitHubIcon className="w-12 h-12" />
      <GitHubIcon className="w-16 h-16" />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'All common sizes from extra small (12px) to extra large (64px).',
      },
    },
  },
};

export const InLink: Story = {
  render: () => (
    <a
      href="https://github.com"
      className="inline-flex items-center gap-2 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-gray-100 transition-colors"
    >
      <GitHubIcon className="w-5 h-5" />
      <span>View on GitHub</span>
    </a>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Common usage pattern: GitHubIcon inside a link with hover effects.',
      },
    },
  },
};
