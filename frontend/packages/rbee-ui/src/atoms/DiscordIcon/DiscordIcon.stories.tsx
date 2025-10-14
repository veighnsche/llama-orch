import type { Meta, StoryObj } from '@storybook/react';
import { DiscordIcon } from './DiscordIcon';

const meta = {
  title: 'Atoms/Icons/DiscordIcon',
  component: DiscordIcon,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The DiscordIcon component renders the Discord logo as an SVG icon. It's a simple, accessible icon component that inherits color from its parent context.

## When to Use
- In navigation bars linking to Discord servers
- In footer social media links
- In community or support sections
- Anywhere you need to represent Discord or link to Discord resources

## Variants
- **Default**: Standard size (20px / size-5)
- **Custom Size**: Any size via className
- **Custom Color**: Inherits currentColor, can be styled via text color utilities

## Examples
\`\`\`tsx
import { DiscordIcon } from '@rbee/ui/atoms/DiscordIcon'

// Default size
<DiscordIcon />

// Large size
<DiscordIcon className="w-8 h-8" />

// Custom color (Discord brand color)
<DiscordIcon className="text-[#5865F2]" />

// In a link
<a href="https://discord.gg/your-server" className="hover:text-[#5865F2]">
  <DiscordIcon className="w-6 h-6" />
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
} satisfies Meta<typeof DiscordIcon>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const Small: Story = {
  args: {
    className: 'w-4 h-4',
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

export const BrandColor: Story = {
  args: {
    className: 'w-8 h-8 text-[#5865F2]',
  },
  parameters: {
    docs: {
      description: {
        story: 'Discord icon in the official Discord brand color (#5865F2).',
      },
    },
  },
};

export const ColoredVariants: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
      <DiscordIcon className="w-8 h-8 text-gray-900 dark:text-gray-100" />
      <DiscordIcon className="w-8 h-8 text-[#5865F2]" />
      <DiscordIcon className="w-8 h-8 text-blue-600" />
      <DiscordIcon className="w-8 h-8 text-purple-600" />
      <DiscordIcon className="w-8 h-8 text-indigo-600" />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'DiscordIcon in various colors. The icon inherits currentColor, making it easy to style with Tailwind text utilities.',
      },
    },
  },
};

export const AllSizes: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
      <DiscordIcon className="w-3 h-3" />
      <DiscordIcon className="w-4 h-4" />
      <DiscordIcon className="w-6 h-6" />
      <DiscordIcon className="w-8 h-8" />
      <DiscordIcon className="w-12 h-12" />
      <DiscordIcon className="w-16 h-16" />
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
      href="https://discord.gg/example"
      className="inline-flex items-center gap-2 text-gray-700 hover:text-[#5865F2] dark:text-gray-300 dark:hover:text-[#5865F2] transition-colors"
    >
      <DiscordIcon className="w-5 h-5" />
      <span>Join our Discord</span>
    </a>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Common usage pattern: DiscordIcon inside a link with hover effects to Discord brand color.',
      },
    },
  },
};

export const SocialMediaRow: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
      <a href="#" className="text-gray-600 hover:text-[#5865F2] transition-colors">
        <DiscordIcon className="w-6 h-6" />
      </a>
      <a href="#" className="text-gray-600 hover:text-gray-900 transition-colors">
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
        </svg>
      </a>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Example of DiscordIcon used alongside other social media icons in a footer or header.',
      },
    },
  },
};
