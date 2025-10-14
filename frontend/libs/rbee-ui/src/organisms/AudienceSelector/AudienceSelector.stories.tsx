import type { Meta, StoryObj } from '@storybook/react';
import { AudienceSelector } from './AudienceSelector';

const meta = {
  title: 'Organisms/AudienceSelector',
  component: AudienceSelector,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The AudienceSelector presents three distinct user paths (Developers, GPU Owners, Enterprise) in an interactive card grid. Each card highlights specific benefits, features, and calls-to-action tailored to that audience segment.

## Composition
This organism contains:
- **Header**: Kicker, title, and description
- **AudienceCard Grid**: Three cards with equal heights
  - **Developers Card**: For building on own hardware
  - **GPU Owners Card**: For monetizing idle GPUs
  - **Enterprise Card**: For compliance and security
- **Bottom Helper Links**: "Compare paths" and "Talk to us"
- **Radial Gradient**: Subtle background effect
- **Decorative Hairline**: Top border with gradient

## When to Use
- On the homepage to segment audiences
- After the hero or "What is rbee" section
- To guide users to relevant content
- As a navigation hub for different user types

## Content Requirements
- **Header**: Clear question or prompt
- **Three Cards**: Each with icon, category, title, description, features, CTA
- **Consistent Structure**: All cards follow same pattern
- **Helper Links**: Additional navigation options

## Variants
- **Default**: Three-column grid on desktop
- **Mobile**: Single column stacked layout
- **Tablet**: Two-column grid

## Examples
\`\`\`tsx
import { AudienceSelector } from '@rbee/ui/organisms/AudienceSelector'

// Simple usage - no props needed
<AudienceSelector />
\`\`\`

## Used In
- Home page (/)

## Related Components
- AudienceCard
- Badge
- IconBox

## Accessibility
- **Keyboard Navigation**: All cards and links are keyboard accessible
- **Focus States**: Visible focus indicators with ring
- **Semantic HTML**: Proper heading hierarchy and landmarks
- **ARIA Labels**: Grid labeled as "Audience options"
- **Hover Effects**: Transform animations respect prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof AudienceSelector>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  parameters: {
    docs: {
      description: {
        story: 'Default audience selector with three cards. Use the theme toggle in the toolbar to switch between light and dark modes.',
      },
    },
  },
};

export const MobileView: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
    docs: {
      description: {
        story: 'Mobile view with stacked single-column layout.',
      },
    },
  },
};

export const TabletView: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
    docs: {
      description: {
        story: 'Tablet view with two-column grid layout.',
      },
    },
  },
};
