import type { Meta, StoryObj } from '@storybook/react';
import { HeroSection } from './HeroSection';

const meta = {
  title: 'Organisms/HeroSection',
  component: HeroSection,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The HeroSection is the primary landing section of the rbee application. It features a compelling headline, value proposition, interactive terminal demo, and clear call-to-action buttons. The section uses a full-viewport height with animated elements and a honeycomb background pattern.

## Composition
This organism contains:
- **PulseBadge**: Animated badge showing open-source status
- **Headline**: Large, bold headline with primary color accent
- **Value Proposition**: Clear description of the product benefits
- **Feature Bullets**: Quick list of key benefits (checkmarks)
- **CTA Buttons**: Primary "Get Started" and secondary "View Docs" buttons
- **Trust Badges**: GitHub stars, API compatibility, cost indicators
- **TerminalWindow**: Interactive demo showing GPU orchestration
- **ProgressBar**: Visual GPU utilization indicators
- **FloatingKPICard**: Animated floating card with metrics
- **HoneycombPattern**: Background pattern for visual interest

## When to Use
- As the first section on the homepage
- To immediately communicate value proposition
- To provide interactive demo of the product
- To drive users to primary call-to-action

## Content Requirements
- **Headline**: Clear, benefit-focused headline (max 2 lines)
- **Subheadline**: Detailed value proposition (2-3 sentences)
- **Feature Bullets**: 3-5 key benefits
- **CTA Buttons**: Primary and secondary actions
- **Terminal Demo**: Realistic code example showing product in action
- **Trust Indicators**: Social proof elements (GitHub stars, etc.)

## Variants
- **Default**: Full hero with all elements
- **Mobile**: Responsive layout with stacked content
- **Reduced Motion**: Respects prefers-reduced-motion setting

## Examples
\`\`\`tsx
import { HeroSection } from '@rbee/ui/organisms/HeroSection'

// Simple usage - no props needed
<HeroSection />
\`\`\`

## Used In
- Home page (/)
- Primary landing page

## Related Components
- PulseBadge
- TerminalWindow
- ProgressBar
- FloatingKPICard
- HoneycombPattern

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **ARIA Labels**: Proper labels on all interactive elements
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Reduced Motion**: Respects prefers-reduced-motion media query
- **Focus States**: Visible focus indicators on all interactive elements
- **Live Regions**: Terminal output uses aria-live for screen readers
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof HeroSection>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  parameters: {
    docs: {
      description: {
        story: 'Default hero section with all elements. Use the theme toggle in the toolbar to switch between light and dark modes.',
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
        story: 'Mobile view with stacked layout. Content and terminal demo stack vertically for optimal mobile experience.',
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
        story: 'Tablet view showing responsive breakpoint behavior.',
      },
    },
  },
};

export const WithScrollIndicator: Story = {
  render: () => (
    <div>
      <HeroSection />
      <div style={{ padding: '4rem 2rem', textAlign: 'center', background: 'rgba(0,0,0,0.02)' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Next Section</h2>
        <p>Scroll up to see the hero section with its full-viewport height and animations.</p>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Hero section with additional content below to demonstrate full-viewport height and scroll behavior.',
      },
    },
  },
};
