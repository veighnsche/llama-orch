import { TemplateContainer } from '@rbee/ui/molecules'
import { audienceSelectorContainerProps, audienceSelectorProps } from '@rbee/ui/pages/HomePage'
import type { Meta, StoryObj } from '@storybook/react'
import { AudienceSelector } from './AudienceSelector'

const meta = {
  title: 'Templates/AudienceSelector',
  component: AudienceSelector,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
## Overview
The AudienceSelector template presents three distinct user paths (Developers, GPU Owners, Enterprise) in an interactive card grid. Each card highlights specific benefits, features, and calls-to-action tailored to that audience segment.

## Composition
This template contains:
- **AudienceCard Grid**: Three cards with equal heights
  - **Developers Card**: For building on own hardware
  - **GPU Owners Card**: For monetizing idle GPUs
  - **Enterprise Card**: For compliance and security
- **Bottom Helper Links**: "Compare paths" and "Talk to us"
- **Radial Gradient**: Subtle background effect (optional)

## When to Use
- On the homepage to segment audiences
- After the hero or "What is rbee" section
- To guide users to relevant content
- As a navigation hub for different user types

## Content Requirements
- **Three Cards**: Each with icon, category, title, description, features, CTA
- **Consistent Structure**: All cards follow same pattern
- **Helper Links**: Additional navigation options (optional)

## Variants
- **Default**: Three-column grid on desktop
- **Mobile**: Single column stacked layout
- **Tablet**: Two-column grid

## Examples
\`\`\`tsx
import { AudienceSelector } from '@rbee/ui/templates/AudienceSelector'
import { TemplateContainer } from '@rbee/ui/molecules'

// Define props in page file
export const audienceSelectorContainerProps = {
  eyebrow: "Choose your path",
  title: "Where should you start?",
  description: "rbee adapts to how you work...",
  bgVariant: "subtle",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
}

export const audienceSelectorProps = {
  cards: [...],
  helperLinks: [...],
}

// In page component
<TemplateContainer {...audienceSelectorContainerProps}>
  <AudienceSelector {...audienceSelectorProps} />
</TemplateContainer>
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
} satisfies Meta<typeof AudienceSelector>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomeAudienceSelector: Story = {
  render: (args) => (
    <TemplateContainer {...audienceSelectorContainerProps}>
      <AudienceSelector {...args} />
    </TemplateContainer>
  ),
  args: audienceSelectorProps,
  parameters: {
    docs: {
      description: {
        story:
          'Audience selector as used on the HomePage. Props are imported from the HomePage to maintain a single source of truth.',
      },
    },
  },
}
