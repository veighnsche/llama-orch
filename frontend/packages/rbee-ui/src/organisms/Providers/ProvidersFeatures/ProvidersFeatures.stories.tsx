import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersFeatures } from './ProvidersFeatures'

const meta = {
  title: 'Organisms/Providers/ProvidersFeatures',
  component: ProvidersFeatures,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The ProvidersFeatures section showcases the professional-grade tools available to GPU providers for managing their fleet and maximizing earnings. It uses a tabbed interface with 6 feature categories, each with code examples and benefits.

## Two-Sided Marketplace Strategy

### Provider Control Features
1. **Flexible Pricing Control:** Set base rates, min/max, demand multipliers, schedules
2. **Availability Management:** Control when GPUs are rentable, priority modes, auto-pause
3. **Security & Privacy:** Sandboxed execution, encrypted communication, malware scanning
4. **Earnings Dashboard:** Real-time tracking, historical charts, utilization metrics
5. **Usage Limits:** Temperature monitoring, power caps, cooldown periods, warranty mode
6. **Performance Optimization:** Automatic model selection, load balancing, priority queues

### Which Features Reduce Provider Friction?
- **Security & Privacy:** Addresses fear of file access and malware
- **Usage Limits:** Addresses fear of hardware damage and warranty voiding
- **Availability Management:** Addresses concern about losing access to own GPU
- **Earnings Dashboard:** Addresses trust in payment transparency

### Which Features Increase Provider Earnings?
- **Flexible Pricing Control:** Dynamic pricing based on demand
- **Performance Optimization:** Automatic optimization for higher earnings
- **Earnings Dashboard:** Insights to optimize pricing and availability

## Composition
This organism contains:
- **Title**: "Everything You Need to Maximize Earnings"
- **Subtitle**: "Professional-grade tools to manage your GPU fleet and optimize your passive income."
- **Feature Tabs**: 6 tabs with icons, titles, descriptions, benefits, and code examples
  1. Pricing (DollarSign icon)
  2. Availability (Clock icon)
  3. Security (Shield icon)
  4. Analytics (BarChart3 icon)
  5. Limits (Sliders icon)
  6. Performance (Zap icon)

## When to Use
- On the GPU Providers page after the how-it-works section
- To showcase advanced control and optimization features
- To differentiate from simple GPU rental platforms

## Content Requirements
- **Feature Tabs**: Must address provider concerns and earning optimization
- **Code Examples**: Realistic configuration snippets
- **Benefits**: Clear value statements for each feature
- **Tone**: Professional, empowering, control-focused

## Marketing Strategy
- **Target Audience:** GPU owners who want control and optimization
- **Primary Message:** "Professional tools to maximize your passive income"
- **Emotional Appeal:** Empowerment (control) + Confidence (professional tools)
- **Copy Tone:** Professional, technical (but accessible), empowering

## Variants
- **Default**: All 6 feature tabs with code examples
- **ControlFocus**: Emphasize pricing, availability, and limits
- **EarningsTrackingFocus**: Lead with analytics and optimization features

## Examples
\`\`\`tsx
import { ProvidersFeatures } from '@rbee/ui/organisms/Providers/ProvidersFeatures'

// Simple usage - no props needed
<ProvidersFeatures />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- FeatureTabsSection (shared base component)
- ProvidersHowItWorks
- ProvidersUseCases

## Accessibility
- **Keyboard Navigation**: Tab list is keyboard accessible (arrow keys)
- **ARIA Roles**: Proper tab/tabpanel roles
- **Focus Management**: Focus moves to selected tab panel
- **Screen Readers**: Tab labels and content are properly announced
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersFeatures>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersFeatures as used on /gpu-providers page.
 * Shows all 6 feature tabs with code examples.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing provider control.
 * Highlights pricing, availability, and usage limits.
 */
export const ControlFocus: Story = {}

/**
 * Variant emphasizing earnings tracking and optimization.
 * Leads with analytics dashboard and performance optimization.
 */
export const EarningsTrackingFocus: Story = {}
