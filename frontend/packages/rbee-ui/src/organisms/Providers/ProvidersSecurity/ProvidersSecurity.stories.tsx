import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersSecurity } from './ProvidersSecurity'

const meta = {
	title: 'Organisms/Providers/ProvidersSecurity',
	component: ProvidersSecurity,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProvidersSecurity section addresses provider security concerns with four key security features (Sandboxed Execution, Encrypted Communication, Malware Scanning, Hardware Protection) and a €1M insurance coverage ribbon. It uses a 2-column grid layout with emerald accent colors for trust signaling.

## Two-Sided Marketplace Strategy

### Provider Security Concerns (FUD Addressing)
1. **"Is my system safe?"** → Sandboxed execution, no file/network access
2. **"Can workloads access my files?"** → No, complete isolation
3. **"What if a workload is malicious?"** → Malware scanning, automatic blocking
4. **"Will this damage my hardware?"** → Temperature monitoring, cooldown periods, warranty-safe

### Trust Building Strategy
- **Technical Security:** Sandboxing, encryption, malware scanning
- **Hardware Protection:** Temperature monitoring, power limits, health monitoring
- **Financial Protection:** €1M insurance coverage included
- **Transparency:** Clear explanation of each security layer

### Risk Mitigation
- **Data Risk:** "No file system access, no network access, no personal data access"
- **Malware Risk:** "Real-time detection, automatic blocking, threat intel updates"
- **Hardware Risk:** "Temperature monitoring, cooldown periods, power limits, health monitoring"
- **Financial Risk:** "€1M insurance coverage is included for all providers"

## Composition
This organism contains:
- **Kicker**: "Security & Trust"
- **Title**: "Your Security Is Our Priority"
- **Subtitle**: "Enterprise-grade protections for your hardware, data, and earnings."
- **Security Cards Grid** (2×2 grid):
  1. Sandboxed Execution (Shield icon)
  2. Encrypted Communication (Lock icon)
  3. Malware Scanning (Eye icon)
  4. Hardware Protection (FileCheck icon)
- **Insurance Ribbon**: "€1M insurance coverage is included for all providers—your hardware is protected."

## When to Use
- On the GPU Providers page to address security concerns
- To build trust with potential providers
- To differentiate from less secure platforms

## Content Requirements
- **Security Features:** Must address all major provider concerns
- **Technical Details:** Specific protections (TLS 1.3, sandboxing, etc.)
- **Insurance:** €1M coverage as ultimate trust signal
- **Tone:** Reassuring, technical (but accessible), trust-building

## Marketing Strategy
- **Target Audience:** Providers concerned about security and hardware safety
- **Primary Message:** "Enterprise-grade security protects your hardware and data"
- **Emotional Appeal:** Trust (comprehensive protection) + Confidence (insurance coverage)
- **Copy Tone:** Reassuring, technical, trust-building

## Variants
- **Default**: All four security features with insurance ribbon
- **IsolationFocus**: Lead with sandboxing and isolation details
- **MonitoringFocus**: Emphasize hardware protection and monitoring

## Examples
\`\`\`tsx
import { ProvidersSecurity } from '@rbee/ui/organisms/Providers/ProvidersSecurity'

// Simple usage - no props needed
<ProvidersSecurity />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- SecuritySection (shared base component)
- IconPlate (for icon display)
- ProvidersMarketplace (trust signals)

## Accessibility
- **Semantic HTML**: Proper heading hierarchy and landmark regions
- **Icon Labels**: Icons have aria-hidden="true"
- **Keyboard Navigation**: All interactive elements are keyboard accessible
- **Color Contrast**: Emerald accent colors meet WCAG AA standards
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof ProvidersSecurity>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersSecurity as used on /gpu-providers page.
 * Shows all four security features with insurance ribbon.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing sandboxing and isolation.
 * Focuses on "no file access, no network access" messaging.
 */
export const IsolationFocus: Story = {}

/**
 * Variant emphasizing hardware protection and monitoring.
 * Highlights temperature monitoring, cooldown periods, and warranty-safe operation.
 */
export const MonitoringFocus: Story = {}
