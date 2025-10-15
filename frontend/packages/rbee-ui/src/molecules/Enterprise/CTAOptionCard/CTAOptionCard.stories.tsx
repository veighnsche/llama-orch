import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '@rbee/ui/atoms/Button'
import { Building2, Users } from 'lucide-react'
import { CTAOptionCard } from './CTAOptionCard'

const meta: Meta<typeof CTAOptionCard> = {
	title: 'Molecules/Enterprise/CTAOptionCard',
	component: CTAOptionCard,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The CTAOptionCard molecule presents a call-to-action option with an icon, title, body text, and action button. It's designed for side-by-side comparison of different paths or options.

## Composition
This molecule is composed of:
- **Icon**: Visual identifier (ReactNode)
- **Title**: Option name
- **Body**: Description text
- **Action**: Button or link component
- **Note**: Optional fine print

## When to Use
- Presenting multiple CTA options (e.g., "Contact Sales" vs "Self-Service")
- Enterprise vs Developer paths
- Different onboarding flows
- Pricing tier selection

## Used In
- **EnterpriseCTA**: Presents enterprise contact vs self-service options
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			control: false,
			description: 'Icon element to display',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
		title: {
			control: 'text',
			description: 'Card title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		body: {
			control: 'text',
			description: 'Description text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		action: {
			control: false,
			description: 'Action button or link',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
		tone: {
			control: 'select',
			options: ['primary', 'outline'],
			description: 'Visual emphasis level',
			table: {
				type: { summary: "'primary' | 'outline'" },
				defaultValue: { summary: 'outline' },
				category: 'Appearance',
			},
		},
		note: {
			control: 'text',
			description: 'Optional fine print below action',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof CTAOptionCard>

export const Default: Story = {
	args: {
		icon: <Building2 className="h-6 w-6" />,
		title: 'Enterprise',
		body: 'Custom solutions for large organizations with dedicated support and SLAs.',
		action: <Button variant="default">Contact Sales</Button>,
	},
}

export const WithIcon: Story = {
	args: {
		icon: <Users className="h-6 w-6" />,
		title: 'Self-Service',
		body: 'Get started immediately with our developer-friendly platform.',
		action: <Button variant="outline">Start Free Trial</Button>,
	},
}

export const Highlighted: Story = {
	args: {
		icon: <Building2 className="h-6 w-6" />,
		title: 'Enterprise',
		body: 'Custom solutions for large organizations with dedicated support and SLAs.',
		action: <Button variant="default">Contact Sales</Button>,
		tone: 'primary',
		note: 'Response within 24 hours',
	},
}

export const InCTAContext: Story = {
	render: () => (
		<div className="w-full max-w-5xl">
			<div className="mb-4 text-sm text-muted-foreground">Example: CTAOptionCard in EnterpriseCTA organism</div>
			<div className="grid gap-6 md:grid-cols-2">
				<CTAOptionCard
					icon={<Building2 className="h-6 w-6" />}
					title="Enterprise"
					body="Custom solutions for large organizations with dedicated support and SLAs."
					action={<Button variant="default" className="w-full">Contact Sales</Button>}
					tone="primary"
					note="Response within 24 hours"
				/>
				<CTAOptionCard
					icon={<Users className="h-6 w-6" />}
					title="Self-Service"
					body="Get started immediately with our developer-friendly platform."
					action={<Button variant="outline" className="w-full">Start Free Trial</Button>}
					note="No credit card required"
				/>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'CTAOptionCard as used in the EnterpriseCTA organism, presenting two paths: enterprise contact and self-service signup.',
			},
		},
	},
}
