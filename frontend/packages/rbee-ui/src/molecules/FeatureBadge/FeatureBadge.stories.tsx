import type { Meta, StoryObj } from '@storybook/react'
import { FeatureBadge } from './FeatureBadge'

const meta = {
	title: 'Molecules/FeatureBadge',
	component: FeatureBadge,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
} satisfies Meta<typeof FeatureBadge>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		label: 'Feature',
	},
}

export const NoAPIFees: Story = {
	args: {
		label: 'No API fees',
	},
}

export const LocalTokens: Story = {
	args: {
		label: 'Local tokens',
	},
}

export const SecureByDefault: Story = {
	args: {
		label: 'Secure by default',
	},
}

export const MultiNode: Story = {
	args: {
		label: 'Multi-node',
	},
}

export const BackendAware: Story = {
	args: {
		label: 'Backend-aware',
	},
}

export const BadgeGroup: Story = {
	args: {
		label: 'No API fees',
	},
	decorators: [
		() => (
			<div className="flex flex-wrap gap-2 max-w-md">
				<FeatureBadge label="No API fees" />
				<FeatureBadge label="Local tokens" />
				<FeatureBadge label="Secure by default" />
				<FeatureBadge label="Multi-node" />
				<FeatureBadge label="Backend-aware" />
				<FeatureBadge label="Auto discovery" />
			</div>
		),
	],
	parameters: {
		docs: {
			description: {
				story: 'Example showing multiple badges together in a flex-wrap container.',
			},
		},
	},
}

export const LongLabel: Story = {
	args: {
		label: 'This is a very long feature label that demonstrates text wrapping',
	},
}
