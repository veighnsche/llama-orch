import type { Meta, StoryObj } from '@storybook/react'
import { FeatureHeader } from './FeatureHeader'

const meta = {
	title: 'Molecules/FeatureHeader',
	component: FeatureHeader,
	parameters: {
		layout: 'padded',
	},
	tags: ['autodocs'],
} satisfies Meta<typeof FeatureHeader>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		title: 'Feature Title',
		subtitle: 'Feature subtitle describing the feature',
	},
}

export const OpenAIAPI: Story = {
	args: {
		title: 'OpenAI-Compatible API',
		subtitle: 'Drop-in replacement for your existing tools',
	},
}

export const MultiGPU: Story = {
	args: {
		title: 'Multi-GPU Orchestration',
		subtitle: 'Unified pool across all your hardware',
	},
}

export const Scheduler: Story = {
	args: {
		title: 'Programmable Rhai Scheduler',
		subtitle: 'Custom routing logic for your workloads',
	},
}

export const RealTime: Story = {
	args: {
		title: 'Task-Based API with SSE',
		subtitle: 'Stream job lifecycle into your UI',
	},
}

export const LongTitle: Story = {
	args: {
		title: 'This is a Very Long Feature Title That Demonstrates Text Wrapping Behavior',
		subtitle: 'And this is a longer subtitle that provides additional context about the feature',
	},
}
