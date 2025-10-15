import type { Meta, StoryObj } from '@storybook/react'
import { BeeArchitecture } from './BeeArchitecture'

const meta: Meta<typeof BeeArchitecture> = {
	title: 'Molecules/BeeArchitecture',
	component: BeeArchitecture,
	parameters: {
		layout: 'padded',
		docs: {
			description: {
				component: `
## Overview
BeeArchitecture is a specialized diagram molecule that visualizes the rbee system architecture. It shows the relationship between queen-rbee (orchestrator), rbee-hive (resource manager), and worker nodes.

## Composition
This molecule is composed of:
- **Queen node**: Orchestrator (brain) with crown emoji
- **Hive node**: Resource manager with honeycomb emoji
- **Worker nodes**: GPU/CPU workers with bee emojis
- **Connectors**: Visual lines connecting components
- **Host containers**: Boxes grouping workers by host

## When to Use
- Solution sections (explaining architecture)
- How It Works sections (technical diagrams)
- Documentation (system overview)
- Marketing materials (visual differentiation)

## Variants
- **Single PC**: All workers on one host
- **Multi-host**: Workers distributed across multiple hosts
- **Worker types**: CUDA, Metal, CPU with different ring colors

## Used In Commercial Site
Used in:
- HomeSolutionSection (architecture overview)
- DevelopersSolution (technical architecture)
- HowItWorksSection (system diagram)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		topology: {
			control: 'object',
			description: 'Topology configuration (single-pc or multi-host)',
			table: {
				type: { summary: 'BeeTopology' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof BeeArchitecture>

export const SingleHost: Story = {
	args: {
		topology: {
			mode: 'single-pc',
			hostLabel: 'Your PC (Ubuntu 22.04)',
			workers: [
				{ id: 'w1', label: 'RTX 4090', kind: 'cuda' },
				{ id: 'w2', label: 'RTX 3080', kind: 'cuda' },
			],
		},
	},
}

export const MultiHost: Story = {
	args: {
		topology: {
			mode: 'multi-host',
			hosts: [
				{
					hostLabel: 'Workstation (Ubuntu)',
					workers: [
						{ id: 'w1', label: 'RTX 4090', kind: 'cuda' },
						{ id: 'w2', label: 'RTX 3080', kind: 'cuda' },
					],
				},
				{
					hostLabel: 'MacBook Pro',
					workers: [{ id: 'w3', label: 'M2 Max', kind: 'metal' }],
				},
				{
					hostLabel: 'Server (CPU)',
					workers: [
						{ id: 'w4', label: 'Xeon 1', kind: 'cpu' },
						{ id: 'w5', label: 'Xeon 2', kind: 'cpu' },
					],
				},
			],
		},
	},
}

export const WithLabels: Story = {
	render: () => (
		<div className="space-y-8">
			<div>
				<h3 className="text-lg font-semibold mb-4">Single PC Setup</h3>
				<p className="text-sm text-muted-foreground mb-6">
					Perfect for developers and small teams. Run everything on one machine.
				</p>
				<BeeArchitecture
					topology={{
						mode: 'single-pc',
						hostLabel: 'Your Development Machine',
						workers: [
							{ id: 'w1', label: 'RTX 4090', kind: 'cuda' },
							{ id: 'w2', label: 'RTX 3080', kind: 'cuda' },
						],
					}}
				/>
			</div>
			<div>
				<h3 className="text-lg font-semibold mb-4">Multi-Host Setup</h3>
				<p className="text-sm text-muted-foreground mb-6">
					Scale across multiple machines for enterprise workloads.
				</p>
				<BeeArchitecture
					topology={{
						mode: 'multi-host',
						hosts: [
							{
								hostLabel: 'GPU Server 1',
								workers: [
									{ id: 'w1', label: 'A100', kind: 'cuda' },
									{ id: 'w2', label: 'A100', kind: 'cuda' },
								],
							},
							{
								hostLabel: 'GPU Server 2',
								workers: [
									{ id: 'w3', label: 'RTX 4090', kind: 'cuda' },
									{ id: 'w4', label: 'RTX 4090', kind: 'cuda' },
								],
							},
						],
					}}
				/>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Architecture diagrams with descriptive labels and context.',
			},
		},
	},
}

export const Interactive: Story = {
	render: () => (
		<div className="w-full">
			<div className="mb-6 text-center">
				<h2 className="text-3xl font-bold mb-2">The Bee Architecture</h2>
				<p className="text-muted-foreground">
					Orchestrate GPU workers across your infrastructure with queen-rbee and rbee-hive
				</p>
			</div>
			<BeeArchitecture
				topology={{
					mode: 'multi-host',
					hosts: [
						{
							hostLabel: 'Office Workstation',
							workers: [
								{ id: 'w1', label: 'RTX 4090', kind: 'cuda' },
								{ id: 'w2', label: 'RTX 3090', kind: 'cuda' },
							],
						},
						{
							hostLabel: 'MacBook Pro M2',
							workers: [{ id: 'w3', label: 'M2 Max', kind: 'metal' }],
						},
						{
							hostLabel: 'Home Server',
							workers: [
								{ id: 'w4', label: 'RTX 3080', kind: 'cuda' },
								{ id: 'w5', label: 'CPU Fallback', kind: 'cpu' },
							],
						},
					],
				}}
			/>
			<div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
				<div className="p-4 bg-card rounded-lg border border-border">
					<div className="flex items-center gap-2 mb-2">
						<div className="h-3 w-3 rounded-full bg-amber-400/30 ring-1 ring-amber-400/30"></div>
						<span className="font-semibold text-sm">CUDA Workers</span>
					</div>
					<p className="text-xs text-muted-foreground">NVIDIA GPUs with CUDA acceleration</p>
				</div>
				<div className="p-4 bg-card rounded-lg border border-border">
					<div className="flex items-center gap-2 mb-2">
						<div className="h-3 w-3 rounded-full bg-sky-400/30 ring-1 ring-sky-400/30"></div>
						<span className="font-semibold text-sm">Metal Workers</span>
					</div>
					<p className="text-xs text-muted-foreground">Apple Silicon with Metal acceleration</p>
				</div>
				<div className="p-4 bg-card rounded-lg border border-border">
					<div className="flex items-center gap-2 mb-2">
						<div className="h-3 w-3 rounded-full bg-emerald-400/30 ring-1 ring-emerald-400/30"></div>
						<span className="font-semibold text-sm">CPU Workers</span>
					</div>
					<p className="text-xs text-muted-foreground">CPU-only fallback workers</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'BeeArchitecture as used in HomeSolutionSection with legend and context.',
			},
		},
	},
}
