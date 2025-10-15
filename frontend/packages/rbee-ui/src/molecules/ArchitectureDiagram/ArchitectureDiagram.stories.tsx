import type { Meta, StoryObj } from '@storybook/react'
import { ArchitectureDiagram } from './ArchitectureDiagram'

const meta: Meta<typeof ArchitectureDiagram> = {
	title: 'Molecules/ArchitectureDiagram',
	component: ArchitectureDiagram,
	parameters: {
		layout: 'padded',
		docs: {
			description: {
				component: `
## Overview
The ArchitectureDiagram molecule displays the rbee system architecture using a topology diagram. It shows the three-tier architecture: Queen (orchestrator), Hive Managers, and Workers.

## Composition
This molecule is composed of:
- **TopologyDiagram**: The underlying organism that renders nodes and edges
- **TDNode**: Individual nodes representing system components
- **TDEdge**: Connections showing control and telemetry flows

## When to Use
- In the Developers page to explain system architecture
- In documentation to show component relationships
- In marketing materials to demonstrate scalability
- In technical presentations

## Variants
- **Simple**: Basic diagram without labels or legend
- **Detailed**: Full diagram with lane labels and legend

## Used In Commercial Site
- **Developers Page**: Shows the three-tier architecture with detailed labels
- **Features Page**: Demonstrates scalability and distributed architecture
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		variant: {
			control: 'select',
			options: ['simple', 'detailed'],
			description: 'Diagram complexity level',
			table: {
				type: { summary: "'simple' | 'detailed'" },
				defaultValue: { summary: 'simple' },
				category: 'Appearance',
			},
		},
		showLabels: {
			control: 'boolean',
			description: 'Whether to show node labels',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Appearance',
			},
		},
		className: {
			control: 'text',
			description: 'Additional CSS classes',
			table: {
				type: { summary: 'string' },
				category: 'Styling',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof ArchitectureDiagram>

export const Default: Story = {
	args: {
		variant: 'simple',
		showLabels: true,
	},
}

export const Simple: Story = {
	args: {
		variant: 'simple',
		showLabels: true,
	},
	parameters: {
		docs: {
			description: {
				story: 'Simple diagram showing the basic architecture without legend or lane labels.',
			},
		},
	},
}

export const Detailed: Story = {
	args: {
		variant: 'detailed',
		showLabels: true,
	},
	parameters: {
		docs: {
			description: {
				story: 'Detailed diagram with legend and lane labels showing control tier, manager tier, and execution tier.',
			},
		},
	},
}

export const WithoutLabels: Story = {
	args: {
		variant: 'simple',
		showLabels: false,
	},
	parameters: {
		docs: {
			description: {
				story: 'Diagram with labels hidden, showing only the topology structure.',
			},
		},
	},
}

export const AllVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-8 p-8">
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Simple Variant</h3>
				<ArchitectureDiagram variant="simple" showLabels={true} />
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Detailed Variant</h3>
				<ArchitectureDiagram variant="detailed" showLabels={true} />
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available variants side by side. Simple shows basic topology, detailed includes legend and tier labels.',
			},
		},
	},
}
