import type { Meta, StoryObj } from '@storybook/react'
import { Badge } from '@rbee/ui/atoms'
import { Button } from '@rbee/ui/atoms'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './Card'

const meta = {
	title: 'Molecules/Card',
	component: Card,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The Card component is a versatile container for card-based layouts. It provides a consistent structure with optional header, content, and footer sections.

## Composition
This molecule is composed of:
- **Card**: The main container with border and background
- **CardHeader**: Optional header section for title and description
- **CardTitle**: Heading text styled with brand typography
- **CardDescription**: Subtitle or description text
- **CardContent**: Main content area
- **CardFooter**: Optional footer section for actions or metadata

## When to Use
- In feature grids to showcase product capabilities
- In problem/solution sections to list pain points or benefits
- For testimonial cards with quotes and attribution
- In pricing tables to display plan details
- Anywhere you need a contained, elevated content block

## Used In Commercial Site
- **ProblemSection**: Cards listing customer pain points
- **FeaturesSection**: Cards showcasing product features with badges
- **UseCasesSection**: Cards for different user personas
- **Enterprise/Providers Pages**: Feature and benefit cards
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof Card>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	render: () => (
		<Card style={{ width: '400px' }}>
			<CardHeader>
				<CardTitle>Card Title</CardTitle>
				<CardDescription>This is a card description</CardDescription>
			</CardHeader>
			<CardContent>
				<p>Card content goes here. You can put any content inside.</p>
			</CardContent>
		</Card>
	),
}

export const WithFooter: Story = {
	render: () => (
		<Card style={{ width: '400px' }}>
			<CardHeader>
				<CardTitle>Card with Footer</CardTitle>
				<CardDescription>This card has action buttons in the footer</CardDescription>
			</CardHeader>
			<CardContent>
				<p>This is the main content of the card. It can contain text, images, or other components.</p>
			</CardContent>
			<CardFooter style={{ gap: '0.5rem' }}>
				<Button size="sm">Confirm</Button>
				<Button size="sm" variant="outline">
					Cancel
				</Button>
			</CardFooter>
		</Card>
	),
}

export const WithBadge: Story = {
	render: () => (
		<Card style={{ width: '400px' }}>
			<CardHeader>
				<div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
					<CardTitle>Feature Card</CardTitle>
					<Badge>New</Badge>
				</div>
				<CardDescription>A new feature announcement</CardDescription>
			</CardHeader>
			<CardContent>
				<p>This card announces a new feature with a badge in the header.</p>
			</CardContent>
		</Card>
	),
}

export const FeatureGrid: Story = {
	render: () => (
		<div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem', maxWidth: '800px' }}>
			<Card>
				<CardHeader>
					<div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
						<CardTitle>Private Infrastructure</CardTitle>
						<Badge>Secure</Badge>
					</div>
					<CardDescription>Your data never leaves your infrastructure</CardDescription>
				</CardHeader>
				<CardContent>
					<p>Deploy models on your own hardware or use our managed service.</p>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
						<CardTitle>GDPR Compliant</CardTitle>
						<Badge variant="secondary">EU</Badge>
					</div>
					<CardDescription>Built for European data regulations</CardDescription>
				</CardHeader>
				<CardContent>
					<p>Full compliance with Dutch and European privacy laws.</p>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>High Performance</CardTitle>
					<CardDescription>Optimized for GPU acceleration</CardDescription>
				</CardHeader>
				<CardContent>
					<p>Fast inference with state-of-the-art optimization techniques.</p>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Easy to Deploy</CardTitle>
					<CardDescription>Simple setup and configuration</CardDescription>
				</CardHeader>
				<CardContent>
					<p>Get started in minutes with our straightforward deployment process.</p>
				</CardContent>
			</Card>
		</div>
	),
}

export const MinimalCard: Story = {
	render: () => (
		<Card style={{ width: '300px' }}>
			<CardContent style={{ paddingTop: '1.5rem' }}>
				<p>A minimal card with just content, no header or footer.</p>
			</CardContent>
		</Card>
	),
}

export const LongContent: Story = {
	render: () => (
		<Card style={{ width: '400px' }}>
			<CardHeader>
				<CardTitle>Long Content Example</CardTitle>
				<CardDescription>This card contains longer text content</CardDescription>
			</CardHeader>
			<CardContent>
				<p style={{ marginBottom: '1rem' }}>
					Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore
					magna aliqua.
				</p>
				<p style={{ marginBottom: '1rem' }}>
					Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
				</p>
				<p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.</p>
			</CardContent>
			<CardFooter>
				<Button variant="outline">Read More</Button>
			</CardFooter>
		</Card>
	),
}

export const ProblemCardExample: Story = {
	render: () => (
		<Card style={{ width: '350px' }}>
			<CardHeader>
				<CardTitle>Vendor Lock-In</CardTitle>
				<CardDescription>Trapped in proprietary ecosystems</CardDescription>
			</CardHeader>
			<CardContent>
				<p className="text-sm text-muted-foreground">
					Once you commit to a cloud AI provider, switching becomes prohibitively expensive. Your data, workflows, and integrations are all locked in.
				</p>
			</CardContent>
		</Card>
	),
	parameters: {
		docs: {
			description: {
				story: 'Card as used in ProblemSection organism. Lists customer pain points with title, description, and body text.',
			},
		},
	},
}

export const FeatureCardExample: Story = {
	render: () => (
		<Card style={{ width: '350px' }}>
			<CardHeader>
				<div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
					<CardTitle>Cross-Node Orchestration</CardTitle>
					<Badge>Core</Badge>
				</div>
				<CardDescription>Distribute workloads across multiple GPUs</CardDescription>
			</CardHeader>
			<CardContent>
				<p className="text-sm text-muted-foreground">
					Automatically route inference requests to available GPU nodes. Load balancing, failover, and health monitoring built-in.
				</p>
			</CardContent>
		</Card>
	),
	parameters: {
		docs: {
			description: {
				story: 'Card as used in FeaturesSection organism. Showcases product features with badge, title, description, and details.',
			},
		},
	},
}
