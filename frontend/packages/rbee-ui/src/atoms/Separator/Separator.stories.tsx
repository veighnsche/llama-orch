// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { Separator } from './Separator'

const meta: Meta<typeof Separator> = {
	title: 'Atoms/Separator',
	component: Separator,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		orientation: {
			control: 'select',
			options: ['horizontal', 'vertical'],
			description: 'Separator orientation',
		},
		decorative: {
			control: 'boolean',
			description: 'Whether separator is purely decorative (affects accessibility)',
		},
	},
}

export default meta
type Story = StoryObj<typeof Separator>

/**
 * ## Overview
 * Separator is a visual divider that creates clear boundaries between content sections.
 * Built on Radix UI Separator primitive with proper ARIA semantics.
 *
 * ## When to Use
 * - Divide sections in a layout
 * - Separate items in a list
 * - Create visual hierarchy
 * - Break up dense content
 *
 * ## Used In
 * - Navigation menus
 * - Card sections
 * - Form groups
 * - Sidebar layouts
 * - And 8+ other organisms
 */

export const Default: Story = {
	render: () => (
		<div className="w-[300px]">
			<div className="space-y-1">
				<h4 className="text-sm font-medium">Section Title</h4>
				<p className="text-sm text-muted-foreground">Section description</p>
			</div>
			<Separator className="my-4" />
			<div className="space-y-1">
				<h4 className="text-sm font-medium">Another Section</h4>
				<p className="text-sm text-muted-foreground">More content here</p>
			</div>
		</div>
	),
}

export const Vertical: Story = {
	render: () => (
		<div className="flex h-20 items-center space-x-4">
			<div className="text-sm">Left Content</div>
			<Separator orientation="vertical" />
			<div className="text-sm">Middle Content</div>
			<Separator orientation="vertical" />
			<div className="text-sm">Right Content</div>
		</div>
	),
}

export const WithText: Story = {
	render: () => (
		<div className="w-[300px] space-y-4">
			<div>
				<p className="text-sm">Content above separator</p>
			</div>
			<div className="relative">
				<Separator />
				<div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-background px-2">
					<span className="text-xs text-muted-foreground">OR</span>
				</div>
			</div>
			<div>
				<p className="text-sm">Content below separator</p>
			</div>
		</div>
	),
}

export const InNavigation: Story = {
	render: () => (
		<div className="w-[250px] rounded-lg border p-4">
			<nav className="space-y-2">
				<div className="space-y-1">
					<a href="#" className="block text-sm font-medium hover:text-primary">
						Dashboard
					</a>
					<a href="#" className="block text-sm font-medium hover:text-primary">
						Models
					</a>
					<a href="#" className="block text-sm font-medium hover:text-primary">
						Deployments
					</a>
				</div>

				<Separator className="my-2" />

				<div className="space-y-1">
					<a href="#" className="block text-sm font-medium hover:text-primary">
						Settings
					</a>
					<a href="#" className="block text-sm font-medium hover:text-primary">
						Billing
					</a>
				</div>

				<Separator className="my-2" />

				<div className="space-y-1">
					<a href="#" className="block text-sm font-medium hover:text-primary">
						Documentation
					</a>
					<a href="#" className="block text-sm font-medium hover:text-primary">
						Support
					</a>
				</div>
			</nav>
		</div>
	),
}
