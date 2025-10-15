// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { RadioGroup, RadioGroupItem } from './RadioGroup'

const meta: Meta<typeof RadioGroup> = {
	title: 'Atoms/RadioGroup',
	component: RadioGroup,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
A radio group component built on Radix UI RadioGroup primitive.

## Features
- Single selection from multiple options
- Accessible with proper ARIA attributes
- Keyboard navigation support
- Focus visible states
- Disabled state support
- Horizontal and vertical layouts

## Used In
- Forms
- Settings panels
- Surveys
- Configuration wizards
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		defaultValue: {
			control: 'text',
			description: 'Default selected value',
		},
		disabled: {
			control: 'boolean',
			description: 'Disable all radio items',
		},
	},
}

export default meta
type Story = StoryObj<typeof RadioGroup>

export const Default: Story = {
	render: () => (
		<RadioGroup defaultValue="option-1">
			<div className="flex items-center gap-2">
				<RadioGroupItem value="option-1" id="option-1" />
				<label htmlFor="option-1" className="text-sm cursor-pointer">
					Option 1
				</label>
			</div>
			<div className="flex items-center gap-2">
				<RadioGroupItem value="option-2" id="option-2" />
				<label htmlFor="option-2" className="text-sm cursor-pointer">
					Option 2
				</label>
			</div>
			<div className="flex items-center gap-2">
				<RadioGroupItem value="option-3" id="option-3" />
				<label htmlFor="option-3" className="text-sm cursor-pointer">
					Option 3
				</label>
			</div>
		</RadioGroup>
	),
}

export const Horizontal: Story = {
	render: () => (
		<RadioGroup defaultValue="small" className="flex gap-4">
			<div className="flex items-center gap-2">
				<RadioGroupItem value="small" id="size-small" />
				<label htmlFor="size-small" className="text-sm cursor-pointer">
					Small
				</label>
			</div>
			<div className="flex items-center gap-2">
				<RadioGroupItem value="medium" id="size-medium" />
				<label htmlFor="size-medium" className="text-sm cursor-pointer">
					Medium
				</label>
			</div>
			<div className="flex items-center gap-2">
				<RadioGroupItem value="large" id="size-large" />
				<label htmlFor="size-large" className="text-sm cursor-pointer">
					Large
				</label>
			</div>
		</RadioGroup>
	),
}

export const WithDescription: Story = {
	render: () => (
		<RadioGroup defaultValue="standard" className="gap-4">
			<div className="flex items-start gap-3">
				<RadioGroupItem value="standard" id="plan-standard" className="mt-1" />
				<div className="grid gap-1.5 leading-none">
					<label
						htmlFor="plan-standard"
						className="text-sm font-medium leading-none cursor-pointer"
					>
						Standard Plan
					</label>
					<p className="text-sm text-muted-foreground">
						Perfect for small teams. Up to 10 users, 100GB storage.
					</p>
					<p className="text-sm font-semibold">$29/month</p>
				</div>
			</div>

			<div className="flex items-start gap-3">
				<RadioGroupItem value="pro" id="plan-pro" className="mt-1" />
				<div className="grid gap-1.5 leading-none">
					<label htmlFor="plan-pro" className="text-sm font-medium leading-none cursor-pointer">
						Pro Plan
					</label>
					<p className="text-sm text-muted-foreground">
						For growing businesses. Up to 50 users, 500GB storage.
					</p>
					<p className="text-sm font-semibold">$99/month</p>
				</div>
			</div>

			<div className="flex items-start gap-3">
				<RadioGroupItem value="enterprise" id="plan-enterprise" className="mt-1" />
				<div className="grid gap-1.5 leading-none">
					<label
						htmlFor="plan-enterprise"
						className="text-sm font-medium leading-none cursor-pointer"
					>
						Enterprise Plan
					</label>
					<p className="text-sm text-muted-foreground">
						Unlimited users and storage. Dedicated support and SLA.
					</p>
					<p className="text-sm font-semibold">Custom pricing</p>
				</div>
			</div>
		</RadioGroup>
	),
}

export const DisabledState: Story = {
	render: () => (
		<div className="flex flex-col gap-6">
			<div>
				<h4 className="text-sm font-semibold mb-3">Normal State</h4>
				<RadioGroup defaultValue="enabled-1">
					<div className="flex items-center gap-2">
						<RadioGroupItem value="enabled-1" id="enabled-1" />
						<label htmlFor="enabled-1" className="text-sm cursor-pointer">
							Enabled option 1
						</label>
					</div>
					<div className="flex items-center gap-2">
						<RadioGroupItem value="enabled-2" id="enabled-2" />
						<label htmlFor="enabled-2" className="text-sm cursor-pointer">
							Enabled option 2
						</label>
					</div>
				</RadioGroup>
			</div>

			<div>
				<h4 className="text-sm font-semibold mb-3">Disabled State</h4>
				<RadioGroup defaultValue="disabled-1" disabled>
					<div className="flex items-center gap-2">
						<RadioGroupItem value="disabled-1" id="disabled-1" />
						<label htmlFor="disabled-1" className="text-sm text-muted-foreground">
							Disabled option 1
						</label>
					</div>
					<div className="flex items-center gap-2">
						<RadioGroupItem value="disabled-2" id="disabled-2" />
						<label htmlFor="disabled-2" className="text-sm text-muted-foreground">
							Disabled option 2
						</label>
					</div>
				</RadioGroup>
			</div>

			<div>
				<h4 className="text-sm font-semibold mb-3">Mixed State</h4>
				<RadioGroup defaultValue="mixed-1">
					<div className="flex items-center gap-2">
						<RadioGroupItem value="mixed-1" id="mixed-1" />
						<label htmlFor="mixed-1" className="text-sm cursor-pointer">
							Available option
						</label>
					</div>
					<div className="flex items-center gap-2">
						<RadioGroupItem value="mixed-2" id="mixed-2" disabled />
						<label htmlFor="mixed-2" className="text-sm text-muted-foreground">
							Unavailable option
						</label>
					</div>
					<div className="flex items-center gap-2">
						<RadioGroupItem value="mixed-3" id="mixed-3" />
						<label htmlFor="mixed-3" className="text-sm cursor-pointer">
							Another available option
						</label>
					</div>
				</RadioGroup>
			</div>
		</div>
	),
}

export const InForm: Story = {
	render: () => (
		<form className="w-full max-w-md space-y-6 p-6 border border-border rounded-lg">
			<div>
				<h3 className="text-lg font-semibold mb-2">Model Configuration</h3>
				<p className="text-sm text-muted-foreground">Choose your deployment settings</p>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">Model Size</label>
				<RadioGroup defaultValue="8b">
					<div className="flex items-center justify-between p-3 border border-border rounded-lg">
						<div className="flex items-center gap-3">
							<RadioGroupItem value="8b" id="model-8b" />
							<label htmlFor="model-8b" className="text-sm cursor-pointer">
								Llama 3.1 8B
							</label>
						</div>
						<span className="text-xs text-muted-foreground">8GB VRAM</span>
					</div>
					<div className="flex items-center justify-between p-3 border border-border rounded-lg">
						<div className="flex items-center gap-3">
							<RadioGroupItem value="70b" id="model-70b" />
							<label htmlFor="model-70b" className="text-sm cursor-pointer">
								Llama 3.1 70B
							</label>
						</div>
						<span className="text-xs text-muted-foreground">40GB VRAM</span>
					</div>
				</RadioGroup>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">Deployment Region</label>
				<RadioGroup defaultValue="nl">
					<div className="flex items-center gap-2">
						<RadioGroupItem value="nl" id="region-nl" />
						<label htmlFor="region-nl" className="text-sm cursor-pointer">
							🇳🇱 Netherlands (Amsterdam)
						</label>
					</div>
					<div className="flex items-center gap-2">
						<RadioGroupItem value="de" id="region-de" />
						<label htmlFor="region-de" className="text-sm cursor-pointer">
							🇩🇪 Germany (Frankfurt)
						</label>
					</div>
					<div className="flex items-center gap-2">
						<RadioGroupItem value="fr" id="region-fr" />
						<label htmlFor="region-fr" className="text-sm cursor-pointer">
							🇫🇷 France (Paris)
						</label>
					</div>
				</RadioGroup>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">Scaling Strategy</label>
				<RadioGroup defaultValue="auto">
					<div className="flex items-start gap-3">
						<RadioGroupItem value="auto" id="scale-auto" className="mt-1" />
						<div className="grid gap-1 leading-none">
							<label htmlFor="scale-auto" className="text-sm font-medium cursor-pointer">
								Auto-scaling
							</label>
							<p className="text-xs text-muted-foreground">
								Automatically scale based on demand
							</p>
						</div>
					</div>
					<div className="flex items-start gap-3">
						<RadioGroupItem value="fixed" id="scale-fixed" className="mt-1" />
						<div className="grid gap-1 leading-none">
							<label htmlFor="scale-fixed" className="text-sm font-medium cursor-pointer">
								Fixed capacity
							</label>
							<p className="text-xs text-muted-foreground">
								Maintain constant number of instances
							</p>
						</div>
					</div>
				</RadioGroup>
			</div>

			<button
				type="submit"
				className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
			>
				Deploy Model
			</button>
		</form>
	),
}
