import type { Meta, StoryObj } from '@storybook/react'
import { FeatureTabContent } from './FeatureTabContent'
import { Alert, AlertDescription } from '@rbee/ui/atoms/Alert'

const meta = {
	title: 'Molecules/FeatureTabContent',
	component: FeatureTabContent,
	parameters: {
		layout: 'padded',
	},
	tags: ['autodocs'],
} satisfies Meta<typeof FeatureTabContent>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		children: (
			<>
				<div className="space-y-2">
					<h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">
						Feature Title
					</h3>
					<p className="text-xs text-muted-foreground">Feature subtitle</p>
				</div>
				<p className="text-base md:text-lg text-muted-foreground">
					Feature description explaining what this feature does and why it matters.
				</p>
			</>
		),
	},
}

export const WithAlert: Story = {
	args: {
		children: (
			<>
				<div className="space-y-2">
					<h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">
						OpenAI-Compatible API
					</h3>
					<p className="text-xs text-muted-foreground">Drop-in replacement for your existing tools</p>
				</div>
				<p className="text-base md:text-lg text-muted-foreground">
					Drop-in for Zed, Cursor, Continue, or any OpenAI client. Keep your SDKs and prompts—just change
					the base URL.
				</p>
				<Alert variant="success">
					<AlertDescription>No code changes. Just point to localhost.</AlertDescription>
				</Alert>
			</>
		),
	},
}

export const WithMultipleSections: Story = {
	args: {
		children: (
			<>
				<div className="space-y-2">
					<h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">
						Complex Feature
					</h3>
					<p className="text-xs text-muted-foreground">With multiple content sections</p>
				</div>
				<p className="text-base md:text-lg text-muted-foreground">
					This demonstrates how the card handles multiple sections of content with consistent spacing.
				</p>
				<div className="flex flex-wrap gap-2">
					<span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
						Feature 1
					</span>
					<span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
						Feature 2
					</span>
					<span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
						Feature 3
					</span>
				</div>
				<Alert variant="info">
					<AlertDescription>All content sections are properly spaced within the card.</AlertDescription>
				</Alert>
			</>
		),
	},
}
