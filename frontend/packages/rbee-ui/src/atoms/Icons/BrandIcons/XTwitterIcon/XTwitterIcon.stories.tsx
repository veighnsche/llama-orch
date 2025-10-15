// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { XTwitterIcon } from './XTwitterIcon'

const meta: Meta<typeof XTwitterIcon> = {
	title: 'Atoms/Icons/Brand/XTwitterIcon',
	component: XTwitterIcon,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
X (formerly Twitter) brand icon component.

## Features
- Official X/Twitter logo SVG
- Respects currentColor for theming
- Accessible with aria-hidden
- Default size of 20px (size-5)

## Used In
- Footer social links
- Share buttons
- Social media sections
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		className: {
			control: 'text',
			description: 'Additional CSS classes',
		},
	},
}

export default meta
type Story = StoryObj<typeof XTwitterIcon>

export const Default: Story = {
	args: {},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex items-end gap-6">
			<div className="flex flex-col items-center gap-2">
				<XTwitterIcon className="size-3" />
				<span className="text-xs text-muted-foreground">12px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<XTwitterIcon className="size-4" />
				<span className="text-xs text-muted-foreground">16px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<XTwitterIcon />
				<span className="text-xs text-muted-foreground">20px (default)</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<XTwitterIcon className="size-6" />
				<span className="text-xs text-muted-foreground">24px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<XTwitterIcon className="size-8" />
				<span className="text-xs text-muted-foreground">32px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<XTwitterIcon className="size-12" />
				<span className="text-xs text-muted-foreground">48px</span>
			</div>
		</div>
	),
}

export const WithLink: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<div>
				<h4 className="text-sm font-semibold mb-3">As Link</h4>
				<a
					href="https://twitter.com/orchyra_ai"
					target="_blank"
					rel="noopener noreferrer"
					className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
				>
					<XTwitterIcon className="size-5" />
					<span>Follow us on X</span>
				</a>
			</div>
			<div>
				<h4 className="text-sm font-semibold mb-3">Icon Only Link</h4>
				<a
					href="https://twitter.com/orchyra_ai"
					target="_blank"
					rel="noopener noreferrer"
					className="inline-flex items-center justify-center size-10 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
					aria-label="Follow us on X"
				>
					<XTwitterIcon />
				</a>
			</div>
			<div>
				<h4 className="text-sm font-semibold mb-3">With Custom Color</h4>
				<a
					href="https://twitter.com/orchyra_ai"
					target="_blank"
					rel="noopener noreferrer"
					className="inline-flex items-center justify-center size-10 rounded-lg text-blue-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-950 transition-colors"
					aria-label="Follow us on X"
				>
					<XTwitterIcon />
				</a>
			</div>
		</div>
	),
}

export const InFooter: Story = {
	render: () => (
		<footer className="w-full max-w-4xl border-t border-border bg-background">
			<div className="p-8">
				<div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
					<div>
						<h3 className="text-lg font-semibold mb-2">Orchyra</h3>
						<p className="text-sm text-muted-foreground">Private LLM Hosting in the Netherlands</p>
					</div>
					<div className="flex flex-col gap-3">
						<span className="text-sm font-medium">Follow us</span>
						<div className="flex gap-3">
							<a
								href="https://twitter.com/orchyra_ai"
								target="_blank"
								rel="noopener noreferrer"
								className="inline-flex items-center justify-center size-10 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
								aria-label="Follow us on X"
							>
								<XTwitterIcon />
							</a>
							<a
								href="https://github.com/orchyra"
								target="_blank"
								rel="noopener noreferrer"
								className="inline-flex items-center justify-center size-10 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
								aria-label="Follow us on GitHub"
							>
								<svg className="size-5" viewBox="0 0 24 24" fill="currentColor">
									<path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
								</svg>
							</a>
						</div>
					</div>
				</div>
			</div>
		</footer>
	),
}
