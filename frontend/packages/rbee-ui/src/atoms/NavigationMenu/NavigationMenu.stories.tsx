import type { Meta, StoryObj } from '@storybook/react'
import {
	NavigationMenu,
	NavigationMenuList,
	NavigationMenuItem,
	NavigationMenuTrigger,
	NavigationMenuContent,
	NavigationMenuLink,
} from './NavigationMenu'
import { HomeIcon, SettingsIcon, UserIcon, FileIcon } from 'lucide-react'

const meta: Meta<typeof NavigationMenu> = {
	title: 'Atoms/NavigationMenu',
	component: NavigationMenu,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof NavigationMenu>

export const Default: Story = {
	render: () => (
		<NavigationMenu>
			<NavigationMenuList>
				<NavigationMenuItem>
					<NavigationMenuTrigger>Getting started</NavigationMenuTrigger>
					<NavigationMenuContent>
						<div className="grid gap-3 p-4 w-[400px]">
							<NavigationMenuLink href="/docs">
								<div className="font-medium">Introduction</div>
								<div className="text-muted-foreground text-sm">
									Re-usable components built using Radix UI and Tailwind CSS.
								</div>
							</NavigationMenuLink>
							<NavigationMenuLink href="/docs/installation">
								<div className="font-medium">Installation</div>
								<div className="text-muted-foreground text-sm">How to install dependencies and structure your app.</div>
							</NavigationMenuLink>
							<NavigationMenuLink href="/docs/primitives/typography">
								<div className="font-medium">Typography</div>
								<div className="text-muted-foreground text-sm">Styles for headings, paragraphs, lists...etc</div>
							</NavigationMenuLink>
						</div>
					</NavigationMenuContent>
				</NavigationMenuItem>
				<NavigationMenuItem>
					<NavigationMenuTrigger>Components</NavigationMenuTrigger>
					<NavigationMenuContent>
						<div className="grid gap-3 p-4 w-[400px]">
							<NavigationMenuLink href="/docs/primitives/alert-dialog">
								<div className="font-medium">Alert Dialog</div>
								<div className="text-muted-foreground text-sm">
									A modal dialog that interrupts the user with important content.
								</div>
							</NavigationMenuLink>
							<NavigationMenuLink href="/docs/primitives/hover-card">
								<div className="font-medium">Hover Card</div>
								<div className="text-muted-foreground text-sm">
									For sighted users to preview content available behind a link.
								</div>
							</NavigationMenuLink>
						</div>
					</NavigationMenuContent>
				</NavigationMenuItem>
			</NavigationMenuList>
		</NavigationMenu>
	),
}

export const WithIcons: Story = {
	render: () => (
		<NavigationMenu>
			<NavigationMenuList>
				<NavigationMenuItem>
					<NavigationMenuTrigger>
						<HomeIcon className="mr-2 size-4" />
						Home
					</NavigationMenuTrigger>
					<NavigationMenuContent>
						<div className="grid gap-3 p-4 w-[300px]">
							<NavigationMenuLink href="/">
								<HomeIcon className="mr-2 size-4" />
								<div className="font-medium">Dashboard</div>
							</NavigationMenuLink>
							<NavigationMenuLink href="/overview">
								<FileIcon className="mr-2 size-4" />
								<div className="font-medium">Overview</div>
							</NavigationMenuLink>
						</div>
					</NavigationMenuContent>
				</NavigationMenuItem>
				<NavigationMenuItem>
					<NavigationMenuTrigger>
						<SettingsIcon className="mr-2 size-4" />
						Settings
					</NavigationMenuTrigger>
					<NavigationMenuContent>
						<div className="grid gap-3 p-4 w-[300px]">
							<NavigationMenuLink href="/settings/profile">
								<UserIcon className="mr-2 size-4" />
								<div className="font-medium">Profile</div>
							</NavigationMenuLink>
							<NavigationMenuLink href="/settings/preferences">
								<SettingsIcon className="mr-2 size-4" />
								<div className="font-medium">Preferences</div>
							</NavigationMenuLink>
						</div>
					</NavigationMenuContent>
				</NavigationMenuItem>
			</NavigationMenuList>
		</NavigationMenu>
	),
}

export const WithMegaMenu: Story = {
	render: () => (
		<NavigationMenu>
			<NavigationMenuList>
				<NavigationMenuItem>
					<NavigationMenuTrigger>Products</NavigationMenuTrigger>
					<NavigationMenuContent>
						<div className="grid grid-cols-2 gap-3 p-4 w-[600px]">
							<div className="space-y-3">
								<div className="font-semibold text-sm">For Developers</div>
								<NavigationMenuLink href="/products/api">
									<div className="font-medium">API</div>
									<div className="text-muted-foreground text-sm">Build with our powerful API</div>
								</NavigationMenuLink>
								<NavigationMenuLink href="/products/sdk">
									<div className="font-medium">SDK</div>
									<div className="text-muted-foreground text-sm">Official SDKs for popular languages</div>
								</NavigationMenuLink>
							</div>
							<div className="space-y-3">
								<div className="font-semibold text-sm">For Teams</div>
								<NavigationMenuLink href="/products/enterprise">
									<div className="font-medium">Enterprise</div>
									<div className="text-muted-foreground text-sm">Advanced features for large teams</div>
								</NavigationMenuLink>
								<NavigationMenuLink href="/products/support">
									<div className="font-medium">Support</div>
									<div className="text-muted-foreground text-sm">24/7 dedicated support</div>
								</NavigationMenuLink>
							</div>
						</div>
					</NavigationMenuContent>
				</NavigationMenuItem>
			</NavigationMenuList>
		</NavigationMenu>
	),
}

export const Mobile: Story = {
	render: () => (
		<NavigationMenu className="w-full max-w-full">
			<NavigationMenuList className="flex-col items-start w-full">
				<NavigationMenuItem className="w-full">
					<NavigationMenuTrigger className="w-full justify-start">Home</NavigationMenuTrigger>
					<NavigationMenuContent>
						<div className="flex flex-col gap-2 p-4 w-full">
							<NavigationMenuLink href="/">Dashboard</NavigationMenuLink>
							<NavigationMenuLink href="/overview">Overview</NavigationMenuLink>
							<NavigationMenuLink href="/analytics">Analytics</NavigationMenuLink>
						</div>
					</NavigationMenuContent>
				</NavigationMenuItem>
				<NavigationMenuItem className="w-full">
					<NavigationMenuTrigger className="w-full justify-start">Products</NavigationMenuTrigger>
					<NavigationMenuContent>
						<div className="flex flex-col gap-2 p-4 w-full">
							<NavigationMenuLink href="/products">All Products</NavigationMenuLink>
							<NavigationMenuLink href="/products/new">New Arrivals</NavigationMenuLink>
							<NavigationMenuLink href="/products/sale">On Sale</NavigationMenuLink>
						</div>
					</NavigationMenuContent>
				</NavigationMenuItem>
			</NavigationMenuList>
		</NavigationMenu>
	),
}
