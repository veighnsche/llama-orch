// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '@rbee/ui/atoms/Button'
import { Drawer, DrawerClose, DrawerContent, DrawerDescription, DrawerFooter, DrawerHeader, DrawerTitle, DrawerTrigger } from './Drawer'

const meta: Meta<typeof Drawer> = {
	title: 'Atoms/Drawer',
	component: Drawer,
	parameters: { layout: 'centered' },
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Drawer>

export const Default: Story = {
	render: () => (
		<Drawer>
			<DrawerTrigger asChild><Button variant="outline">Open Drawer</Button></DrawerTrigger>
			<DrawerContent>
				<DrawerHeader>
					<DrawerTitle>Drawer Title</DrawerTitle>
					<DrawerDescription>This is a drawer description.</DrawerDescription>
				</DrawerHeader>
				<div className="p-4"><p className="text-sm">Drawer content goes here.</p></div>
				<DrawerFooter>
					<Button>Submit</Button>
					<DrawerClose asChild><Button variant="outline">Cancel</Button></DrawerClose>
				</DrawerFooter>
			</DrawerContent>
		</Drawer>
	),
}

export const AllSides: Story = {
	render: () => (
		<div className="flex gap-4">
			<Drawer direction="bottom">
				<DrawerTrigger asChild><Button>Bottom</Button></DrawerTrigger>
				<DrawerContent><DrawerHeader><DrawerTitle>Bottom Drawer</DrawerTitle></DrawerHeader></DrawerContent>
			</Drawer>
			<Drawer direction="right">
				<DrawerTrigger asChild><Button>Right</Button></DrawerTrigger>
				<DrawerContent><DrawerHeader><DrawerTitle>Right Drawer</DrawerTitle></DrawerHeader></DrawerContent>
			</Drawer>
		</div>
	),
}

export const WithForm: Story = {
	render: () => (
		<Drawer>
			<DrawerTrigger asChild><Button>Edit Profile</Button></DrawerTrigger>
			<DrawerContent>
				<DrawerHeader>
					<DrawerTitle>Edit Profile</DrawerTitle>
					<DrawerDescription>Make changes here.</DrawerDescription>
				</DrawerHeader>
				<div className="p-4 space-y-4">
					<input className="w-full border rounded px-3 py-2" placeholder="Name" />
					<input className="w-full border rounded px-3 py-2" placeholder="Email" />
				</div>
				<DrawerFooter>
					<Button>Save</Button>
					<DrawerClose asChild><Button variant="outline">Cancel</Button></DrawerClose>
				</DrawerFooter>
			</DrawerContent>
		</Drawer>
	),
}

export const Nested: Story = {
	render: () => (
		<Drawer>
			<DrawerTrigger asChild><Button>Open First</Button></DrawerTrigger>
			<DrawerContent>
				<DrawerHeader><DrawerTitle>First Drawer</DrawerTitle></DrawerHeader>
				<div className="p-4">
					<Drawer>
						<DrawerTrigger asChild><Button size="sm">Open Nested</Button></DrawerTrigger>
						<DrawerContent>
							<DrawerHeader><DrawerTitle>Nested Drawer</DrawerTitle></DrawerHeader>
						</DrawerContent>
					</Drawer>
				</div>
			</DrawerContent>
		</Drawer>
	),
}
