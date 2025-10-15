import type { Meta, StoryObj } from '@storybook/react'
import {
	Field,
	FieldLabel,
	FieldDescription,
	FieldError,
	FieldGroup,
	FieldContent,
	FieldTitle,
} from './Field'
import { Input } from '@rbee/ui/atoms/Input'
import { Label } from '@rbee/ui/atoms/Label'

const meta: Meta<typeof Field> = {
	title: 'Atoms/Field',
	component: Field,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		orientation: {
			control: 'select',
			options: ['vertical', 'horizontal', 'responsive'],
		},
	},
}

export default meta
type Story = StoryObj<typeof Field>

export const Default: Story = {
	render: () => (
		<Field className="w-96">
			<FieldContent>
				<FieldLabel htmlFor="email">Email</FieldLabel>
				<Input id="email" type="email" placeholder="you@example.com" />
				<FieldDescription>We'll never share your email with anyone else.</FieldDescription>
			</FieldContent>
		</Field>
	),
}

export const WithError: Story = {
	render: () => (
		<Field className="w-96" data-invalid="true">
			<FieldContent>
				<FieldLabel htmlFor="email-error">Email</FieldLabel>
				<Input id="email-error" type="email" placeholder="you@example.com" aria-invalid="true" />
				<FieldError>Please enter a valid email address.</FieldError>
			</FieldContent>
		</Field>
	),
}

export const WithHelp: Story = {
	render: () => (
		<Field className="w-96">
			<FieldContent>
				<FieldLabel htmlFor="password">Password</FieldLabel>
				<Input id="password" type="password" placeholder="Enter your password" />
				<FieldDescription>
					Password must be at least 8 characters long and contain uppercase, lowercase, and numbers.
				</FieldDescription>
			</FieldContent>
		</Field>
	),
}

export const Required: Story = {
	render: () => (
		<Field className="w-96">
			<FieldContent>
				<FieldLabel htmlFor="username">
					Username <span className="text-destructive">*</span>
				</FieldLabel>
				<Input id="username" placeholder="Enter your username" required />
				<FieldDescription>This field is required.</FieldDescription>
			</FieldContent>
		</Field>
	),
}
