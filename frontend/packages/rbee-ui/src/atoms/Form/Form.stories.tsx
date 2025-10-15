import type { Meta, StoryObj } from '@storybook/react'
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from './Form'
import { Input } from '@rbee/ui/atoms/Input'
import { Button } from '@rbee/ui/atoms/Button'
import { useForm } from 'react-hook-form'

const meta: Meta<typeof Form> = {
	title: 'Atoms/Form',
	component: Form,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Form>

export const Default: Story = {
	render: () => {
		const form = useForm({
			defaultValues: {
				username: '',
				email: '',
			},
		})

		return (
			<Form {...form}>
				<form onSubmit={form.handleSubmit((data) => console.log(data))} className="w-96 space-y-6">
					<FormField
						control={form.control}
						name="username"
						render={({ field }) => (
							<FormItem>
								<FormLabel>Username</FormLabel>
								<FormControl>
									<Input placeholder="Enter your username" {...field} />
								</FormControl>
								<FormDescription>This is your public display name.</FormDescription>
								<FormMessage />
							</FormItem>
						)}
					/>
					<FormField
						control={form.control}
						name="email"
						render={({ field }) => (
							<FormItem>
								<FormLabel>Email</FormLabel>
								<FormControl>
									<Input type="email" placeholder="you@example.com" {...field} />
								</FormControl>
								<FormMessage />
							</FormItem>
						)}
					/>
					<Button type="submit">Submit</Button>
				</form>
			</Form>
		)
	},
}

export const WithValidation: Story = {
	render: () => {
		const form = useForm({
			defaultValues: {
				email: '',
				password: '',
			},
		})

		return (
			<Form {...form}>
				<form onSubmit={form.handleSubmit((data) => console.log(data))} className="w-96 space-y-6">
					<FormField
						control={form.control}
						name="email"
						rules={{
							required: 'Email is required',
							pattern: {
								value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
								message: 'Invalid email address',
							},
						}}
						render={({ field }) => (
							<FormItem>
								<FormLabel>Email</FormLabel>
								<FormControl>
									<Input type="email" placeholder="you@example.com" {...field} />
								</FormControl>
								<FormMessage />
							</FormItem>
						)}
					/>
					<FormField
						control={form.control}
						name="password"
						rules={{
							required: 'Password is required',
							minLength: {
								value: 8,
								message: 'Password must be at least 8 characters',
							},
						}}
						render={({ field }) => (
							<FormItem>
								<FormLabel>Password</FormLabel>
								<FormControl>
									<Input type="password" placeholder="Enter your password" {...field} />
								</FormControl>
								<FormMessage />
							</FormItem>
						)}
					/>
					<Button type="submit">Sign In</Button>
				</form>
			</Form>
		)
	},
}

export const MultiStep: Story = {
	render: () => {
		const form = useForm({
			defaultValues: {
				firstName: '',
				lastName: '',
				company: '',
			},
		})

		return (
			<Form {...form}>
				<form onSubmit={form.handleSubmit((data) => console.log(data))} className="w-96 space-y-6">
					<div className="space-y-4">
						<h3 className="text-lg font-semibold">Step 1: Personal Information</h3>
						<FormField
							control={form.control}
							name="firstName"
							render={({ field }) => (
								<FormItem>
									<FormLabel>First Name</FormLabel>
									<FormControl>
										<Input placeholder="John" {...field} />
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
						<FormField
							control={form.control}
							name="lastName"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Last Name</FormLabel>
									<FormControl>
										<Input placeholder="Doe" {...field} />
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
					</div>
					<div className="space-y-4">
						<h3 className="text-lg font-semibold">Step 2: Company Information</h3>
						<FormField
							control={form.control}
							name="company"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Company</FormLabel>
									<FormControl>
										<Input placeholder="Acme Inc." {...field} />
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
					</div>
					<Button type="submit">Continue</Button>
				</form>
			</Form>
		)
	},
}

export const WithSections: Story = {
	render: () => {
		const form = useForm({
			defaultValues: {
				name: '',
				email: '',
				bio: '',
			},
		})

		return (
			<Form {...form}>
				<form onSubmit={form.handleSubmit((data) => console.log(data))} className="w-96 space-y-8">
					<div className="space-y-4">
						<div>
							<h3 className="text-lg font-semibold">Profile</h3>
							<p className="text-sm text-muted-foreground">Update your profile information</p>
						</div>
						<FormField
							control={form.control}
							name="name"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Name</FormLabel>
									<FormControl>
										<Input placeholder="Your name" {...field} />
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
						<FormField
							control={form.control}
							name="bio"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Bio</FormLabel>
									<FormControl>
										<Input placeholder="Tell us about yourself" {...field} />
									</FormControl>
									<FormDescription>Brief description for your profile.</FormDescription>
									<FormMessage />
								</FormItem>
							)}
						/>
					</div>
					<div className="space-y-4">
						<div>
							<h3 className="text-lg font-semibold">Contact</h3>
							<p className="text-sm text-muted-foreground">How can we reach you?</p>
						</div>
						<FormField
							control={form.control}
							name="email"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Email</FormLabel>
									<FormControl>
										<Input type="email" placeholder="you@example.com" {...field} />
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
					</div>
					<Button type="submit">Save Changes</Button>
				</form>
			</Form>
		)
	},
}
