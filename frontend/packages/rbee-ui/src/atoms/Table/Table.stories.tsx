import type { Meta, StoryObj } from '@storybook/react'
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from './Table'

const meta: Meta<typeof Table> = {
	title: 'Atoms/Table',
	component: Table,
	parameters: {
		layout: 'padded',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Table>

const invoices = [
	{ invoice: 'INV001', status: 'Paid', method: 'Credit Card', amount: '$250.00' },
	{ invoice: 'INV002', status: 'Pending', method: 'PayPal', amount: '$150.00' },
	{ invoice: 'INV003', status: 'Unpaid', method: 'Bank Transfer', amount: '$350.00' },
	{ invoice: 'INV004', status: 'Paid', method: 'Credit Card', amount: '$450.00' },
	{ invoice: 'INV005', status: 'Paid', method: 'PayPal', amount: '$550.00' },
]

export const Default: Story = {
	render: () => (
		<Table>
			<TableCaption>A list of your recent invoices.</TableCaption>
			<TableHeader>
				<TableRow>
					<TableHead>Invoice</TableHead>
					<TableHead>Status</TableHead>
					<TableHead>Method</TableHead>
					<TableHead className="text-right">Amount</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{invoices.map((invoice) => (
					<TableRow key={invoice.invoice}>
						<TableCell className="font-medium">{invoice.invoice}</TableCell>
						<TableCell>{invoice.status}</TableCell>
						<TableCell>{invoice.method}</TableCell>
						<TableCell className="text-right">{invoice.amount}</TableCell>
					</TableRow>
				))}
			</TableBody>
		</Table>
	),
}

export const Striped: Story = {
	render: () => (
		<Table>
			<TableHeader>
				<TableRow>
					<TableHead>Name</TableHead>
					<TableHead>Email</TableHead>
					<TableHead>Role</TableHead>
					<TableHead>Status</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{[
					{ name: 'John Doe', email: 'john@example.com', role: 'Admin', status: 'Active' },
					{ name: 'Jane Smith', email: 'jane@example.com', role: 'User', status: 'Active' },
					{ name: 'Bob Johnson', email: 'bob@example.com', role: 'User', status: 'Inactive' },
					{ name: 'Alice Williams', email: 'alice@example.com', role: 'Editor', status: 'Active' },
				].map((user, i) => (
					<TableRow key={user.email} className={i % 2 === 0 ? 'bg-muted/50' : ''}>
						<TableCell className="font-medium">{user.name}</TableCell>
						<TableCell>{user.email}</TableCell>
						<TableCell>{user.role}</TableCell>
						<TableCell>{user.status}</TableCell>
					</TableRow>
				))}
			</TableBody>
		</Table>
	),
}

export const WithSorting: Story = {
	render: () => (
		<Table>
			<TableHeader>
				<TableRow>
					<TableHead className="cursor-pointer hover:bg-muted/50">
						Name <span className="ml-1">↕</span>
					</TableHead>
					<TableHead className="cursor-pointer hover:bg-muted/50">
						Age <span className="ml-1">↕</span>
					</TableHead>
					<TableHead className="cursor-pointer hover:bg-muted/50">
						Department <span className="ml-1">↕</span>
					</TableHead>
					<TableHead className="cursor-pointer text-right hover:bg-muted/50">
						Salary <span className="ml-1">↕</span>
					</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{[
					{ name: 'Alice', age: 28, dept: 'Engineering', salary: '$95,000' },
					{ name: 'Bob', age: 35, dept: 'Marketing', salary: '$75,000' },
					{ name: 'Charlie', age: 42, dept: 'Sales', salary: '$85,000' },
					{ name: 'Diana', age: 31, dept: 'Engineering', salary: '$105,000' },
				].map((employee) => (
					<TableRow key={employee.name}>
						<TableCell className="font-medium">{employee.name}</TableCell>
						<TableCell>{employee.age}</TableCell>
						<TableCell>{employee.dept}</TableCell>
						<TableCell className="text-right">{employee.salary}</TableCell>
					</TableRow>
				))}
			</TableBody>
		</Table>
	),
}

export const WithPagination: Story = {
	render: () => (
		<div className="space-y-4">
			<Table>
				<TableHeader>
					<TableRow>
						<TableHead>ID</TableHead>
						<TableHead>Product</TableHead>
						<TableHead>Category</TableHead>
						<TableHead className="text-right">Price</TableHead>
					</TableRow>
				</TableHeader>
				<TableBody>
					{[
						{ id: '001', product: 'Laptop', category: 'Electronics', price: '$1,299' },
						{ id: '002', product: 'Mouse', category: 'Accessories', price: '$29' },
						{ id: '003', product: 'Keyboard', category: 'Accessories', price: '$79' },
						{ id: '004', product: 'Monitor', category: 'Electronics', price: '$399' },
						{ id: '005', product: 'Headphones', category: 'Audio', price: '$199' },
					].map((product) => (
						<TableRow key={product.id}>
							<TableCell className="font-medium">{product.id}</TableCell>
							<TableCell>{product.product}</TableCell>
							<TableCell>{product.category}</TableCell>
							<TableCell className="text-right">{product.price}</TableCell>
						</TableRow>
					))}
				</TableBody>
			</Table>
			<div className="flex items-center justify-between">
				<div className="text-sm text-muted-foreground">Showing 1-5 of 50 results</div>
				<div className="flex gap-2">
					<button className="rounded-md border px-3 py-1 text-sm hover:bg-muted">Previous</button>
					<button className="rounded-md border px-3 py-1 text-sm hover:bg-muted">Next</button>
				</div>
			</div>
		</div>
	),
}
