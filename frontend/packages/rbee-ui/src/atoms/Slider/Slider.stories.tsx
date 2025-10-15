// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { Slider } from './Slider'

const meta: Meta<typeof Slider> = {
	title: 'Atoms/Slider',
	component: Slider,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		min: {
			control: 'number',
			description: 'Minimum value',
		},
		max: {
			control: 'number',
			description: 'Maximum value',
		},
		step: {
			control: 'number',
			description: 'Step increment',
		},
		disabled: {
			control: 'boolean',
		},
	},
}

export default meta
type Story = StoryObj<typeof Slider>

/**
 * ## Overview
 * Slider allows users to select a value or range from a continuous or discrete set.
 * Built on Radix UI Slider primitive with keyboard and touch support.
 *
 * ## When to Use
 * - Select numeric values
 * - Adjust settings
 * - Filter ranges
 * - Volume/brightness controls
 *
 * ## Used In
 * - ProvidersEarningsCalculator (primary use case)
 */

export const Default: Story = {
	args: {
		defaultValue: [50],
		max: 100,
	},
}

export const WithLabels: Story = {
	render: () => {
		const [value, setValue] = useState([50])
		return (
			<div className="w-[300px] space-y-4">
				<div>
					<label className="text-sm font-medium mb-2 block">Volume</label>
					<Slider value={value} onValueChange={setValue} max={100} />
					<div className="flex justify-between text-xs text-muted-foreground mt-2">
						<span>0</span>
						<span>50</span>
						<span>100</span>
					</div>
				</div>
			</div>
		)
	},
}

export const WithValue: Story = {
	render: () => {
		const [value, setValue] = useState([25])
		return (
			<div className="w-[300px] space-y-4">
				<div className="flex justify-between items-center mb-2">
					<label className="text-sm font-medium">GPU Count</label>
					<span className="text-sm font-semibold">{value[0]}</span>
				</div>
				<Slider value={value} onValueChange={setValue} min={1} max={100} step={1} />
			</div>
		)
	},
}

export const InEarningsCalculator: Story = {
	render: () => {
		const [gpuCount, setGpuCount] = useState([4])
		const [hoursPerDay, setHoursPerDay] = useState([12])
		const [pricePerHour, setPricePerHour] = useState([2.5])

		const dailyEarnings = gpuCount[0] * hoursPerDay[0] * pricePerHour[0]
		const monthlyEarnings = dailyEarnings * 30

		return (
			<div className="max-w-md p-6 border rounded-lg">
				<h3 className="text-xl font-bold mb-6">Earnings Calculator</h3>

				<div className="space-y-6">
					<div>
						<div className="flex justify-between items-center mb-2">
							<label className="text-sm font-medium">Number of GPUs</label>
							<span className="text-sm font-semibold">{gpuCount[0]}</span>
						</div>
						<Slider value={gpuCount} onValueChange={setGpuCount} min={1} max={20} step={1} />
					</div>

					<div>
						<div className="flex justify-between items-center mb-2">
							<label className="text-sm font-medium">Hours per Day</label>
							<span className="text-sm font-semibold">{hoursPerDay[0]}h</span>
						</div>
						<Slider value={hoursPerDay} onValueChange={setHoursPerDay} min={1} max={24} step={1} />
					</div>

					<div>
						<div className="flex justify-between items-center mb-2">
							<label className="text-sm font-medium">Price per Hour</label>
							<span className="text-sm font-semibold">€{pricePerHour[0].toFixed(2)}</span>
						</div>
						<Slider value={pricePerHour} onValueChange={setPricePerHour} min={0.5} max={10} step={0.5} />
					</div>

					<div className="pt-4 border-t space-y-2">
						<div className="flex justify-between">
							<span className="text-sm text-muted-foreground">Daily Earnings</span>
							<span className="font-semibold">€{dailyEarnings.toFixed(2)}</span>
						</div>
						<div className="flex justify-between">
							<span className="text-sm text-muted-foreground">Monthly Earnings</span>
							<span className="text-lg font-bold text-primary">€{monthlyEarnings.toFixed(2)}</span>
						</div>
					</div>
				</div>
			</div>
		)
	},
}
