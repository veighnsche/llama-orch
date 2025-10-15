import type { Meta, StoryObj } from '@storybook/react'
import { TestimonialCard } from './TestimonialCard'

const meta: Meta<typeof TestimonialCard> = {
	title: 'Molecules/TestimonialCard',
	component: TestimonialCard,
	parameters: {
		layout: 'padded',
		docs: {
			description: {
				component: `
## Overview
TestimonialCard displays customer testimonials with support for both new API (structured testimonial data) and legacy API (individual props). It includes sector icons, ratings, verification badges, and responsive layouts.

## Composition
This molecule is composed of:
- **Sector icon**: Visual indicator of customer industry
- **Author info**: Name, role, organization
- **Rating stars**: Optional 1-5 star rating
- **Quote**: Customer testimonial text
- **Verification badge**: Optional verified customer/payout badge

## When to Use
- Testimonials sections (social proof)
- Case studies (customer stories)
- Reviews (product feedback)
- Success stories (customer wins)

## Variants
- **New API**: Accepts structured testimonial object with sector, rating, payout
- **Legacy API**: Accepts individual props for backward compatibility
- **With/without avatar**: Optional avatar image or gradient
- **With/without rating**: Optional star rating display

## Used In Commercial Site
Used in:
- TestimonialsRail (scrolling testimonials)
- TestimonialsSection (grid of testimonials)
- EnterpriseTestimonials (enterprise customer stories)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		t: {
			control: 'object',
			description: 'Testimonial data (new API)',
			table: {
				type: { summary: 'Testimonial' },
				category: 'Content',
			},
		},
		showVerified: {
			control: 'boolean',
			description: 'Show verified badge',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Appearance',
			},
		},
		delayIndex: {
			control: 'number',
			description: 'Animation delay index',
			table: {
				type: { summary: 'number' },
				defaultValue: { summary: '0' },
				category: 'Animation',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof TestimonialCard>

const sampleTestimonial = {
	id: 'sample-1',
	name: 'Dr. Sarah van den Berg',
	role: 'Chief Data Officer',
	org: 'Amsterdam Medical Center',
	sector: 'healthcare' as const,
	quote:
		'rbee allows us to process patient data with complete GDPR compliance. The Dutch data sovereignty gives us peace of mind that no patient information ever leaves the Netherlands.',
	rating: 5 as const,
}

const providerTestimonial = {
	id: 'sample-2',
	name: 'Mark de Vries',
	role: 'GPU Provider',
	org: 'Rotterdam',
	sector: 'provider' as const,
	quote:
		'I earn €150-200/month by sharing my gaming PC when I\'m not using it. Setup took 10 minutes, and payments are automatic.',
	payout: '€150-200/mo',
	rating: 5 as const,
}

export const Default: Story = {
	args: {
		t: sampleTestimonial,
		showVerified: true,
	},
}

export const WithAvatar: Story = {
	args: {
		name: 'Emma Jansen',
		role: 'Legal Counsel',
		quote:
			'The GDPR compliance documentation is thorough and well-maintained. We can confidently use rbee for processing sensitive legal documents.',
		avatar: { from: 'primary', to: 'chart-2' },
		company: { name: 'Jansen & Partners Advocaten' },
		verified: true,
		rating: 5,
	},
}

export const WithRating: Story = {
	render: () => (
		<div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl">
			<TestimonialCard
				t={{
					id: 'rating-5',
					name: 'Jan Bakker',
					role: 'CTO',
					org: 'FinTech Startup',
					sector: 'finance',
					quote: 'Outstanding service. The performance is incredible and the support team is very responsive.',
					rating: 5,
				}}
			/>
			<TestimonialCard
				t={{
					id: 'rating-4',
					name: 'Lisa Vermeer',
					role: 'DevOps Engineer',
					org: 'Tech Company',
					sector: 'provider',
					quote: 'Great platform overall. Setup was straightforward and documentation is clear.',
					rating: 4,
				}}
			/>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Testimonials with different star ratings.',
			},
		},
	},
}

export const InRailContext: Story = {
	render: () => (
		<div className="w-full">
			<div className="mb-6 text-center">
				<h2 className="text-3xl font-bold mb-2">What Our Customers Say</h2>
				<p className="text-muted-foreground">Trusted by enterprises and developers across the Netherlands</p>
			</div>
			<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
				<TestimonialCard t={sampleTestimonial} delayIndex={0} />
				<TestimonialCard t={providerTestimonial} delayIndex={1} />
				<TestimonialCard
					t={{
						id: 'sample-3',
						name: 'Peter Smit',
						role: 'Senior Developer',
						org: 'Government Agency',
						sector: 'government',
						quote: 'The API is well-designed and the SDK makes integration easy. Excellent developer experience.',
						rating: 5,
					}}
					delayIndex={2}
				/>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'TestimonialCard as used in TestimonialsRail, showing multiple testimonials in a grid.',
			},
		},
	},
}
