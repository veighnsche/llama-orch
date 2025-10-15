import type { Meta, StoryObj } from '@storybook/react'
import { Shield, Lock, CheckCircle, FileCheck } from 'lucide-react'
import { ComplianceChip } from './ComplianceChip'

const meta: Meta<typeof ComplianceChip> = {
	title: 'Molecules/ComplianceChip',
	component: ComplianceChip,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
ComplianceChip is a compact badge molecule for displaying compliance certifications and security indicators. It features optional icons and hover effects.

## Composition
This molecule is composed of:
- **Container**: Rounded chip with border and background
- **Icon**: Optional icon element (typically Lucide icons)
- **Label**: Compliance certification or security indicator text
- **Hover effect**: Subtle border and background transitions

## When to Use
- Compliance sections (GDPR, ISO, SOC2)
- Security badges (encryption, certifications)
- Trust indicators (verified, certified)
- Feature highlights (compliance features)
- Enterprise pages (security credentials)

## Variants
- **With icon**: Icon + text
- **Without icon**: Text only
- **Custom styling**: Additional className support

## Used In Commercial Site
Used in:
- EnterpriseHero (compliance badges)
- SecuritySection (security certifications)
- ComplianceSection (regulatory compliance)
- Footer (trust indicators)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			control: false,
			description: 'Icon element (e.g., Lucide icon)',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
		children: {
			control: 'text',
			description: 'Chip label',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
		ariaLabel: {
			control: 'text',
			description: 'Accessible label for screen readers',
			table: {
				type: { summary: 'string' },
				category: 'Accessibility',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof ComplianceChip>

export const Default: Story = {
	args: {
		icon: <Shield className="w-3 h-3" />,
		children: 'GDPR Compliant',
		ariaLabel: 'GDPR Compliant certification',
	},
}

export const AllTypes: Story = {
	render: () => (
		<div className="flex flex-wrap gap-3">
			<ComplianceChip icon={<Shield className="w-3 h-3" />} ariaLabel="GDPR Compliant">
				GDPR Compliant
			</ComplianceChip>
			<ComplianceChip icon={<Lock className="w-3 h-3" />} ariaLabel="ISO 27001 Certified">
				ISO 27001
			</ComplianceChip>
			<ComplianceChip icon={<CheckCircle className="w-3 h-3" />} ariaLabel="SOC 2 Type II">
				SOC 2 Type II
			</ComplianceChip>
			<ComplianceChip icon={<FileCheck className="w-3 h-3" />} ariaLabel="AVG Compliant">
				AVG Compliant
			</ComplianceChip>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Common compliance and security certifications.',
			},
		},
	},
}

export const WithIcon: Story = {
	render: () => (
		<div className="space-y-6">
			<div>
				<h3 className="text-sm font-semibold mb-3">With Icons</h3>
				<div className="flex flex-wrap gap-3">
					<ComplianceChip icon={<Shield className="w-3 h-3" />}>GDPR</ComplianceChip>
					<ComplianceChip icon={<Lock className="w-3 h-3" />}>ISO 27001</ComplianceChip>
					<ComplianceChip icon={<CheckCircle className="w-3 h-3" />}>SOC 2</ComplianceChip>
					<ComplianceChip icon={<FileCheck className="w-3 h-3" />}>AVG</ComplianceChip>
				</div>
			</div>
			<div>
				<h3 className="text-sm font-semibold mb-3">Without Icons</h3>
				<div className="flex flex-wrap gap-3">
					<ComplianceChip>GDPR Compliant</ComplianceChip>
					<ComplianceChip>ISO 27001 Certified</ComplianceChip>
					<ComplianceChip>SOC 2 Type II</ComplianceChip>
					<ComplianceChip>AVG Compliant</ComplianceChip>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Compliance chips with and without icons.',
			},
		},
	},
}

export const InEnterpriseContext: Story = {
	render: () => (
		<div className="w-full max-w-4xl">
			<div className="mb-8 text-center">
				<h2 className="text-3xl font-bold mb-2">Enterprise-Grade Security</h2>
				<p className="text-muted-foreground">Certified and compliant with international standards</p>
			</div>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
				<div className="p-6 bg-card rounded-lg border">
					<div className="flex items-center gap-3 mb-4">
						<div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
							<Shield className="w-6 h-6 text-primary" />
						</div>
						<h3 className="font-semibold text-lg">Data Protection</h3>
					</div>
					<p className="text-sm text-muted-foreground mb-4">
						Full compliance with European data protection regulations
					</p>
					<div className="flex flex-wrap gap-2">
						<ComplianceChip icon={<Shield className="w-3 h-3" />}>GDPR</ComplianceChip>
						<ComplianceChip icon={<FileCheck className="w-3 h-3" />}>AVG</ComplianceChip>
						<ComplianceChip icon={<CheckCircle className="w-3 h-3" />}>NIS2</ComplianceChip>
					</div>
				</div>
				<div className="p-6 bg-card rounded-lg border">
					<div className="flex items-center gap-3 mb-4">
						<div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
							<Lock className="w-6 h-6 text-primary" />
						</div>
						<h3 className="font-semibold text-lg">Security Standards</h3>
					</div>
					<p className="text-sm text-muted-foreground mb-4">
						Certified against international security frameworks
					</p>
					<div className="flex flex-wrap gap-2">
						<ComplianceChip icon={<Lock className="w-3 h-3" />}>ISO 27001</ComplianceChip>
						<ComplianceChip icon={<CheckCircle className="w-3 h-3" />}>SOC 2 Type II</ComplianceChip>
						<ComplianceChip icon={<Shield className="w-3 h-3" />}>ISO 27017</ComplianceChip>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'ComplianceChip as used in EnterpriseHero, showing security certifications in feature cards.',
			},
		},
	},
}
