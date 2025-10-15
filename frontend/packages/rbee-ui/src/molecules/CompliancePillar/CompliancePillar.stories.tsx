import type { Meta, StoryObj } from '@storybook/react'
import { Globe, Shield, Lock } from 'lucide-react'
import { CompliancePillar } from './CompliancePillar'

const meta: Meta<typeof CompliancePillar> = {
	title: 'Molecules/CompliancePillar',
	component: CompliancePillar,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The CompliancePillar molecule displays a compliance standard with its requirements checklist and optional callout content. Used to communicate regulatory compliance and security certifications.

## Composition
This molecule is composed of:
- **IconPlate**: Large icon representing the standard
- **Title**: Standard name (e.g., "GDPR")
- **Subtitle**: Standard type (e.g., "EU Regulation")
- **Checklist**: List of compliance requirements with check icons
- **Callout**: Optional additional content

## When to Use
- Displaying compliance certifications
- Explaining regulatory standards
- Security and privacy pages
- Enterprise trust-building content

## Used In
- **EnterpriseCompliance**: Shows GDPR, SOC2, ISO 27001 compliance
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			control: false,
			description: 'Icon element (e.g., Globe, Shield, Lock)',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
		title: {
			control: 'text',
			description: 'Standard name',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		subtitle: {
			control: 'text',
			description: 'Standard type or jurisdiction',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		checklist: {
			control: 'object',
			description: 'List of compliance requirements',
			table: {
				type: { summary: 'string[]' },
				category: 'Content',
			},
		},
		callout: {
			control: false,
			description: 'Optional callout content',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof CompliancePillar>

export const Default: Story = {
	args: {
		icon: <Globe className="h-6 w-6" />,
		title: 'GDPR',
		subtitle: 'EU Regulation',
		checklist: [
			'Data processing agreements',
			'Right to erasure',
			'Data portability',
			'Privacy by design',
		],
	},
}

export const WithIcon: Story = {
	args: {
		icon: <Shield className="h-6 w-6" />,
		title: 'SOC 2 Type II',
		subtitle: 'US Standard',
		checklist: [
			'Security controls',
			'Availability monitoring',
			'Processing integrity',
			'Confidentiality measures',
			'Privacy safeguards',
		],
	},
}

export const WithList: Story = {
	args: {
		icon: <Lock className="h-6 w-6" />,
		title: 'ISO 27001',
		subtitle: 'International Standard',
		checklist: [
			'Information security management',
			'Risk assessment procedures',
			'Access control policies',
			'Incident response plans',
			'Business continuity',
			'Compliance audits',
		],
	},
}

export const InComplianceContext: Story = {
	render: () => (
		<div className="w-full max-w-6xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: CompliancePillar in EnterpriseCompliance organism
			</div>
			<div className="grid gap-6 md:grid-cols-3">
				<CompliancePillar
					icon={<Globe className="h-6 w-6" />}
					title="GDPR"
					subtitle="EU Regulation"
					checklist={[
						'Data processing agreements',
						'Right to erasure',
						'Data portability',
						'Privacy by design',
					]}
					callout={
						<div className="rounded-lg bg-primary/10 p-3 text-xs text-muted-foreground">
							<strong className="text-foreground">Dutch hosting:</strong> All data stays in NL datacenters
						</div>
					}
				/>
				<CompliancePillar
					icon={<Shield className="h-6 w-6" />}
					title="SOC 2 Type II"
					subtitle="US Standard"
					checklist={[
						'Security controls',
						'Availability monitoring',
						'Processing integrity',
						'Confidentiality measures',
					]}
				/>
				<CompliancePillar
					icon={<Lock className="h-6 w-6" />}
					title="ISO 27001"
					subtitle="International Standard"
					checklist={[
						'Information security management',
						'Risk assessment procedures',
						'Access control policies',
						'Incident response plans',
					]}
				/>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'CompliancePillar as used in the EnterpriseCompliance organism, showing three major compliance standards.',
			},
		},
	},
}
