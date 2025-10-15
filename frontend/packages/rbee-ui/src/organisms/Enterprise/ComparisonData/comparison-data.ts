import type { Provider, Row } from '@rbee/ui/molecules'

export const PROVIDERS: Provider[] = [
	{ key: 'rbee', label: 'rbee (Self-Hosted)', accent: true },
	{ key: 'openai', label: 'OpenAI / Anthropic' },
	{ key: 'azure', label: 'Azure OpenAI' },
] as const

export const FEATURES: Row[] = [
	{
		feature: 'Data Sovereignty',
		values: { rbee: true, openai: false, azure: 'Partial' },
	},
	{
		feature: 'EU-Only Residency',
		values: { rbee: true, openai: false, azure: 'Partial' },
	},
	{
		feature: 'GDPR Compliant',
		values: { rbee: true, openai: 'Partial', azure: 'Partial' },
	},
	{
		feature: 'Immutable Audit Logs',
		values: { rbee: true, openai: false, azure: false },
	},
	{
		feature: '7-Year Audit Retention',
		values: { rbee: true, openai: false, azure: false },
	},
	{
		feature: 'SOC2 Type II Ready',
		values: { rbee: true, openai: true, azure: true },
	},
	{
		feature: 'ISO 27001 Aligned',
		values: { rbee: true, openai: true, azure: true },
	},
	{
		feature: 'Zero US Cloud Dependencies',
		values: { rbee: true, openai: false, azure: false },
	},
	{
		feature: 'On-Premises Deployment',
		values: { rbee: true, openai: false, azure: false },
	},
	{
		feature: 'Complete Control',
		values: { rbee: true, openai: false, azure: 'Partial' },
	},
	{
		feature: 'Custom SLAs',
		values: { rbee: true, openai: false, azure: true },
	},
	{
		feature: 'White-Label Option',
		values: { rbee: true, openai: false, azure: false },
	},
]
