export type Sector = 'finance' | 'healthcare' | 'legal' | 'government' | 'provider'

export interface Testimonial {
	id: string
	name: string
	role: string
	org?: string
	sector: Sector
	payout?: string
	rating?: number
	quote: string
	avatar?: string
}

export interface TestimonialStat {
	id: string
	value: string
	label: string
	note?: string
}

export const TESTIMONIALS: Testimonial[] = [
	{
		id: 'marcus',
		name: 'Marcus T.',
		role: 'Gaming PC Owner',
		org: '',
		sector: 'provider',
		payout: '€160/mo',
		rating: 5,
		quote: 'My RTX 4090 used to sit idle. Now it brings in €160/mo—set up in under 10 minutes.',
	},
	{
		id: 'sarah',
		name: 'Sarah K.',
		role: 'Homelab Enthusiast',
		sector: 'provider',
		payout: '€420/mo',
		rating: 5,
		quote: 'Four homelab GPUs now cover electricity plus profit—€420/mo combined.',
	},
	{
		id: 'david',
		name: 'David L.',
		role: 'Former Miner',
		sector: 'provider',
		payout: '€780/mo',
		rating: 5,
		quote: 'After ETH went PoS, my rig gathered dust. With rbee I average €780/mo—better than mining.',
	},
	{
		id: 'klaus',
		name: 'Dr. Klaus M.',
		role: 'CTO',
		org: 'European Bank',
		sector: 'finance',
		rating: 5,
		quote: 'PCI-DSS blocked external AI. On-prem rbee + immutable logs → SOC2 audit with zero findings.',
	},
	{
		id: 'anna',
		name: 'Anna S.',
		role: 'DPO',
		org: 'Healthcare Provider',
		sector: 'healthcare',
		rating: 5,
		quote: 'HIPAA/GDPR were non-negotiable. EU-only deploy + 7-year retention gave us confidence to ship.',
	},
	{
		id: 'michael',
		name: 'Michael R.',
		role: 'Managing Partner',
		org: 'Law Firm',
		sector: 'legal',
		rating: 5,
		quote: 'Attorney-client privilege demanded on-prem + zero external transfer. Client confidentiality protected.',
	},
]

export const TESTIMONIAL_STATS: TestimonialStat[] = [
	{ id: 'gdpr', value: '100%', label: 'GDPR Compliant' },
	{ id: 'retention', value: '7 Years', label: 'Audit Retention' },
	{ id: 'violations', value: 'Zero', label: 'Compliance Violations' },
	{ id: 'support', value: '24/7', label: 'Enterprise Support' },
]
