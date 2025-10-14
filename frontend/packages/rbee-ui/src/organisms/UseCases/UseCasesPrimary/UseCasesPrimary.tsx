'use client'

import { SectionContainer } from '@rbee/ui/molecules'
import { UseCaseCard, type UseCaseCardProps } from '@rbee/ui/molecules/UseCaseCard'
import { Briefcase, Building, Code, GraduationCap, Home, Laptop, Server, Users } from 'lucide-react'
import Image from 'next/image'

const useCases: UseCaseCardProps[] = [
	{
		icon: Laptop,
		color: 'chart-2',
		title: 'The Solo Developer',
		scenario: 'Building a SaaS with AI, wants Claude-level coding without vendor lock-in.',
		solution: 'Run rbee on gaming PC + spare workstation; Llama 70B for code, SD for assets.',
		highlights: ['$0/mo inference', 'Full control', 'No rate limits'],
		anchor: 'developers',
	},
	{
		icon: Users,
		color: 'primary',
		title: 'The Small Team',
		scenario: '5-person startup spending ~$500/mo on AI APIs; needs to cut burn.',
		solution: 'Pool 3 workstations + 2 Macs (8 GPUs) into one rbee cluster.',
		highlights: ['~$6k/yr saved', 'Faster tokens', 'GDPR-friendly'],
		badge: 'Most Popular',
	},
	{
		icon: Home,
		color: 'chart-3',
		title: 'The Homelab Enthusiast',
		scenario: 'Has 4 GPUs collecting dust; wants to build AI agents for personal projects.',
		solution: 'Run rbee across homelab; build custom AI coder, docs generator, code reviewer.',
		highlights: ['Idle hardware → productive', 'Zero ongoing costs', 'Full customization'],
		anchor: 'homelab',
	},
	{
		icon: Building,
		color: 'chart-4',
		title: 'The Enterprise',
		scenario: "50-dev team; code can't leave network due to compliance.",
		solution: 'On-prem rbee with 20 GPUs; custom Rhai routing for data residency.',
		highlights: ['EU-only routing', 'Full audit trail', 'Zero external deps'],
		anchor: 'enterprise',
		badge: 'GDPR',
	},
	{
		icon: Briefcase,
		color: 'primary',
		title: 'The Freelance Developer',
		scenario: "Works on multiple client projects; can't share code with external APIs.",
		solution: 'Run rbee locally; all client code stays on machine; Llama for generation.',
		highlights: ['Client confidentiality', 'Professional AI tools', 'Zero subscriptions'],
	},
	{
		icon: GraduationCap,
		color: 'chart-2',
		title: 'The Research Lab',
		scenario: 'University lab with grant funding; limited budget for cloud services.',
		solution: 'Deploy rbee on lab GPU cluster; use grant for hardware, not subscriptions.',
		highlights: ['Maximize research budget', 'Reproducible experiments', 'No vendor lock-in'],
	},
	{
		icon: Code,
		color: 'chart-3',
		title: 'The Open Source Maintainer',
		scenario: "Maintains popular OSS projects; wants AI for reviews/docs but can't afford enterprise.",
		solution: 'Run rbee on personal hardware; build custom agents for PR reviews, docs, triage.',
		highlights: ['Sustainable AI tooling', 'Community-aligned', 'Zero ongoing costs'],
	},
	{
		icon: Server,
		color: 'chart-4',
		title: 'The GPU Provider',
		scenario: 'Has idle GPU hardware (former mining rig, gaming PC); wants to monetize.',
		solution: 'Join rbee marketplace (M3); set pricing and availability; earn passive income.',
		highlights: ['Passive income stream', 'Help the community', 'Control availability'],
	},
]

const filters = [
	{ label: 'All', anchor: '#use-cases' },
	{ label: 'Solo', anchor: '#developers' },
	{ label: 'Team', anchor: '#use-cases' },
	{ label: 'Enterprise', anchor: '#enterprise' },
	{ label: 'Research', anchor: '#use-cases' },
]

export function UseCasesPrimary() {
	const handleFilterClick = (anchor: string) => {
		const element = document.querySelector(anchor)
		if (element) {
			element.scrollIntoView({ behavior: 'smooth', block: 'start' })
		}
	}

	return (
		<SectionContainer title="Real Scenarios. Real Solutions." bgVariant="background">
			{/* Header block with eyebrow */}
			<div className="max-w-6xl mx-auto mb-8 animate-in fade-in duration-500">
				<p className="text-center text-sm text-muted-foreground mb-6">OpenAI-compatible · Your GPUs · Zero API fees</p>

				{/* Hero strip image */}
				<div className="relative overflow-hidden rounded-lg border border-border/60 mb-8">
					<Image
						src="/illustrations/usecases-grid-dark.svg"
						width={1920}
						height={640}
						priority
						alt="cinematic ultra-wide banner 16:5 showing three developer workstations in modern dark office connected by private AI network, shot from elevated 30-degree angle looking down at desk setup, LEFT THIRD: large gaming PC tower case with tempered glass side panel revealing two NVIDIA RTX 4090 graphics cards with visible PCB green circuit boards and silver heatsink fins, amber LED strips #f59e0b glowing along GPU edges, black case with subtle RGB accent lighting in teal, tower is approximately 20 inches tall sitting on dark wood desk, CENTER THIRD: professional workstation setup with dual 27-inch monitors in landscape orientation showing split screen, left monitor displays code editor with syntax highlighting in blue/green/yellow, right monitor shows terminal with streaming green text output, sleek aluminum monitor arms, black mechanical keyboard and mouse on desk, modern office chair partially visible, workstation has clean cable management, RIGHT THIRD: compact Mac Studio in silver aluminum finish approximately 4 inches tall, single 24-inch display showing design software interface, minimalist setup with wireless keyboard and trackpad, small desk plant in white ceramic pot, NETWORK VISUALIZATION: glowing neon teal #14b8a6 network lines connecting all three machines floating 6 inches above desk surface, lines form mesh topology with nodes at each machine, data packets visualized as small glowing dots traveling along the lines from left to right, network lines have soft glow effect with slight blur, lines are 2-3 pixels thick with brighter core and softer outer glow, FLOATING UI OVERLAYS: semi-transparent dark panels with rounded corners floating above the setup showing 'Private AI Cluster' in white text, '8 GPUs' with small GPU icon, '3 nodes' with network icon, '$0/mo API costs' in emerald green #10b981, panels have subtle drop shadow and 10% opacity dark background, BACKGROUND: deep navy blue #0f172a office wall with subtle texture, professional studio lighting with key light from upper left creating soft shadows, teal accent strip lighting along wall edges, background has subtle radial gradient getting darker toward edges, shallow depth of field with background softly blurred while workstations are tack sharp, DESK SURFACE: dark walnut wood finish with matte texture, clean and organized with minimal cable clutter, soft reflections of monitor glow and GPU lighting on glossy surface, LIGHTING: each workstation has its own lighting character - gaming PC has warm amber glow from GPUs, center workstation has cool blue-white monitor glow, Mac Studio has neutral white light, overall scene has cool color temperature with warm accents, professional photography aesthetic similar to tech company marketing materials or Verge hardware reviews, shot with wide-angle lens 24mm equivalent creating slight perspective distortion, f/4 aperture for good depth while keeping all machines sharp, cinematic color grading with lifted blacks, teal and amber color palette, slight vignette darkening edges, conveys professional distributed on-premises AI infrastructure, private network without cloud dependencies, developer-focused setup, enterprise-grade but accessible, mood is productive and empowering, 1920x640 pixels 16:5 aspect ratio, high detail on hardware showing logos and model numbers, network lines should have realistic glow with falloff, UI overlays should be legible but not distracting"
						className="w-full h-32 md:h-40 object-cover"
					/>
				</div>

				{/* Filter pills */}
				<nav
					aria-label="Filter use cases"
					className="flex flex-wrap items-center justify-center gap-2 animate-in slide-in-from-top-2 duration-500 delay-100"
				>
					{filters.map((filter) => (
						<button
							key={filter.label}
							onClick={() => handleFilterClick(filter.anchor)}
							className="inline-flex items-center rounded-full border border-border/60 bg-card px-4 py-2 text-sm font-medium text-foreground hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
						>
							{filter.label}
						</button>
					))}
				</nav>
			</div>

			{/* Responsive grid: 1 col mobile, 2 cols tablet+ */}
			<div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8 max-w-6xl mx-auto">
				{useCases.map((useCase, index) => (
					<UseCaseCard key={useCase.title} {...useCase} style={{ animationDelay: `${index * 60}ms` }} />
				))}
			</div>
		</SectionContainer>
	)
}
