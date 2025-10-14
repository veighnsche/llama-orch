import { Button } from '@rbee/ui/atoms/Button'
import { cn } from '@rbee/ui/utils'
import { Cpu, Gamepad2, Monitor, Server } from 'lucide-react'
import Image from 'next/image'
import * as React from 'react'

// ============================================================================
// Types
// ============================================================================

export type Case = {
	icon: React.ReactNode
	title: string
	subtitle?: string
	quote: string
	facts: { label: string; value: string }[]
	image?: { src: string; alt: string }
	highlight?: string
}

export type UseCasesSectionProps = {
	kicker?: string
	title: string
	subtitle?: string
	cases: Case[]
	ctas?: {
		primary?: { label: string; href: string }
		secondary?: { label: string; href: string }
	}
	className?: string
}

// ============================================================================
// CaseCard Molecule
// ============================================================================

function CaseCard({ caseData, index }: { caseData: Case; index: number }) {
	const delays = ['delay-75', 'delay-150', 'delay-200', 'delay-300']
	const delay = delays[index % delays.length]

	return (
		<div
			className={cn(
				'group min-h-[320px] rounded-2xl border border-border/70 bg-gradient-to-b from-card/70 to-background/60 p-6 backdrop-blur transition-transform hover:translate-y-0.5 supports-[backdrop-filter]:bg-background/60 sm:p-7',
				'animate-in fade-in slide-in-from-bottom-2',
				delay,
			)}
		>
			{/* Header row */}
			<div className="mb-4 flex items-center gap-4">
				<div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-primary/10 transition-transform group-hover:scale-[1.02]">
					{React.cloneElement(caseData.icon as React.ReactElement<any>, {
						className: 'h-7 w-7 text-primary',
						'aria-hidden': true,
					})}
				</div>
				<div className="flex-1">
					<h3 className="text-lg font-semibold text-foreground">{caseData.title}</h3>
					{caseData.subtitle && <div className="text-xs text-muted-foreground">{caseData.subtitle}</div>}
				</div>
				{caseData.image && (
					<Image
						src={caseData.image.src}
						width={48}
						height={48}
						className="hidden rounded-full border border-border/70 sm:block"
						alt={caseData.image.alt}
					/>
				)}
			</div>

			{/* Optional highlight badge */}
			{caseData.highlight && (
				<div className="mb-3 inline-flex items-center rounded-full bg-primary/10 px-2.5 py-1 text-[11px] text-primary">
					{caseData.highlight}
				</div>
			)}

			{/* Quote block */}
			<p className="relative mb-4 text-pretty leading-relaxed text-muted-foreground">
				<span className="mr-1 text-primary">&ldquo;</span>
				{caseData.quote}
			</p>

			{/* Facts list */}
			<div className="space-y-2 text-sm">
				{caseData.facts.map((fact, idx) => {
					const isEarnings = fact.label.toLowerCase().includes('monthly')
					return (
						<div key={idx} className="flex justify-between">
							<span className="text-muted-foreground">{fact.label}</span>
							<span className={cn('tabular-nums text-foreground', isEarnings && 'font-semibold text-primary')}>
								{fact.value}
							</span>
						</div>
					)
				})}
			</div>
		</div>
	)
}

// ============================================================================
// UseCasesSection Organism
// ============================================================================

export function UseCasesSection({ kicker, title, subtitle, cases, ctas, className }: UseCasesSectionProps) {
	return (
		<section
			className={cn(
				'border-b border-border bg-gradient-to-b from-background via-primary/5 to-card px-6 py-20 lg:py-28',
				className,
			)}
		>
			<div className="mx-auto max-w-7xl">
				{/* Header stack */}
				<div className="mb-16 animate-in fade-in slide-in-from-bottom-2 text-center">
					{kicker && <div className="mb-2 text-sm font-medium text-primary/80">{kicker}</div>}
					<h2 className="mb-4 text-balance text-4xl font-extrabold tracking-tight text-foreground lg:text-5xl">
						{title}
					</h2>
					{subtitle && (
						<p className="mx-auto max-w-2xl text-pretty text-lg leading-snug text-muted-foreground lg:text-xl">
							{subtitle}
						</p>
					)}
				</div>

				{/* Grid */}
				<div className="grid gap-6 md:grid-cols-2">
					{cases.map((caseData, index) => (
						<CaseCard key={index} caseData={caseData} index={index} />
					))}
				</div>

				{/* Micro-CTA rail */}
				{ctas && (ctas.primary || ctas.secondary) && (
					<div className="mt-8 text-center">
						<p className="mb-4 text-sm font-medium text-muted-foreground">Ready to join them?</p>
						<div className="flex flex-col items-center justify-center gap-2 sm:flex-row">
							{ctas.primary && (
								<Button asChild size="lg">
									<a href={ctas.primary.href}>{ctas.primary.label}</a>
								</Button>
							)}
							{ctas.secondary && (
								<Button asChild variant="outline" size="lg">
									<a href={ctas.secondary.href}>{ctas.secondary.label}</a>
								</Button>
							)}
						</div>
					</div>
				)}
			</div>
		</section>
	)
}

// ============================================================================
// ProvidersUseCases (thin wrapper with defaults)
// ============================================================================

export function ProvidersUseCases() {
	const cases: Case[] = [
		{
			icon: <Gamepad2 />,
			title: 'Gaming PC Owners',
			subtitle: 'Most common provider type',
			quote: "I game ~3-4 hours/day. The rest, my 4090 was idle. Now it earns ~€150/mo while I'm at work or asleep.",
			facts: [
				{ label: 'Typical GPU:', value: 'RTX 4080–4090' },
				{ label: 'Availability:', value: '16–20 h/day' },
				{ label: 'Monthly:', value: '€120–180' },
			],
			image: {
				src: '/illustrations/gaming-pc-owner.svg',
				alt: 'illustration of a modern gaming PC setup with RGB-lit tower showing GPU fans through tempered glass panel, dual monitors, and mechanical keyboard with colorful backlighting',
			},
		},
		{
			icon: <Server />,
			title: 'Homelab Enthusiasts',
			subtitle: 'Multiple GPUs, high earnings',
			quote: 'Four GPUs across my homelab bring ~€400/mo. It covers power and leaves profit.',
			facts: [
				{ label: 'Setup:', value: '3–6 GPUs' },
				{ label: 'Availability:', value: '20–24 h/day' },
				{ label: 'Monthly:', value: '€300–600' },
			],
			image: {
				src: '/illustrations/homelab-enthusiast.svg',
				alt: 'illustration of a professional server rack with 19-inch rails, 4U chassis showing multiple GPUs through ventilated panel, blue LED status indicators, and color-coded ethernet cables with cable management',
			},
		},
		{
			icon: <Cpu />,
			title: 'Former Crypto Miners',
			subtitle: 'Repurpose mining rigs',
			quote: 'After PoS, my rig idled. rbee now earns more than mining—with better margins.',
			facts: [
				{ label: 'Setup:', value: '6–12 GPUs' },
				{ label: 'Availability:', value: '24 h/day' },
				{ label: 'Monthly:', value: '€600–1,200' },
			],
			image: {
				src: '/illustrations/former-crypto-miner.svg',
				alt: 'illustration of a repurposed open-air mining frame with aluminum rails, 8 GPUs mounted horizontally with PCIe risers, clean cable management with zip ties, LED strip lighting, and industrial power supply',
			},
		},
		{
			icon: <Monitor />,
			title: 'Workstation Owners',
			subtitle: 'Professional GPUs earning',
			quote: 'My RTX 4080 is busy on renders only. The rest of the time it makes ~€100/mo on rbee.',
			facts: [
				{ label: 'Typical GPU:', value: 'RTX 4070–4080' },
				{ label: 'Availability:', value: '12–16 h/day' },
				{ label: 'Monthly:', value: '€80–140' },
			],
			image: {
				src: '/illustrations/workstation-owner.svg',
				alt: 'illustration of a creative workstation with 34-inch ultrawide curved monitor displaying 3D modeling software, graphics tablet with stylus, and powerful tower with mesh front panel and white LED accents',
			},
		},
	]

	return (
		<UseCasesSection
			kicker="Real Providers, Real Earnings"
			title="Who's Earning with rbee?"
			subtitle="From gamers to homelab builders, anyone with a spare GPU can turn idle time into income."
			cases={cases}
			ctas={{
				primary: { label: 'Start Earning', href: '/signup' },
				secondary: { label: 'Estimate My Payout', href: '#earnings-calculator' },
			}}
		/>
	)
}
