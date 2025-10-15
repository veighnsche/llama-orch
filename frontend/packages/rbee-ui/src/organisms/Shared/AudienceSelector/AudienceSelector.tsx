'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { AudienceCard } from '@rbee/ui/molecules'
import { DevGrid, GpuMarket, ComplianceShield } from '@rbee/ui/icons'
import { ChevronRight, Code2, Server, Shield } from 'lucide-react'
import Link from 'next/link'

export function AudienceSelector() {
	return (
		<section className="relative bg-background py-24 sm:py-32">
			{/* Enhanced top hairline with backdrop blur */}
			<div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent backdrop-blur-sm" />

			{/* Radial gradient backplate */}
			<div
				className="pointer-events-none absolute inset-x-0 top-0 h-[600px] opacity-40"
				style={{
					background: 'radial-gradient(ellipse 80% 50% at 50% 0%, hsl(var(--primary) / 0.05), transparent)',
				}}
				aria-hidden="true"
			/>

			<div className="relative mx-auto max-w-7xl px-6 lg:px-8">
				{/* Header block with tighter max-width */}
				<header className="mx-auto mb-10 max-w-2xl text-center sm:mb-12">
					<p className="mb-4 font-sans text-sm font-medium uppercase tracking-wider text-primary">Choose your path</p>

					<h2 className="mb-6 font-sans text-3xl font-semibold tracking-tight text-foreground sm:text-4xl lg:text-5xl">
						Where should you start?
					</h2>

					<p className="font-sans text-lg leading-relaxed text-muted-foreground">
						rbee adapts to how you work—build on your own GPUs, monetize idle capacity, or deploy compliant AI at scale.
					</p>
				</header>

				{/* Grid with responsive 2→3 column layout and equal heights */}
				<div
					className="mx-auto grid max-w-6xl grid-cols-1 content-start gap-6 sm:grid-cols-2 xl:grid-cols-3 xl:gap-8"
					aria-label="Audience options: Developers, GPU Owners, Enterprise"
				>
					{/* Developers Card */}
					<div className="flex h-full">
						<AudienceCard
							icon={Code2}
							category="For Developers"
							title="Build on Your Hardware"
							description="Power Zed, Cursor, and your own agents on YOUR GPUs. OpenAI-compatible—drop-in, zero API fees."
							features={[
								'Zero API costs, unlimited usage',
								'Your code stays on your network',
								'Agentic API + TypeScript utils',
							]}
							href="/developers"
							ctaText="Explore Developer Path"
							color="chart-2"
							className="h-full min-h-[22rem] p-6 transition-transform duration-300 hover:-translate-y-1 focus-visible:ring-2 focus-visible:ring-primary/40 sm:p-7 md:min-h-[22rem] lg:p-8"
							imageSlot={
								<DevGrid
									size={56}
									aria-hidden
								/>
							}
							badgeSlot={
								<Badge variant="outline" className="border-chart-2/30 bg-chart-2/5 text-chart-2">
									Homelab-ready
								</Badge>
							}
							decisionLabel="Code with AI locally"
						/>
					</div>

					{/* GPU Owners Card */}
					<div className="flex h-full">
						<AudienceCard
							icon={Server}
							category="For GPU Owners"
							title="Monetize Your Hardware"
							description="Join the rbee marketplace and earn from gaming rigs to server farms—set price, stay in control."
							features={['Set pricing & availability', 'Audit trails and payouts', 'Passive income from idle GPUs']}
							href="/gpu-providers"
							ctaText="Become a Provider"
							color="chart-3"
							className="h-full min-h-[22rem] p-6 transition-transform duration-300 hover:-translate-y-1 focus-visible:ring-2 focus-visible:ring-primary/40 sm:p-7 md:min-h-[22rem] lg:p-8"
							imageSlot={
								<GpuMarket
									size={56}
									aria-hidden
								/>
							}
							decisionLabel="Earn from idle GPUs"
						/>
					</div>

					{/* Enterprise Card */}
					<div className="flex h-full">
						<AudienceCard
							icon={Shield}
							category="For Enterprise"
							title="Compliance & Security"
							description="EU-native compliance, audit trails, and zero-trust architecture—from day one."
							features={['GDPR with 7-year retention', 'SOC2 & ISO 27001 aligned', 'Private cloud or on-prem']}
							href="/enterprise"
							ctaText="Enterprise Solutions"
							color="primary"
							className="h-full min-h-[22rem] p-6 transition-transform duration-300 hover:-translate-y-1 focus-visible:ring-2 focus-visible:ring-primary/40 sm:p-7 md:min-h-[22rem] lg:p-8"
							imageSlot={
								<ComplianceShield
									size={56}
									aria-hidden
								/>
							}
							decisionLabel="Deploy with compliance"
						/>
					</div>
				</div>

				{/* Bottom helper links */}
				<div className="mx-auto mt-12 text-center">
					<Link
						href="#compare"
						className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40"
					>
						Not sure? Compare paths
						<ChevronRight className="h-3.5 w-3.5" />
					</Link>
					<span className="mx-3 text-muted-foreground/50">·</span>
					<Link
						href="#contact"
						className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40"
					>
						Talk to us
						<ChevronRight className="h-3.5 w-3.5" />
					</Link>
				</div>
			</div>
		</section>
	)
}
