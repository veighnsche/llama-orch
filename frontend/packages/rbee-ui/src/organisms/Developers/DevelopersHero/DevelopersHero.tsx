import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import { GitHubIcon } from '@rbee/ui/atoms/GitHubIcon'
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'
import { ArrowRight, Check } from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'
import { homelabHardwareMontage } from '@rbee/ui/assets'

export function DevelopersHero() {
	return (
		<section className="relative isolate overflow-hidden border-b border-border bg-background before:absolute before:inset-0 before:bg-[radial-gradient(70%_60%_at_50%_-10%,hsl(var(--primary)/0.15),transparent_60%)] before:pointer-events-none">
			<div className="relative mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
				<div className="lg:grid lg:grid-cols-12 lg:gap-12">
					{/* Left Column: Content */}
					<div className="lg:col-span-7 space-y-8">
						{/* Announcement Badge */}
						<div className="animate-in fade-in duration-500 delay-100" aria-live="polite" aria-atomic="true">
							<Badge
								variant="outline"
								className="inline-flex items-center gap-2 rounded-full border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary"
							>
								<span className="relative flex h-2 w-2" aria-hidden="true">
									<span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
									<span className="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
								</span>
								For developers who build with AI
							</Badge>
						</div>

						{/* H1: Two-line lockup */}
						<h1 className="text-balance tracking-tight text-5xl sm:text-6xl lg:text-7xl font-bold">
							<span className="block animate-in fade-in-50 slide-in-from-bottom-1 duration-500 delay-150">
								Build with AI.
							</span>
							<span className="block animate-in fade-in-50 slide-in-from-bottom-1 duration-500 delay-250 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
								Own your infrastructure.
							</span>
						</h1>

						{/* Benefit Subline */}
						<p className="animate-in fade-in-50 duration-500 delay-300 text-balance text-xl leading-relaxed text-muted-foreground max-w-2xl">
							Stop depending on external AI. <strong className="font-semibold text-foreground">rbee</strong> (pronounced
							&quot;are-bee&quot;) gives you an OpenAI-compatible API that runs on{' '}
							<strong className="font-semibold text-foreground">ALL your home network hardware</strong>—GPUs, Macs,
							workstations—with <strong className="font-semibold text-foreground">zero ongoing costs</strong>.
						</p>

						{/* CTA Row */}
						<div className="animate-in fade-in zoom-in-50 duration-500 delay-400 flex flex-col sm:flex-row items-start sm:items-center gap-4">
							<Button asChild size="lg" className="group">
								<Link href="#get-started">
									Get started free
									<ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" aria-hidden="true" />
								</Link>
							</Button>
							<Button asChild size="lg" variant="outline">
								<Link href="https://github.com/orchyra/rbee" target="_blank" rel="noopener noreferrer">
									<GitHubIcon className="h-4 w-4" aria-hidden="true" />
									View on GitHub
								</Link>
							</Button>
						</div>

						{/* Tertiary Link (Mobile Only) */}
						<div className="sm:hidden">
							<Link
								href="#how-it-works"
								className="inline-flex items-center gap-2 text-sm text-primary hover:underline underline-offset-4"
							>
								How it works
								<ArrowRight className="h-3 w-3" aria-hidden="true" />
							</Link>
						</div>

						{/* Trust Chips */}
						<div className="flex flex-wrap items-center gap-3 text-sm">
							{[
								{ label: 'Open source (GPL-3.0)', delay: 'delay-500' },
								{ label: 'OpenAI-compatible API', delay: 'delay-[600ms]' },
								{ label: 'Works with Zed & Cursor', delay: 'delay-[700ms]' },
								{ label: 'No cloud required', delay: 'delay-[800ms]' },
							].map(({ label, delay }) => (
								<Badge
									key={label}
									variant="outline"
									className={`animate-in fade-in duration-500 ${delay} inline-flex items-center gap-2 rounded-full border px-3 py-1 text-muted-foreground`}
								>
									<Check className="h-3 w-3" aria-hidden="true" />
									{label}
								</Badge>
							))}
						</div>
					</div>

					{/* Right Column: Visual Stack */}
					<div className="lg:col-span-5 mt-16 lg:mt-0 space-y-6">
						{/* Terminal Window */}
						<div className="animate-in fade-in slide-in-from-right-2 duration-500 delay-300">
							<TerminalWindow title="terminal">
								<div className="space-y-2">
									<div className="text-muted-foreground">
										<span className="text-chart-3">$</span> rbee-keeper infer --model llama-3.1-70b --prompt
										&quot;Generate API&quot;
									</div>
									<div className="text-muted-foreground">
										<span className="animate-pulse">▊</span> Streaming tokens...
									</div>
									<div className="space-y-1 text-foreground pt-2">
										<div>
											<span className="text-chart-2">export</span> <span className="text-primary">async</span>{' '}
											<span className="text-chart-4">function</span> <span className="text-chart-3">getUsers</span>(){' '}
											{'{'}
										</div>
										<div className="pl-4">
											<span className="text-chart-2">const</span> response = <span className="text-chart-2">await</span>{' '}
											<span className="text-chart-3">fetch</span>(
											<span className="text-primary">&apos;/api/users&apos;</span>)
										</div>
										<div className="pl-4">
											<span className="text-chart-2">return</span> response.<span className="text-chart-3">json</span>()
										</div>
										<div>{'}'}</div>
									</div>
									<div className="flex items-center gap-4 text-muted-foreground pt-2">
										<div>GPU 1: 87%</div>
										<div>GPU 2: 92%</div>
										<div>Cost: $0.00</div>
									</div>
								</div>
							</TerminalWindow>
						</div>

						{/* Hardware Montage */}
						<div className="animate-in fade-in duration-500 delay-400 relative">
							<Card className="relative overflow-hidden ring-1 ring-border/50 p-0">
								<Image
									src={homelabHardwareMontage}
									alt="Professional product photography of a modern homelab setup on a dark wooden desk with warm ambient lighting: foreground shows a matte black GPU tower PC with subtle RGB accents and visible PCIe slots, mid-ground features a silver M-series MacBook Pro with glowing Apple logo, background includes a compact mini-ITX workstation with exposed heatsinks and a consumer-grade WiFi router with antenna array. Shallow depth of field creates bokeh effect on background elements. Organized cable management with braided black cables. Dark navy gradient backdrop (hex #0f172a to #1e293b). Matte finishes throughout, no glossy surfaces. Studio lighting creates soft highlights on metal chassis. Conveys distributed computing across ALL your home network hardware working in harmony."
									width={840}
									height={525}
									className="rounded-xl object-cover w-full aspect-video"
									priority={false}
									sizes="(min-width: 1024px) 420px, 100vw"
								/>
								{/* Stat Chips Overlay */}
								<div className="absolute top-4 right-4 flex flex-col gap-2">
									<Badge className="bg-background/90 backdrop-blur-sm border-border text-foreground shadow-lg">
										Zed & Cursor: drop-in via OpenAI API
									</Badge>
									<Badge className="bg-background/90 backdrop-blur-sm border-border text-foreground shadow-lg">
										Zero ongoing costs
									</Badge>
								</div>
							</Card>
						</div>
					</div>
				</div>
			</div>
		</section>
	)
}
