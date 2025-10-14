import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import { Separator } from '@rbee/ui/atoms/Separator'
import { HoneycombPattern } from '@rbee/ui/patterns/HoneycombPattern'
import { Cpu, Download, Power } from 'lucide-react'

// Feature flags for quick toggles
const SHOW_HONEYCOMB_BG = true
const SHOW_MOSAIC_IMAGE = false // Set to true when real photo exists
const SHOW_STAT_STRIP = true

export function FeaturesHero() {
	return (
		<section className="py-24 bg-gradient-to-b from-background to-card relative overflow-hidden">
			{SHOW_HONEYCOMB_BG && <HoneycombPattern id="features" size="small" fadeDirection="bottom" />}

			<div className="container mx-auto px-4 grid gap-10 lg:grid-cols-2 items-center relative z-10">
				{/* Left column: Copy */}
				<div className="max-w-prose">
					<h1 className="text-5xl sm:text-6xl font-bold tracking-tight leading-[1.05] text-foreground animate-in fade-in slide-in-from-bottom-2 duration-700">
						Enterprise-grade AI.{' '}
						<span className="bg-gradient-to-r from-primary to-amber-500 bg-clip-text text-transparent contrast-more:text-primary">
							Homelab simple.
						</span>
					</h1>

					<p className="text-xl text-muted-foreground mt-6 max-w-prose animate-in fade-in duration-700 delay-100">
						rbee (pronounced &ldquo;are-bee&rdquo;) gives you enterprise AI orchestration that runs on your own
						hardware. OpenAI-compatible. No cloud lock-in. No ongoing API costs. Full control.
					</p>

					{/* Micro-badges */}
					<div className="flex flex-wrap gap-2 mt-6 animate-in fade-in duration-700 delay-100">
						<Badge variant="secondary">Runs on your GPUs</Badge>
						<Badge variant="secondary">OpenAI-compatible</Badge>
						<Badge variant="secondary">GDPR-ready</Badge>
						<Badge variant="secondary">CUDA · Metal · CPU</Badge>
					</div>

					{/* CTAs */}
					<div className="mt-8 flex flex-col sm:flex-row gap-3 animate-in fade-in slide-in-from-bottom-2 duration-700 delay-150">
						<Button asChild size="lg" className="h-12 px-6">
							<a href="#feature-list">See all features</a>
						</Button>
						<Button asChild size="lg" variant="secondary" className="h-12 px-6">
							<a href="#how-it-works">How it works</a>
						</Button>
					</div>
				</div>

				{/* Right column: Feature Mosaic */}
				<div className="relative order-last lg:order-none">
					<div className="grid grid-cols-2 gap-4 lg:max-w-md mx-auto lg:mx-0">
						{/* Card 1: Programmable Scheduler (tall) */}
						<Card className="row-span-2 p-4 gap-3 animate-in zoom-in-95 fade-in duration-700 delay-200 transition-transform will-change-transform hover:-translate-y-0.5">
							<div className="rounded-lg bg-primary/10 p-2.5 w-fit">
								<Cpu className="h-5 w-5 text-primary" aria-hidden="true" focusable="false" />
							</div>
							<div>
								<h3 className="font-semibold text-card-foreground text-sm mb-1">Programmable Scheduler</h3>
								<p className="text-xs text-muted-foreground leading-relaxed">
									Write custom routing logic in Rhai scripts. Route by cost, latency, GPU type, or compliance
									requirements. Update policies without recompiling. 40+ built-in helpers for worker selection, VRAM
									queries, and smart eviction.
								</p>
							</div>
						</Card>

						{/* Card 2: Model Catalog */}
						<Card className="p-4 gap-3 animate-in zoom-in-95 fade-in duration-700 delay-300 transition-transform will-change-transform hover:-translate-y-0.5">
							<div className="rounded-lg bg-chart-2/10 p-2.5 w-fit">
								<Download className="h-5 w-5 text-chart-2" aria-hidden="true" focusable="false" />
							</div>
							<div>
								<h3 className="font-semibold text-card-foreground text-sm mb-1">Model Catalog</h3>
								<p className="text-xs text-muted-foreground leading-relaxed">
									One click to load models. Watch download and loading progress.
								</p>
							</div>
						</Card>

						{/* Card 3: Cascading Shutdown */}
						<Card className="p-4 gap-3 animate-in zoom-in-95 fade-in duration-700 delay-400 transition-transform will-change-transform hover:-translate-y-0.5">
							<div className="rounded-lg bg-chart-3/10 p-2.5 w-fit">
								<Power className="h-5 w-5 text-chart-3" aria-hidden="true" focusable="false" />
							</div>
							<div>
								<h3 className="font-semibold text-card-foreground text-sm mb-1">Cascading Shutdown</h3>
								<p className="text-xs text-muted-foreground leading-relaxed">
									Ctrl+C cleanly tears down hives and workers. No leftovers.
								</p>
							</div>
						</Card>
					</div>

					{/* Optional background image (gated by flag) */}
					{SHOW_MOSAIC_IMAGE && (
						<div className="pointer-events-none select-none absolute inset-x-0 bottom-0 opacity-10 blur-[1px] hidden lg:block">
							{/* Placeholder for real asset - uncomment when image exists */}
							{/* <Image
                src="/images/homelab-rack.jpg"
                alt="high-contrast photo of a tidy homelab GPU mini-rack with soft rim light, shallow depth of field, cinematic 35mm, conveys approachable enterprise hardware at home"
                width={1200}
                height={800}
                priority
                className="w-full h-auto object-cover"
              /> */}
						</div>
					)}
				</div>
			</div>

			{/* Stat strip */}
			{SHOW_STAT_STRIP && (
				<div className="container mx-auto px-4 relative z-10">
					<Separator className="mt-10 opacity-60" />
					<div className="mt-6 pt-0 text-sm text-muted-foreground grid grid-cols-1 sm:grid-cols-3 gap-2 text-center sm:text-left">
						<div>
							<strong className="font-semibold">42/62</strong> BDD scenarios passing
						</div>
						<div>
							<strong className="font-semibold">Zero</strong> cloud dependencies
						</div>
						<div>
							Multi-backend: <strong className="font-semibold">CUDA · Metal · CPU</strong>
						</div>
					</div>
				</div>
			)}
		</section>
	)
}
