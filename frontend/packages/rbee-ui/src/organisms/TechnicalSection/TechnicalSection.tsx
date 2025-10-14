import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import { SectionContainer } from '@rbee/ui/molecules/SectionContainer'
import { RbeeArch, GithubIcon } from '@rbee/ui/icons'
import { Terminal } from 'lucide-react'
import Link from 'next/link'

export function TechnicalSection() {
	return (
		<SectionContainer
			title="Built by Engineers, for Engineers"
			description="Rust-native orchestrator with process isolation, protocol awareness, and policy routing via Rhai."
			headingId="tech-title"
			align="center"
		>
			<div className="grid grid-cols-12 gap-6 lg:gap-10 max-w-6xl mx-auto motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-500">
				{/* Left Column: Architecture Highlights + Diagram */}
				<div className="col-span-12 lg:col-span-6 space-y-6">
					{/* Architecture Highlights */}
					<div>
						<div className="text-xs tracking-wide uppercase text-muted-foreground mb-3">Core Principles</div>
						<h3 className="text-2xl font-bold text-foreground mb-4">Architecture Highlights</h3>
						<ul className="space-y-3">
							<li>
								<div className="text-sm font-medium text-foreground">BDD-Driven Development</div>
								<div className="text-xs text-muted-foreground">42/62 scenarios passing (68% complete)</div>
								<div className="text-xs text-muted-foreground">Live CI coverage</div>
							</li>
							<li>
								<div className="text-sm font-medium text-foreground">Cascading Shutdown Guarantee</div>
								<div className="text-xs text-muted-foreground">No orphaned processes. Clean VRAM lifecycle.</div>
							</li>
							<li>
								<div className="text-sm font-medium text-foreground">Process Isolation</div>
								<div className="text-xs text-muted-foreground">Worker-level sandboxes. Zero cross-leak.</div>
							</li>
							<li>
								<div className="text-sm font-medium text-foreground">Protocol-Aware Orchestration</div>
								<div className="text-xs text-muted-foreground">SSE, JSON, binary protocols.</div>
							</li>
							<li>
								<div className="text-sm font-medium text-foreground">Smart/Dumb Separation</div>
								<div className="text-xs text-muted-foreground">Central brain, distributed execution.</div>
							</li>
						</ul>

						{/* BDD Coverage Progress Bar */}
						<div className="mt-6">
							<div className="flex items-center justify-between mb-2">
								<span className="text-sm font-medium text-foreground">BDD Coverage</span>
								<span className="text-xs text-muted-foreground">42/62 scenarios passing</span>
							</div>
							<div className="relative h-2 rounded bg-muted">
								<div className="absolute inset-y-0 left-0 w-[68%] bg-chart-3 rounded" />
							</div>
							<p className="text-xs text-muted-foreground mt-1">68% complete</p>
						</div>
					</div>

					{/* Architecture Diagram (Desktop Only) */}
					<RbeeArch
						className="hidden md:block rounded-2xl ring-1 ring-border/60 shadow-sm"
						aria-label="rbee architecture diagram showing orchestrator, policy engine, and worker pools"
					/>
				</div>

				{/* Right Column: Technology Stack (Sticky on Large Screens) */}
				<div className="col-span-12 lg:col-span-6 space-y-6 lg:sticky lg:top-20">
					<div>
						<div className="text-xs tracking-wide uppercase text-muted-foreground mb-3">Stack</div>
						<h3 className="text-2xl font-bold text-foreground mb-4">Technology Stack</h3>
						<div className="space-y-3">
							{/* Spec Cards */}
							<article
								role="group"
								aria-label="Tech: Rust"
								className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400"
							>
								<div className="font-semibold text-foreground">Rust</div>
								<div className="text-sm text-muted-foreground">Performance + memory safety.</div>
							</article>

							<article
								role="group"
								aria-label="Tech: Candle ML"
								className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-100"
							>
								<div className="font-semibold text-foreground">Candle ML</div>
								<div className="text-sm text-muted-foreground">Rust-native inference.</div>
							</article>

							<article
								role="group"
								aria-label="Tech: Rhai Scripting"
								className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-200"
							>
								<div className="font-semibold text-foreground">Rhai Scripting</div>
								<div className="text-sm text-muted-foreground">Embedded, sandboxed policies.</div>
							</article>

							<article
								role="group"
								aria-label="Tech: SQLite"
								className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-300"
							>
								<div className="font-semibold text-foreground">SQLite</div>
								<div className="text-sm text-muted-foreground">Embedded, zero-ops DB.</div>
							</article>

							<article
								role="group"
								aria-label="Tech: Axum + Vue.js"
								className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-400"
							>
								<div className="font-semibold text-foreground">Axum + Vue.js</div>
								<div className="text-sm text-muted-foreground">Async backend + modern UI.</div>
							</article>

							{/* Open Source CTA Card */}
							<article
								role="group"
								aria-label="Open Source Information"
								className="bg-primary/10 border border-primary/30 rounded-xl p-5 flex items-center justify-between motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-500"
							>
								<div>
									<div className="font-bold text-foreground">100% Open Source</div>
									<div className="text-sm text-muted-foreground">MIT License</div>
								</div>
								<Button
									variant="outline"
									size="sm"
									className="border-primary/30 bg-transparent"
									aria-label="View rbee source on GitHub"
									asChild
								>
									<a href="https://github.com/yourusername/rbee" target="_blank" rel="noopener noreferrer">
										<GithubIcon size={16} />
										View Source
									</a>
								</Button>
							</article>

							{/* Architecture Docs Link */}
							<Link
								href="/docs/architecture"
								className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
							>
								Read Architecture →
							</Link>
						</div>
					</div>
				</div>
			</div>
		</SectionContainer>
	)
}
