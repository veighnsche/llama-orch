import { IconPlate, SectionContainer } from '@rbee/ui/molecules'
import { Activity, XCircle } from 'lucide-react'

export function RealTimeProgress() {
	return (
		<SectionContainer
			title="Real‑time Progress Tracking"
			bgVariant="background"
			subtitle="Live narration of each step—model loading, token generation, resource usage—as it happens."
		>
			<div className="max-w-6xl mx-auto space-y-10">
				{/* Block 1: Live Terminal Timeline */}
				<div>
					<div className="flex items-start gap-3 mb-4">
						<IconPlate icon={Activity} tone="primary" size="md" shape="rounded" />
						<div>
							<h3 className="text-2xl font-bold tracking-tight text-foreground">SSE Narration Architecture</h3>
							<p className="text-muted-foreground">
								Workers stream every step as Server-Sent Events—from model load to token generation.
							</p>
						</div>
					</div>

					<div className="rounded-2xl border border-border bg-card overflow-hidden animate-in fade-in slide-in-from-bottom-2">
						{/* Terminal bar */}
						<div className="flex items-center gap-1 bg-muted/60 px-4 py-2">
							<span className="size-2 rounded-full bg-red-500/70" aria-hidden="true" />
							<span className="size-2 rounded-full bg-yellow-500/70" aria-hidden="true" />
							<span className="size-2 rounded-full bg-green-500/70" aria-hidden="true" />
							<span className="ml-3 font-mono text-xs text-muted-foreground">SSE narration — worker 8001</span>
							<div className="ml-auto hidden sm:flex items-center gap-2 text-[10px]">
								<span className="rounded px-1.5 py-0.5 bg-chart-3/15 text-chart-3">OK</span>
								<span className="rounded px-1.5 py-0.5 bg-primary/15 text-primary">IO</span>
								<span className="rounded px-1.5 py-0.5 bg-destructive/15 text-destructive">ERR</span>
							</div>
						</div>

						{/* Scrollable log */}
						<div
							className="bg-background p-6 font-mono text-sm leading-relaxed max-h-[340px] overflow-auto"
							role="log"
							aria-live="polite"
						>
							<div className="text-muted-foreground animate-in fade-in duration-300">
								[00:00.00] [worker] start :8001
							</div>
							<div className="text-muted-foreground animate-in fade-in duration-300 delay-75">
								[00:00.03] [device] CUDA#1 initialized
							</div>
							<div className="text-primary animate-in fade-in duration-300 delay-150">
								[00:00.12] [loader] /models/tinyllama-q4.gguf → loading…
							</div>
							<div className="text-chart-3 animate-in fade-in duration-300 delay-200">
								[00:01.02] [loader] loaded 669MB in VRAM ✓
							</div>
							<div className="text-muted-foreground animate-in fade-in duration-300 delay-300">
								[00:01.05] [http] server ready :8001
							</div>

							<div className="mt-2 text-muted-foreground animate-in fade-in duration-300 delay-400">
								[00:01.10] [candle] inference start (18 chars)
							</div>
							<div className="text-muted-foreground animate-in fade-in duration-300 delay-500">
								[00:01.11] [tokenizer] prompt → 4 tokens
							</div>
							<div className="text-foreground animate-in fade-in duration-300 delay-600">Once upon a time…</div>
							<div className="text-chart-3 animate-in fade-in duration-300 delay-700">
								[00:01.26] [candle] generated 20 tokens (133 tok/s) ✓
							</div>
						</div>

						{/* Footer hint */}
						<div className="border-t border-border px-6 py-3 text-xs text-muted-foreground">
							Narration → <code className="bg-muted px-1 rounded">stderr</code> · Tokens →{' '}
							<code className="bg-muted px-1 rounded">stdout</code>
						</div>
					</div>
				</div>

				{/* Block 2: Stream Meter Row */}
				<div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2 delay-100">
					<div className="bg-card border border-border rounded-xl p-4 hover:-translate-y-0.5 transition-transform">
						<div className="text-xs text-muted-foreground">Throughput</div>
						<div className="mt-1 text-lg font-semibold text-foreground">133 tok/s</div>
						<div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
							<div className="h-full w-[80%] bg-chart-3" />
						</div>
					</div>
					<div className="bg-card border border-border rounded-xl p-4 hover:-translate-y-0.5 transition-transform">
						<div className="text-xs text-muted-foreground">First token latency</div>
						<div className="mt-1 text-lg font-semibold text-foreground">150 ms</div>
						<div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
							<div className="h-full w-[60%] bg-primary" />
						</div>
					</div>
					<div className="bg-card border border-border rounded-xl p-4 hover:-translate-y-0.5 transition-transform">
						<div className="text-xs text-muted-foreground">VRAM used</div>
						<div className="mt-1 text-lg font-semibold text-foreground">669 MB</div>
						<div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
							<div className="h-full w-[45%] bg-emerald-500/80" />
						</div>
					</div>
				</div>

				{/* Block 3: Cancellation Sequence Card */}
				<div className="rounded-2xl border border-border bg-card p-6 animate-in fade-in slide-in-from-bottom-2 delay-150">
					<div className="flex items-start gap-3 mb-4">
						<IconPlate icon={XCircle} tone="warning" size="md" shape="rounded" />
						<div>
							<h3 className="text-2xl font-bold tracking-tight text-foreground">Request Cancellation</h3>
							<p className="text-muted-foreground">
								Ctrl+C or API cancel stops the job, frees resources, and leaves no orphaned processes.
							</p>
						</div>
					</div>

					{/* Sequence */}
					<ol className="grid gap-3 sm:grid-cols-4 text-sm" aria-label="Cancellation sequence">
						<li className="bg-background border border-border rounded-xl p-4 hover:ring-1 hover:ring-border transition-all">
							<div className="text-xs text-muted-foreground">t+0ms</div>
							<div className="font-semibold text-foreground">
								Client sends <code className="bg-muted px-1 rounded text-xs">POST /v1/cancel</code>
							</div>
							<p className="mt-1 text-muted-foreground">Idempotent request.</p>
						</li>
						<li className="bg-background border border-border rounded-xl p-4 hover:ring-1 hover:ring-border transition-all">
							<div className="text-xs text-muted-foreground">t+50ms</div>
							<div className="font-semibold text-foreground">SSE disconnect detected</div>
							<p className="mt-1 text-muted-foreground">Stream closes ≤ 1s.</p>
						</li>
						<li className="bg-background border border-border rounded-xl p-4 hover:ring-1 hover:ring-border transition-all">
							<div className="text-xs text-muted-foreground">t+80ms</div>
							<div className="font-semibold text-foreground">Immediate cleanup</div>
							<p className="mt-1 text-muted-foreground">Stop tokens, release slot, log event.</p>
						</li>
						<li className="bg-background border border-border rounded-xl p-4 hover:ring-1 hover:ring-border transition-all">
							<div className="text-xs text-muted-foreground">t+120ms</div>
							<div className="font-semibold text-chart-3">Worker idle ✓</div>
							<p className="mt-1 text-muted-foreground">Ready for next task.</p>
						</li>
					</ol>
				</div>
			</div>
		</SectionContainer>
	)
}
