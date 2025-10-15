import { Badge } from '@rbee/ui/atoms/Badge'
import { IconPlate, SectionContainer } from '@rbee/ui/molecules'
import { CheckCircle2, Lock, Shield } from 'lucide-react'

export function SecurityIsolation() {
	return (
		<SectionContainer
			title="Security & Isolation"
			bgVariant="background"
			subtitle="Defense-in-depth with five focused Rust crates. Enterprise-grade security for your homelab."
		>
			<div className="max-w-6xl mx-auto space-y-8">
				{/* Block 1: Crate Lattice */}
				<div className="rounded-2xl border border-border bg-card p-6 md:p-8 animate-in fade-in slide-in-from-bottom-2">
					<div className="flex items-start gap-3 mb-6">
						<IconPlate icon={Shield} tone="chart-2" size="md" shape="rounded" />
						<div>
							<h3 className="text-2xl font-bold tracking-tight text-foreground">Five Specialized Security Crates</h3>
							<p className="text-muted-foreground">
								Each concern ships as its own Rust crate—focused responsibility, no monolith.
							</p>
						</div>
					</div>

					{/* Crate lattice grid */}
					<div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
						<div className="group rounded-lg bg-background border border-border p-4 transition-colors hover:border-chart-2/50">
							<div className="font-semibold text-foreground mb-1">auth-min</div>
							<p className="text-sm text-muted-foreground">Timing-safe tokens, zero-trust auth.</p>
						</div>

						<div className="group rounded-lg bg-background border border-border p-4 transition-colors hover:border-chart-3/50">
							<div className="font-semibold text-foreground mb-1">audit-logging</div>
							<p className="text-sm text-muted-foreground">Append-only logs, 7-year retention.</p>
						</div>

						<div className="group rounded-lg bg-background border border-border p-4 transition-colors hover:border-primary/50">
							<div className="font-semibold text-foreground mb-1">input-validation</div>
							<p className="text-sm text-muted-foreground">Injection prevention, schema validation.</p>
						</div>

						<div className="group rounded-lg bg-background border border-border p-4 transition-colors hover:border-amber-500/50">
							<div className="font-semibold text-foreground mb-1">secrets-management</div>
							<p className="text-sm text-muted-foreground">Encrypted storage, rotation, KMS-friendly.</p>
						</div>

						<div className="group rounded-lg bg-background border border-border p-4 transition-colors hover:border-chart-2/50">
							<div className="font-semibold text-foreground mb-1">deadline-propagation</div>
							<p className="text-sm text-muted-foreground">Timeouts, cleanup, cascading shutdown.</p>
						</div>
					</div>
				</div>

				{/* Block 2: Process Isolation */}
				<div className="grid md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-2 delay-100">
					<div className="rounded-2xl border border-border bg-card p-6">
						<div className="flex items-start gap-3 mb-4">
							<IconPlate icon={Lock} tone="chart-3" size="sm" shape="rounded" />
							<div>
								<h3 className="text-lg font-bold text-foreground">Process Isolation</h3>
								<p className="text-sm text-muted-foreground mt-1">
									Workers run in isolated processes with clean shutdown.
								</p>
							</div>
						</div>
						<div className="space-y-2 text-sm">
							<div className="flex items-center gap-2">
								<div className="size-1.5 rounded-full bg-chart-3" />
								<span className="text-muted-foreground">Sandboxed execution</span>
							</div>
							<div className="flex items-center gap-2">
								<div className="size-1.5 rounded-full bg-chart-3" />
								<span className="text-muted-foreground">Cascading shutdown</span>
							</div>
							<div className="flex items-center gap-2">
								<div className="size-1.5 rounded-full bg-chart-3" />
								<span className="text-muted-foreground">VRAM cleanup</span>
							</div>
						</div>
					</div>

					<div className="rounded-2xl border border-border bg-card p-6">
						<div className="flex items-start gap-3 mb-4">
							<IconPlate icon={Shield} tone="chart-2" size="sm" shape="rounded" />
							<div>
								<h3 className="text-lg font-bold text-foreground">Zero-Trust Architecture</h3>
								<p className="text-sm text-muted-foreground mt-1">
									Defense-in-depth with timing-safe auth and audit logs.
								</p>
							</div>
						</div>
						<div className="space-y-2 text-sm">
							<div className="flex items-center gap-2">
								<div className="size-1.5 rounded-full bg-chart-2" />
								<span className="text-muted-foreground">Timing-safe authentication</span>
							</div>
							<div className="flex items-center gap-2">
								<div className="size-1.5 rounded-full bg-chart-2" />
								<span className="text-muted-foreground">Immutable audit logs</span>
							</div>
							<div className="flex items-center gap-2">
								<div className="size-1.5 rounded-full bg-chart-2" />
								<span className="text-muted-foreground">Input validation</span>
							</div>
						</div>
					</div>
				</div>
			</div>
		</SectionContainer>
	)
}
