import { IconPlate } from '@rbee/ui/molecules/IconPlate'
import { cn } from '@rbee/ui/utils'
import { Eye, FileCheck, Lock, type LucideIcon, Shield } from 'lucide-react'

type SecurityPoint = string

type SecurityCardItem = {
	icon: LucideIcon
	title: string
	subtitle?: string
	body: string
	points: SecurityPoint[]
}

type SecuritySectionProps = {
	kicker?: string
	title: string
	subtitle?: string
	items: SecurityCardItem[]
	ribbon?: { text: string }
}

export function SecuritySection({ kicker, title, subtitle, items, ribbon }: SecuritySectionProps) {
	return (
		<section className="border-b border-border bg-gradient-to-b from-background via-emerald-500/5 to-background px-6 py-20 lg:py-28">
			<div className="mx-auto max-w-7xl">
				{/* Header */}
				<div className="animate-in fade-in slide-in-from-bottom-2 mb-16 text-center motion-reduce:animate-none">
					{kicker && <div className="mb-2 text-sm font-medium text-emerald-400/80">{kicker}</div>}
					<h2 className="mb-4 text-balance text-4xl font-extrabold tracking-tight text-foreground lg:text-5xl">
						{title}
					</h2>
					{subtitle && (
						<p className="mx-auto max-w-2xl text-pretty text-lg leading-snug text-muted-foreground lg:text-xl">
							{subtitle}
						</p>
					)}
				</div>

				{/* Security Cards Grid */}
				<div className="grid gap-6 md:grid-cols-2">
					{items.map((item, idx) => {
						const Icon = item.icon
						const delays = ['delay-75', 'delay-150', 'delay-200', 'delay-300']
						return (
							<div
								key={idx}
								className={cn(
									'animate-in fade-in slide-in-from-bottom-2 rounded-2xl border border-border/70 bg-gradient-to-b from-card/70 to-background/60 p-6 backdrop-blur transition-transform hover:translate-y-0.5 motion-reduce:animate-none supports-[backdrop-filter]:bg-background/60 sm:p-7',
									delays[idx % delays.length],
								)}
							>
								<div className="mb-5 flex items-center gap-4">
									<IconPlate
										icon={<Icon className="h-6 w-6" aria-hidden="true" />}
										size="lg"
										className="bg-emerald-400/10 text-emerald-400"
									/>
									<div>
										<h3 className="text-lg font-semibold text-foreground">{item.title}</h3>
										{item.subtitle && <div className="text-xs text-muted-foreground">{item.subtitle}</div>}
									</div>
								</div>
								<p className="mb-4 line-clamp-3 text-sm leading-relaxed text-muted-foreground">{item.body}</p>
								<ul className="space-y-2">
									{item.points.map((point, pidx) => (
										<li key={pidx} className="flex items-center gap-2 text-sm text-muted-foreground">
											<div className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
											{point}
										</li>
									))}
								</ul>
							</div>
						)
					})}
				</div>

				{/* Insurance Ribbon */}
				{ribbon && (
					<div className="mt-10 rounded-2xl border border-emerald-400/30 bg-emerald-400/10 p-5 text-center">
						<p className="flex items-center justify-center gap-2 text-balance text-base font-medium text-emerald-400 lg:text-lg">
							<Shield className="h-4 w-4" aria-hidden="true" />
							<span className="tabular-nums">{ribbon.text}</span>
						</p>
					</div>
				)}
			</div>
		</section>
	)
}

// Provider-specific wrapper with data
export function ProvidersSecurity() {
	const items: SecurityCardItem[] = [
		{
			icon: Shield,
			title: 'Sandboxed Execution',
			subtitle: 'Complete isolation',
			body: 'All jobs run in isolated sandboxes with no access to your files, network, or personal data.',
			points: ['No file system access', 'No network access', 'No personal data access', 'Automatic cleanup'],
		},
		{
			icon: Lock,
			title: 'Encrypted Communication',
			subtitle: 'End-to-end encryption',
			body: 'All communication between your GPU and the marketplace is encrypted using industry-standard protocols.',
			points: ['TLS 1.3', 'Secure payment processing', 'Protected earnings data', 'Private job details'],
		},
		{
			icon: Eye,
			title: 'Malware Scanning',
			subtitle: 'Automatic protection',
			body: 'Every job is automatically scanned for malware before execution. Suspicious jobs are blocked.',
			points: ['Real-time detection', 'Automatic blocking', 'Threat intel updates', 'Customer vetting'],
		},
		{
			icon: FileCheck,
			title: 'Hardware Protection',
			subtitle: 'Warranty-safe operation',
			body: 'Temperature monitoring, cooldown periods, and power limits protect your hardware and warranty.',
			points: ['Temperature monitoring', 'Cooldown periods', 'Power limits', 'Health monitoring'],
		},
	]

	return (
		<SecuritySection
			kicker="Security & Trust"
			title="Your Security Is Our Priority"
			subtitle="Enterprise-grade protections for your hardware, data, and earnings."
			items={items}
			ribbon={{ text: 'Plus: €1M insurance coverage is included for all providers—your hardware is protected.' }}
		/>
	)
}
