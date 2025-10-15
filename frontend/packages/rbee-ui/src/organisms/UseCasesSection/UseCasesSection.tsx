import { Badge } from '@rbee/ui/atoms/Badge'
import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type { ComponentType } from 'react'

export type UseCase = {
	icon: ComponentType<{ className?: string }>
	title: string
	scenario: string
	solution: string
	outcome: string
	tags?: string[]
	cta?: { label: string; href: string }
	illustrationSrc?: string
}

export type UseCasesSectionProps = {
	title: string
	subtitle?: string
	items: UseCase[]
	columns?: 2 | 3
	id?: string
	className?: string
}

export function UseCasesSection({ title, subtitle, items, columns = 3, id, className }: UseCasesSectionProps) {
	const gridCols = columns === 2 ? 'sm:grid-cols-2' : 'sm:grid-cols-2 lg:grid-cols-3'

	return (
		<section id={id} className={cn('border-b border-border bg-secondary py-24', className)}>
			<div className="container mx-auto px-4 animate-in fade-in-50 duration-400">
				{/* Heading block */}
				<div className="mx-auto max-w-3xl text-center">
					<h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">{title}</h2>
					{subtitle && <p className="text-balance text-lg leading-relaxed text-muted-foreground">{subtitle}</p>}
				</div>

				{/* Cards grid */}
				<div className={cn('mx-auto mt-16 grid max-w-6xl gap-6', gridCols)}>
					{items.map((item, i) => {
						const Icon = item.icon
						return (
							<div
								key={i}
								tabIndex={0}
								style={{ animationDelay: `${i * 80}ms` }}
								className={cn(
									'group rounded-xl border border-border/80 bg-card p-6 transition-all',
									'hover:border-primary/40 hover:bg-card/80',
									'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40',
									'animate-in fade-in slide-in-from-bottom-2 duration-400',
								)}
							>
								{/* Header row */}
								<div className="mb-4 flex items-center gap-3">
									<IconPlate icon={<Icon className="h-5 w-5" />} size="md" tone="primary" />
									<h3 className="text-base font-semibold tracking-tight text-card-foreground">{item.title}</h3>
								</div>

								{/* Body blocks */}
								<div className="space-y-4">
									{/* Scenario */}
									<div>
										<div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Scenario</div>
										<div className="mt-1 h-[3.5rem] text-sm leading-relaxed text-muted-foreground">{item.scenario}</div>
									</div>

									<div className="h-px bg-border/60" />

									{/* Solution */}
									<div>
										<div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Solution</div>
										<div className="mt-1 h-[3.5rem] text-sm leading-relaxed text-muted-foreground">{item.solution}</div>
									</div>

									<div className="h-px bg-border/60" />

									{/* Outcome callout */}
									<div className="rounded-lg border border-primary/30 bg-primary/10 p-3">
										<div className="text-xs font-semibold uppercase tracking-wide text-primary">Outcome</div>
										<div className="mt-1 min-h-[2.5rem] text-sm text-foreground">{item.outcome}</div>
									</div>
								</div>

								{/* Optional footer */}
								{(item.tags || item.cta) && (
									<div className="mt-4 flex flex-wrap items-center gap-2">
										{item.tags?.map((tag, idx) => (
											<Badge
												key={idx}
												variant="outline"
												className="rounded-full border px-2 py-0.5 text-[11px] text-muted-foreground"
											>
												{tag}
											</Badge>
										))}
										{item.cta && (
											<Link href={item.cta.href} className="ml-auto text-sm font-medium text-primary hover:underline">
												{item.cta.label}
											</Link>
										)}
									</div>
								)}
							</div>
						)
					})}
				</div>
			</div>
		</section>
	)
}
