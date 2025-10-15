import { IconPlate } from '@rbee/ui/molecules'
import type { IconPlateProps } from '@rbee/ui/molecules'
import type { LucideIcon } from 'lucide-react'

interface PlaybookItemProps {
	icon: LucideIcon
	color: IconPlateProps['tone']
	title: string
	checkCount: number
	severityDots: Array<'destructive' | 'primary' | 'chart-2' | 'chart-3'>
	description: string
	checks: Array<{
		severity: 'destructive' | 'primary' | 'chart-2' | 'chart-3'
		text: string
		detail: string
	}>
	footer: React.ReactNode
}

export function PlaybookItem({
	icon,
	color,
	title,
	checkCount,
	severityDots,
	description,
	checks,
	footer,
}: PlaybookItemProps) {
	return (
		<details className="group border-b border-border last:border-b-0">
			<summary className="flex items-center justify-between gap-3 cursor-pointer px-5 py-4 hover:bg-muted/50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
				<span className="flex items-center gap-3">
					<IconPlate icon={icon} tone={color} size="sm" shape="rounded" />
					<span className="font-semibold text-foreground">{title}</span>
					<span className="ml-2 text-xs text-muted-foreground">{checkCount} checks</span>
				</span>
				<span className="hidden sm:flex items-center gap-2 mr-1" aria-hidden="true">
					{severityDots.map((severity, i) => (
						<span key={i} className={`size-1.5 rounded-full bg-${severity}/80`} />
					))}
					<span className="text-muted-foreground text-xs">severity</span>
					<span className="text-muted-foreground text-xs group-open:rotate-180 transition-transform">▾</span>
				</span>
			</summary>
			<div className="px-6 pb-5 animate-in fade-in duration-200 group-open:border-t group-open:border-border/80">
				<p className="text-sm text-muted-foreground/90 mb-3 mt-3">{description}</p>
				<ul className="grid sm:grid-cols-2 gap-x-6 gap-y-2 text-sm">
					{checks.map((check, i) => (
						<li key={i} className="flex items-start gap-2" data-playbook-item>
							<span className={`mt-2 size-1.5 rounded-full bg-${check.severity}/80 shrink-0`} aria-hidden="true" />
							<span>
								{check.text} <span className="text-muted-foreground">({check.detail})</span>
							</span>
						</li>
					))}
				</ul>
				{footer}
			</div>
		</details>
	)
}

interface PlaybookHeaderProps {
	title: string
	description: string
	filterCategories: string[]
	onExpandAll: () => void
	onCollapseAll: () => void
}

export function PlaybookHeader({
	title,
	description,
	filterCategories,
	onExpandAll,
	onCollapseAll,
}: PlaybookHeaderProps) {
	return (
		<div className="flex items-center justify-between gap-3 px-5 py-4 border-b border-border">
			<div className="flex items-center gap-2">
				<span className="inline-flex items-center justify-center rounded-md border border-transparent bg-secondary text-secondary-foreground px-2 py-0.5 text-xs font-medium">
					{title}
				</span>
				<span className="text-sm text-muted-foreground">{description}</span>
			</div>
			<div className="flex items-center gap-2">
				{filterCategories.map((category) => (
					<button
						key={category}
						className="px-2.5 py-1 rounded-md text-xs bg-muted hover:bg-muted/70 transition-colors"
					>
						{category}
					</button>
				))}
				<button
					onClick={onExpandAll}
					className="ml-2 px-2.5 py-1 rounded-md text-xs border hover:bg-muted transition-colors"
				>
					Expand all
				</button>
				<button
					onClick={onCollapseAll}
					className="px-2.5 py-1 rounded-md text-xs border hover:bg-muted transition-colors"
				>
					Collapse all
				</button>
			</div>
		</div>
	)
}
