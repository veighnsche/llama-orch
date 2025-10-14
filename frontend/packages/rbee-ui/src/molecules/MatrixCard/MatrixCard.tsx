import { cn } from '@rbee/ui/utils'
import { Check, X } from 'lucide-react'
import type { Provider, Row } from '../MatrixTable'

export interface MatrixCardProps {
	provider: Provider
	rows: Row[]
	className?: string
}

export function MatrixCard({ provider, rows, className }: MatrixCardProps) {
	const renderStatus = (value: boolean | 'Partial' | string) => {
		if (value === true) {
			return (
				<span className="flex items-center gap-1.5 text-chart-3" aria-label="Included">
					<Check className="h-4 w-4" />
					<span className="text-xs">Included</span>
				</span>
			)
		}
		if (value === false) {
			return (
				<span className="flex items-center gap-1.5 text-destructive" aria-label="Not available">
					<X className="h-4 w-4" />
					<span className="text-xs">Not available</span>
				</span>
			)
		}
		if (value === 'Partial') {
			return (
				<span
					className="inline-flex rounded-full border border-border/60 bg-background px-2 py-0.5 text-xs text-foreground/80"
					aria-label="Partial"
					title="Available with constraints (region, SKU, or config)"
				>
					Partial
				</span>
			)
		}
		return <span className="text-xs text-foreground/80">{value}</span>
	}

	return (
		<div
			className={cn(
				'mb-4 rounded-2xl border border-border bg-card/60 p-5',
				provider.accent && 'border-primary/30 bg-primary/5',
				className,
			)}
		>
			<h3 className={cn('mb-4 text-lg font-bold', provider.accent ? 'text-primary' : 'text-foreground')}>
				{provider.label}
			</h3>
			<ul className="divide-y divide-border/60">
				{rows.map((row, i) => (
					<li key={i} className="flex items-center justify-between py-3">
						<span className="text-sm text-muted-foreground">{row.feature}</span>
						<div className="ml-4">{renderStatus(row.values[provider.key])}</div>
					</li>
				))}
			</ul>
		</div>
	)
}
