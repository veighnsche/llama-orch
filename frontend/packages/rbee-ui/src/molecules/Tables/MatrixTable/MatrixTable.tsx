import { cn } from '@rbee/ui/utils'
import { Check, X } from 'lucide-react'

export interface Provider {
	key: string
	label: string
	accent?: boolean
}

export interface Row {
	feature: string
	values: Record<string, boolean | 'Partial' | string>
	note?: string
}

export interface MatrixTableProps {
	columns: Provider[]
	rows: Row[]
	className?: string
}

export function MatrixTable({ columns, rows, className }: MatrixTableProps) {
	const renderCell = (value: boolean | 'Partial' | string, providerKey: string) => {
		if (value === true) {
			return <Check className="mx-auto h-5 w-5 text-chart-3" aria-label="Included" />
		}
		if (value === false) {
			return <X className="mx-auto h-5 w-5 text-destructive" aria-label="Not available" />
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
		<div className={cn('overflow-x-auto', className)}>
			<table className="w-full border-collapse">
				<caption className="sr-only">Compliance & security feature comparison across providers</caption>
				<thead>
					<tr className="border-b border-border/80">
						<th scope="col" className="p-3 text-left text-xs uppercase tracking-wide text-muted-foreground">
							Feature
						</th>
						{columns.map((col) => (
							<th
								key={col.key}
								scope="col"
								className={cn(
									'p-3 text-center text-xs uppercase tracking-wide',
									col.accent ? 'bg-primary/5 font-bold text-primary' : 'text-muted-foreground',
								)}
							>
								{col.label}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{rows.map((row, i) => (
						<tr
							key={i}
							className="border-b border-border/80 transition-colors hover:bg-secondary/30 odd:bg-background even:bg-background/60"
						>
							<th scope="row" className="p-3 text-left text-sm font-normal text-muted-foreground">
								{row.feature}
							</th>
							{columns.map((col) => (
								<td key={col.key} className={cn('p-3 text-center', col.accent && 'bg-primary/5')}>
									{renderCell(row.values[col.key], col.key)}
								</td>
							))}
						</tr>
					))}
				</tbody>
			</table>
		</div>
	)
}
