export interface FeatureHeaderProps {
	title: string
	subtitle: string
}

/**
 * A consistent header pattern for feature sections with a large title
 * and smaller subtitle text.
 */
export function FeatureHeader({ title, subtitle }: FeatureHeaderProps) {
	return (
		<div className="space-y-2">
			<h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">{title}</h3>
			<p className="text-xs text-muted-foreground">{subtitle}</p>
		</div>
	)
}
