import { RatingStars } from '@rbee/ui/atoms/RatingStars'
import { StarIcon } from '@rbee/ui/atoms'
import type { Sector, Testimonial } from '@rbee/ui/data/testimonials'
import { cn } from '@rbee/ui/utils'
import { Building2, Cpu, Heart, Scale } from 'lucide-react'
import Image from 'next/image'

export interface TestimonialCardProps {
	/** Testimonial data */
	t: Testimonial
	/** Additional CSS classes */
	className?: string
	/** Show verified badge */
	showVerified?: boolean
	/** Animation delay index */
	delayIndex?: number
}

// Legacy props for backward compatibility
export interface LegacyTestimonialCardProps {
	/** Person's name */
	name: string
	/** Person's role */
	role: string
	/** Testimonial quote */
	quote: string
	/** Avatar (image URL or gradient colors) */
	avatar?: string | { from: string; to: string }
	/** Company information */
	company?: { name: string; logo?: string }
	/** Verified badge */
	verified?: boolean
	/** Source link (tweet, GH issue, blog) */
	link?: string
	/** Date of testimonial (ISO or human string) */
	date?: string
	/** Rating (1-5 stars) */
	rating?: 1 | 2 | 3 | 4 | 5
	/** Highlight badge (e.g., "$500/mo → $0") */
	highlight?: string
	/** Additional CSS classes */
	className?: string
}

const SECTOR_ICONS: Record<Sector, typeof Building2> = {
	finance: Building2,
	healthcare: Heart,
	legal: Scale,
	government: Building2,
	provider: Cpu,
}

export function TestimonialCard(props: TestimonialCardProps | LegacyTestimonialCardProps) {
	// Check if using new API or legacy API
	const isNewAPI = 't' in props

	if (isNewAPI) {
		return <NewTestimonialCard {...props} />
	}

	return <LegacyTestimonialCard {...props} />
}

function NewTestimonialCard({ t, className, showVerified = true, delayIndex = 0 }: TestimonialCardProps) {
	const SectorIcon = SECTOR_ICONS[t.sector]
	const authorId = `testimonial-${t.id}-author`
	const roleId = `testimonial-${t.id}-role`

	return (
		<article
			className={cn(
				'h-full flex flex-col rounded-2xl border bg-gradient-to-b from-card to-background p-6',
				'hover:shadow-md transition-shadow',
				className,
			)}
			style={{ animationDelay: `${delayIndex * 60}ms` }}
		>
			{/* Sector Chip */}
			<div className="mb-4 flex items-center gap-2">
				<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
					<SectorIcon className="h-5 w-5 text-primary" aria-hidden="true" />
				</div>
				{t.payout && showVerified && (
					<span className="ml-auto text-[11px] bg-secondary rounded-full px-2 py-0.5 font-medium text-foreground">
						Verified payout
					</span>
				)}
				{!t.payout && showVerified && (
					<span className="ml-auto text-[11px] bg-secondary rounded-full px-2 py-0.5 font-medium text-foreground">
						Verified customer
					</span>
				)}
			</div>

			{/* Author/Role/Org */}
			<div className="mb-3">
				<div className="font-semibold text-foreground" id={authorId}>
					{t.name}
				</div>
				<div className="text-sm text-muted-foreground" id={roleId}>
					{t.role}
					{t.org && ` • ${t.org}`}
					{t.payout && ` • ${t.payout}`}
				</div>
			</div>

			{/* Rating */}
			{t.rating && (
				<div className="mb-3">
					<RatingStars rating={t.rating} size="sm" />
				</div>
			)}

			{/* Quote */}
			<blockquote className="flex-1 mb-4" aria-describedby={`${authorId} ${roleId}`}>
				<p className="text-sm leading-6 text-muted-foreground">
					<cite className="not-italic">
						<span className="text-primary mr-1">&ldquo;</span>
						{t.quote}
						<span className="text-primary ml-1">&rdquo;</span>
					</cite>
				</p>
			</blockquote>
		</article>
	)
}

function LegacyTestimonialCard({
	name,
	role,
	quote,
	avatar,
	company,
	verified,
	link,
	date,
	rating,
	highlight,
	className,
}: LegacyTestimonialCardProps) {
	const gradientClasses = {
		primary: 'from-primary to-primary',
		'chart-1': 'from-chart-1 to-chart-1',
		'chart-2': 'from-chart-2 to-chart-2',
		'chart-3': 'from-chart-3 to-chart-3',
		'chart-4': 'from-chart-4 to-chart-4',
		'chart-5': 'from-chart-5 to-chart-5',
		'primary-chart-2': 'from-primary to-chart-2',
		'chart-1-chart-3': 'from-chart-1 to-chart-3',
		'chart-2-chart-4': 'from-chart-2 to-chart-4',
	}

	const renderAvatar = () => {
		if (!avatar) {
			return (
				<div className="h-12 w-12 rounded-full bg-gradient-to-br from-primary to-chart-2 motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300"></div>
			)
		}

		if (typeof avatar === 'string') {
			return (
				<img
					src={avatar}
					alt={name}
					className="h-12 w-12 rounded-full object-cover motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300"
				/>
			)
		}

		const gradientKey = `${avatar.from}-${avatar.to}` as keyof typeof gradientClasses
		const gradient = gradientClasses[gradientKey] || gradientClasses.primary

		return (
			<div
				className={cn(
					'h-12 w-12 rounded-full bg-gradient-to-br motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300',
					gradient,
				)}
			></div>
		)
	}

	const renderStars = () => {
		if (!rating) return null
		return (
			<div className="flex gap-0.5" aria-label={`Rating: ${rating} out of 5 stars`}>
				{Array.from({ length: 5 }).map((_, i) => (
					<StarIcon key={i} filled={i < rating} />
				))}
			</div>
		)
	}

	return (
		<article
			className={cn(
				'bg-card/90 border rounded-xl p-6 flex flex-col gap-4',
				'hover:border-primary/40 motion-safe:hover:translate-y-[-2px] motion-safe:hover:shadow-lg',
				'motion-safe:transition-all motion-safe:duration-200',
				className,
			)}
			itemScope
			itemType="https://schema.org/Review"
		>
			{/* Header row */}
			<div className="flex items-start justify-between gap-3">
				<div className="flex items-center gap-3 flex-1 min-w-0">
					{renderAvatar()}
					<div className="flex-1 min-w-0">
						<div className="flex items-center gap-2 flex-wrap">
							<span className="font-bold text-card-foreground" itemProp="author">
								{name}
							</span>
							{company?.logo && (
								<Image src={company.logo} alt={company.name} width={24} height={24} className="object-contain" />
							)}
						</div>
						<div className="text-sm text-muted-foreground">{role}</div>
						{company?.name && !company.logo && <div className="text-xs text-muted-foreground/80">{company.name}</div>}
					</div>
				</div>
				{verified && (
					<span className="text-[11px] bg-primary/10 text-primary px-2 py-0.5 rounded-full whitespace-nowrap font-medium">
						Verified
					</span>
				)}
			</div>

			{/* Rating */}
			{renderStars()}

			{/* Quote block */}
			<blockquote className="flex-1">
				<p className="text-sm leading-6 text-muted-foreground line-clamp-6 md:line-clamp-none" itemProp="reviewBody">
					<span className="text-primary mr-1">&ldquo;</span>
					{quote}
				</p>
			</blockquote>

			{/* Footer row */}
			{(highlight || date || link) && (
				<div className="flex items-center justify-between gap-3 text-xs text-muted-foreground/80 flex-wrap">
					<div className="flex items-center gap-2">
						{highlight && (
							<span className="bg-chart-3/10 text-chart-3 px-2 py-1 rounded font-medium whitespace-nowrap">
								{highlight}
							</span>
						)}
						{date && <time dateTime={date}>{date}</time>}
					</div>
					{link && (
						<a
							href={link}
							target="_blank"
							rel="noopener noreferrer"
							className="hover:text-primary transition-colors underline"
						>
							Source
						</a>
					)}
				</div>
			)}
		</article>
	)
}
