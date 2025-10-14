'use client'

import { SectionContainer } from '@rbee/ui/molecules'
import { IndustryCard, type IndustryCardProps } from '@rbee/ui/molecules/IndustryCard'
import { Banknote, Factory, GraduationCap, Heart, Landmark, Scale } from 'lucide-react'
import Image from 'next/image'

const industries: IndustryCardProps[] = [
	{
		title: 'Financial Services',
		icon: Banknote,
		color: 'primary',
		badge: 'GDPR',
		copy: 'GDPR-ready with audit trails and data residency. Run AI code review and risk analysis without sending financial data to external APIs.',
		anchor: 'finance',
	},
	{
		title: 'Healthcare',
		icon: Heart,
		color: 'chart-2',
		badge: 'HIPAA',
		copy: 'HIPAA-compliant by design. Patient data stays on your network while AI assists with medical coding, documentation, and research.',
		anchor: 'healthcare',
	},
	{
		title: 'Legal',
		icon: Scale,
		color: 'chart-3',
		copy: 'Preserve attorney–client privilege. Perform document/contract analysis and legal research with AI—without client data leaving your environment.',
		anchor: 'legal',
	},
	{
		title: 'Government',
		icon: Landmark,
		color: 'chart-4',
		badge: 'ITAR',
		copy: 'Sovereign, no foreign cloud dependency. Full auditability and policy-enforced routing to meet government security standards.',
		anchor: 'government',
	},
	{
		title: 'Education',
		icon: GraduationCap,
		color: 'chart-2',
		badge: 'FERPA',
		copy: 'Protect student information (FERPA-friendly). AI tutoring, grading assistance, and research tools with zero third-party data sharing.',
		anchor: 'education',
	},
	{
		title: 'Manufacturing',
		icon: Factory,
		color: 'primary',
		copy: 'Safeguard IP and trade secrets. AI-assisted CAD review, quality control, and process optimization—no exposure of proprietary designs.',
		anchor: 'manufacturing',
	},
]

const filters = [
	{ label: 'All', anchor: '#architecture' },
	{ label: 'Finance', anchor: '#finance' },
	{ label: 'Healthcare', anchor: '#healthcare' },
	{ label: 'Legal', anchor: '#legal' },
	{ label: 'Public Sector', anchor: '#government' },
	{ label: 'Education', anchor: '#education' },
	{ label: 'Manufacturing', anchor: '#manufacturing' },
]

export function UseCasesIndustry() {
	const handleFilterClick = (anchor: string) => {
		const element = document.querySelector(anchor)
		if (element) {
			element.scrollIntoView({ behavior: 'smooth', block: 'start' })
		}
	}

	return (
		<SectionContainer
			title="Industry-Specific Solutions"
			bgVariant="secondary"
			subtitle="rbee adapts to the unique compliance and security requirements of regulated industries."
		>
			{/* Header block */}
			<div className="max-w-6xl mx-auto mb-8 animate-in fade-in duration-400">
				<p className="text-center text-sm text-muted-foreground mb-6">Regulated sectors · Private-by-design</p>

				{/* Hero banner */}
				<div className="overflow-hidden rounded-lg border border-border/60 mb-8">
					<Image
						src="/illustrations/industries-hero.svg"
						width={1920}
						height={600}
						priority
						alt="cinematic ultra-wide banner 16:5 seamless collage of six regulated industry environments blended horizontally, professional enterprise photography aesthetic with consistent cool teal accent lighting throughout, LEFT 0-20% FINANCIAL SERVICES: massive circular bank vault door made of polished stainless steel with intricate spoke wheel mechanism in center, digital touchscreen keypad with glowing teal numbers #14b8a6 on right side, vault door is partially open revealing thick 12-inch steel walls, subtle GDPR compliance badge hologram floating in foreground, brushed metal texture with soft reflections, dramatic side lighting creating depth, TRANSITION 20-35% HEALTHCARE: modern hospital server room with floor-to-ceiling black server racks containing blade servers with hundreds of small green and amber status LEDs, prominent white labels reading 'HIPAA COMPLIANT' and 'PHI ENCRYPTED' on rack doors, cool blue LED strip lighting along floor, cable management visible overhead, clean room aesthetic with white tile floor reflecting lights, medical cross symbol subtly integrated into server rack design, TRANSITION 35-50% LEGAL: classical courthouse interior with towering white marble Corinthian columns approximately 30 feet tall, polished marble floor with geometric inlay pattern, bronze scales of justice statue in foreground approximately 4 feet tall with patina finish, dramatic uplighting on columns creating strong vertical lines, subtle teal accent light along column bases, formal and authoritative atmosphere, TRANSITION 50-65% GOVERNMENT: official government seal approximately 6 feet diameter mounted on dark navy wall, seal features eagle with shield and olive branches in gold metallic finish, security clearance badge with holographic elements floating in foreground showing 'TOP SECRET' and 'ITAR COMPLIANT' text, red and blue accent lighting creating patriotic feel mixed with teal brand color, professional government facility aesthetic, TRANSITION 65-80% EDUCATION: modern university classroom or computer lab with rows of sleek monitors showing encrypted login screens with padlock icons, screens display 'FERPA PROTECTED' watermarks, teal accent lighting under desks, clean minimalist design with white walls and light wood furniture, digital learning interface visible on screens with privacy shields, contemporary educational technology aesthetic, RIGHT 80-100% MANUFACTURING: industrial factory floor with precision assembly line showing robotic arms in motion blur, large 'INTELLECTUAL PROPERTY PROTECTED' warning signs with shield icons, orange safety lighting mixed with teal accent strips, visible CAD workstation monitors showing 3D models with 'CONFIDENTIAL' watermarks, industrial metal surfaces with rivets and warning stripes, high-tech manufacturing aesthetic, SEAMLESS TRANSITIONS: each environment blends into next using subtle gradient masks and shared lighting, no hard edges between sections, consistent depth of field with foreground sharp and background slightly soft, OVERALL LIGHTING: cool color temperature base with warm accents, teal #14b8a6 accent lighting present in all sections creating brand cohesion, professional studio lighting with soft shadows, subtle rim lighting separating subjects from backgrounds, ATMOSPHERE: dark enterprise aesthetic with deep navy #0f172a and charcoal #1e293b backgrounds, professional and secure mood, conveys trust and compliance, high-end corporate photography style similar to Fortune 500 annual reports or IBM security marketing, shot with wide-angle lens creating slight perspective, f/5.6 aperture for good depth across wide scene, cinematic color grading with lifted blacks and teal/orange color palette, subtle vignette darkening edges, 1920x600 pixels 16:5 aspect ratio, extremely high detail showing textures of metal, marble, screens, and industrial surfaces, compliance badges and security indicators should be clearly visible but not overwhelming, overall composition should flow smoothly left to right telling story of comprehensive industry coverage"
						className="w-full h-28 md:h-40 object-cover"
					/>
				</div>

				{/* Filter pills */}
				<nav
					aria-label="Filter industries"
					className="flex flex-wrap items-center justify-center gap-2 animate-in slide-in-from-top-2 duration-400 delay-75"
				>
					{filters.map((filter) => (
						<button
							key={filter.label}
							onClick={() => handleFilterClick(filter.anchor)}
							className="inline-flex items-center rounded-full border border-border/60 bg-card px-4 py-2 text-sm font-medium text-foreground hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
						>
							{filter.label}
						</button>
					))}
				</nav>
			</div>

			{/* Responsive grid: 1 col mobile, 2 cols tablet, 3 cols desktop */}
			<div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 lg:gap-8 max-w-6xl mx-auto">
				{industries.map((industry, index) => (
					<IndustryCard key={industry.title} {...industry} style={{ animationDelay: `${index * 60}ms` }} />
				))}
			</div>
		</SectionContainer>
	)
}
