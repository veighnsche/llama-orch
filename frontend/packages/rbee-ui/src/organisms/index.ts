// Barrel exports for all organisms
export * from './AudienceSelector'
export * from './CodeExamplesSection'
export * from './ComparisonSection'
export * from './CtaSection'
export * from './Developers'
export * from './EmailCapture'
export * from './Enterprise'
export * from './FaqSection'
export * from './Features'
export * from './FeaturesSection'
export * from './FeatureTabsSection'
export * from './Footer'
export * from './HeroSection'
export * from './HowItWorksSection'
export * from './Navigation'
export * from './Pricing'
export * from './PricingSection'
export * from './ProblemSection'
export type {
	Case as ProvidersCase,
	UseCasesSectionProps as ProvidersUseCasesSectionProps,
} from './Providers'
// Providers exports (contains some components that conflict with standalone versions)
export {
	CTASectionProviders,
	ProvidersCTA,
	ProvidersEarnings,
	ProvidersFeatures,
	ProvidersHero,
	ProvidersHowItWorks,
	ProvidersMarketplace,
	ProvidersProblem,
	ProvidersSecurity,
	ProvidersSolution,
	ProvidersTestimonials,
	ProvidersUseCases,
	SecuritySection,
	// Re-export with Providers prefix to avoid conflicts with standalone organisms
	SocialProofSection as ProvidersSocialProofSection,
	UseCasesSection as ProvidersUseCasesSection,
} from './Providers'

// Standalone organisms (may have same names as Providers sub-components)
export * from './SocialProofSection'
export type {
	EarningRow,
	Earnings,
	Feature as SolutionFeature,
	SolutionSectionProps,
	Step as SolutionStep,
} from './SolutionSection'

// Solution section exports Step type which conflicts with StepsSection
export { HomeSolutionSection, SolutionSection } from './SolutionSection'
export type { Step as TimelineStep, StepsSectionProps } from './StepsSection'

// Steps section exports Step type
export { StepsSection } from './StepsSection'
export * from './TechnicalSection'
export * from './TestimonialsRail'
export * from './TopologyDiagram'
export * from './UseCases'
export * from './UseCasesSection'
export * from './WhatIsRbee'
