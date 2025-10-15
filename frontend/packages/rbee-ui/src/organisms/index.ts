// Barrel exports for all organisms

// Home page organisms
export * from './Home/CodeExamplesSection'
export * from './Home/ComparisonSection'
export * from './Home/CtaSection'
export * from './EmailCapture'
export * from './Home/FaqSection'
export * from './Home/FeaturesSection'
export * from './Home/FeatureTabsSection'
export * from './Home/HeroSection'
export * from './Home/HowItWorksSection'
export * from './Home/PricingSection'
export * from './Home/ProblemSection'
export * from './Home/SocialProofSection'
export * from './Home/TestimonialsRail'
export * from './Home/TopologyDiagram'
export * from './Home/WhatIsRbee'

// Shared organisms
export * from './Home/AudienceSelector'
export * from './Shared/Footer'
export * from './Shared/Navigation'

// Page-specific organisms
export * from './Developers'
export * from './Enterprise'
export * from './Features'
export * from './Pricing'
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

export type {
	EarningRow,
	Earnings,
	Feature as SolutionFeature,
	SolutionSectionProps,
	Step as SolutionStep,
} from './Home/SolutionSection'

// Solution section exports Step type which conflicts with StepsSection
export { HomeSolutionSection, SolutionSection } from './Home/SolutionSection'
export type { Step as TimelineStep, StepsSectionProps } from './Home/StepsSection'

// Steps section exports Step type
export { StepsSection } from './Home/StepsSection'
export * from './Home/TechnicalSection'
export * from './UseCases'
export * from './UseCasesSection'
