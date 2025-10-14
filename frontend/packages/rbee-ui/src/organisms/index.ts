// Barrel exports for all organisms
export * from './AudienceSelector'
export * from './CodeExamplesSection'
export * from './ComparisonSection'
export * from './CtaSection'
export * from './Developers'
export * from './EmailCapture'
export * from './Enterprise'
export * from './FaqSection'
export * from './FeatureTabsSection'
export * from './Features'
export * from './FeaturesSection'
export * from './Footer'
export * from './HeroSection'
export * from './HowItWorksSection'
export * from './Navigation'
export * from './Pricing'
export * from './PricingSection'
export * from './ProblemSection'

// Providers exports (contains some components that conflict with standalone versions)
export {
  ProvidersHero,
  ProvidersProblem,
  ProvidersSolution,
  ProvidersHowItWorks,
  ProvidersFeatures,
  ProvidersUseCases,
  ProvidersEarnings,
  ProvidersMarketplace,
  ProvidersSecurity,
  ProvidersTestimonials,
  ProvidersCTA,
  CTASectionProviders,
  SecuritySection,
  // Re-export with Providers prefix to avoid conflicts with standalone organisms
  SocialProofSection as ProvidersSocialProofSection,
  UseCasesSection as ProvidersUseCasesSection,
} from './Providers'
export type {
  Case as ProvidersCase,
  UseCasesSectionProps as ProvidersUseCasesSectionProps,
} from './Providers'

// Standalone organisms (may have same names as Providers sub-components)
export * from './SocialProofSection'
export * from './UseCasesSection'

// Solution section exports Step type which conflicts with StepsSection
export { SolutionSection, HomeSolutionSection } from './SolutionSection'
export type {
  SolutionSectionProps,
  Feature as SolutionFeature,
  Step as SolutionStep,
  EarningRow,
  Earnings,
} from './SolutionSection'

// Steps section exports Step type
export { StepsSection } from './StepsSection'
export type { StepsSectionProps, Step as TimelineStep } from './StepsSection'

export * from './TechnicalSection'
export * from './TestimonialsRail'
export * from './TopologyDiagram'
export * from './UseCases'
export * from './WhatIsRbee'
