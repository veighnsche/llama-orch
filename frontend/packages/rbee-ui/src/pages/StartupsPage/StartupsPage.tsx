import { TemplateContainer } from '@rbee/ui/molecules'
import {
  ComparisonTemplate,
  CTATemplate,
  EmailCapture,
  EnterpriseHowItWorks,
  FAQTemplate,
  HeroTemplate,
  ProblemTemplate,
  ProvidersEarnings,
  SolutionTemplate,
  TechnicalTemplate,
  TestimonialsTemplate,
  UseCasesTemplate,
} from '@rbee/ui/templates'
import {
  startupsComparisonContainerProps,
  startupsComparisonProps,
  startupsCTAContainerProps,
  startupsCTAProps,
  startupsEmailCaptureContainerProps,
  startupsEmailCaptureProps,
  startupsFAQContainerProps,
  startupsFAQProps,
  startupsGrowthRoadmapContainerProps,
  startupsGrowthRoadmapProps,
  startupsHeroProps,
  startupsProblemContainerProps,
  startupsProblemProps,
  startupsROICalculatorContainerProps,
  startupsROICalculatorProps,
  startupsSolutionContainerProps,
  startupsSolutionProps,
  startupsTechnicalContainerProps,
  startupsTechnicalProps,
  startupsTestimonialsContainerProps,
  startupsTestimonialsProps,
  startupsUseCasesContainerProps,
  startupsUseCasesProps,
} from './StartupsPageProps'

/**
 * StartupsPage - Landing page for startups and small teams
 *
 * Target audience: Startup founders, small dev teams, bootstrapped companies
 * Key message: Build AI products without burning cash on API fees
 *
 * @example
 * ```tsx
 * <StartupsPage />
 * ```
 */
export default function StartupsPage() {
  return (
    <main>
      {/* Hero */}
      <HeroTemplate {...startupsHeroProps} />

      {/* Email Capture */}
      <TemplateContainer {...startupsEmailCaptureContainerProps}>
        <EmailCapture {...startupsEmailCaptureProps} />
      </TemplateContainer>

      {/* Problem: API Cost Spiral */}
      <TemplateContainer {...startupsProblemContainerProps}>
        <ProblemTemplate {...startupsProblemProps} />
      </TemplateContainer>

      {/* Solution: Own Your Stack */}
      <TemplateContainer {...startupsSolutionContainerProps}>
        <SolutionTemplate {...startupsSolutionProps} />
      </TemplateContainer>

      {/* ROI Calculator */}
      <TemplateContainer {...startupsROICalculatorContainerProps}>
        <ProvidersEarnings {...startupsROICalculatorProps} />
      </TemplateContainer>

      {/* Growth Roadmap */}
      <TemplateContainer {...startupsGrowthRoadmapContainerProps}>
        <EnterpriseHowItWorks {...startupsGrowthRoadmapProps} />
      </TemplateContainer>

      {/* Startup Scenarios */}
      <TemplateContainer {...startupsUseCasesContainerProps}>
        <UseCasesTemplate {...startupsUseCasesProps} />
      </TemplateContainer>

      {/* Comparison: rbee vs API Providers */}
      <TemplateContainer {...startupsComparisonContainerProps}>
        <ComparisonTemplate {...startupsComparisonProps} />
      </TemplateContainer>

      {/* Technical Stack */}
      <TemplateContainer {...startupsTechnicalContainerProps}>
        <TechnicalTemplate {...startupsTechnicalProps} />
      </TemplateContainer>

      {/* Testimonials */}
      <TemplateContainer {...startupsTestimonialsContainerProps}>
        <TestimonialsTemplate {...startupsTestimonialsProps} />
      </TemplateContainer>

      {/* FAQ */}
      <TemplateContainer {...startupsFAQContainerProps}>
        <FAQTemplate {...startupsFAQProps} />
      </TemplateContainer>

      {/* Final CTA */}
      <TemplateContainer {...startupsCTAContainerProps}>
        <CTATemplate {...startupsCTAProps} />
      </TemplateContainer>
    </main>
  )
}
