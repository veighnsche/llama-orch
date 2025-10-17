import { TemplateContainer } from '@rbee/ui/molecules'
import {
  CrossNodeOrchestration,
  CTATemplate,
  EmailCapture,
  FAQTemplate,
  HeroTemplate,
  HowItWorks,
  MultiBackendGpuTemplate,
  ProblemTemplate,
  ProvidersEarnings,
  SecurityIsolation,
  SolutionTemplate,
  UseCasesTemplate,
} from '@rbee/ui/templates'
import {
  homelabCrossNodeContainerProps,
  homelabCrossNodeProps,
  homelabCTAContainerProps,
  homelabCTAProps,
  homelabEmailCaptureContainerProps,
  homelabEmailCaptureProps,
  homelabFAQContainerProps,
  homelabFAQProps,
  homelabHeroContainerProps,
  homelabHeroProps,
  homelabHowItWorksContainerProps,
  homelabHowItWorksProps,
  homelabMultiBackendContainerProps,
  homelabMultiBackendProps,
  homelabPowerCostContainerProps,
  homelabPowerCostProps,
  homelabProblemContainerProps,
  homelabProblemProps,
  homelabSecurityContainerProps,
  homelabSecurityProps,
  homelabSolutionContainerProps,
  homelabSolutionProps,
  homelabUseCasesContainerProps,
  homelabUseCasesProps,
} from './HomelabPageProps'

/**
 * HomelabPage - Homelab & Self-Hosting page
 *
 * Target audience: Homelab enthusiasts, self-hosters, hardware tinkerers, privacy advocates
 *
 * Key message: Self-hosted LLMs across all your machines. Turn idle hardware into productive AI infrastructure.
 *
 * @example
 * ```tsx
 * import HomelabPage from '@rbee/ui/pages/HomelabPage'
 *
 * <HomelabPage />
 * ```
 */
export default function HomelabPage() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <TemplateContainer {...homelabHeroContainerProps}>
        <HeroTemplate {...homelabHeroProps} />
      </TemplateContainer>

      {/* Email Capture - Setup Guide */}
      <TemplateContainer {...homelabEmailCaptureContainerProps}>
        <EmailCapture {...homelabEmailCaptureProps} />
      </TemplateContainer>

      {/* Problem Section - Homelab Complexity */}
      <TemplateContainer {...homelabProblemContainerProps}>
        <ProblemTemplate {...homelabProblemProps} />
      </TemplateContainer>

      {/* Solution Section - Unified Orchestration */}
      <TemplateContainer {...homelabSolutionContainerProps}>
        <SolutionTemplate {...homelabSolutionProps} />
      </TemplateContainer>

      {/* How It Works - Setup Steps */}
      <TemplateContainer {...homelabHowItWorksContainerProps}>
        <HowItWorks {...homelabHowItWorksProps} />
      </TemplateContainer>

      {/* Cross-Node Orchestration - Multi-Machine Visualization */}
      <TemplateContainer {...homelabCrossNodeContainerProps}>
        <CrossNodeOrchestration {...homelabCrossNodeProps} />
      </TemplateContainer>

      {/* Multi-Backend GPU - Hardware Support */}
      <TemplateContainer {...homelabMultiBackendContainerProps}>
        <MultiBackendGpuTemplate {...homelabMultiBackendProps} />
      </TemplateContainer>

      {/* Power Cost Calculator */}
      <TemplateContainer {...homelabPowerCostContainerProps}>
        <ProvidersEarnings {...homelabPowerCostProps} />
      </TemplateContainer>

      {/* Use Cases - Homelab Scenarios */}
      <TemplateContainer {...homelabUseCasesContainerProps}>
        <UseCasesTemplate {...homelabUseCasesProps} />
      </TemplateContainer>

      {/* Security & Privacy */}
      <TemplateContainer {...homelabSecurityContainerProps}>
        <SecurityIsolation {...homelabSecurityProps} />
      </TemplateContainer>

      {/* FAQ Section */}
      <TemplateContainer {...homelabFAQContainerProps}>
        <FAQTemplate {...homelabFAQProps} />
      </TemplateContainer>

      {/* Final CTA */}
      <TemplateContainer {...homelabCTAContainerProps}>
        <CTATemplate {...homelabCTAProps} />
      </TemplateContainer>
    </div>
  )
}
