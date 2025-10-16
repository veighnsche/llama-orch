'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import { FeatureTabsSection, StepsSection } from '@rbee/ui/organisms'
import {
  ProblemTemplate,
  ProvidersCTATemplate,
  ProvidersEarningsTemplate,
  ProvidersHeroTemplate,
  ProvidersMarketplaceTemplate,
  ProvidersSecurityTemplate,
  ProvidersTestimonialsTemplate,
  ProvidersUseCasesTemplate,
  SolutionTemplate,
} from '@rbee/ui/templates'
import {
  providersCTAProps,
  providersEarningsContainerProps,
  providersEarningsProps,
  providersFeaturesProps,
  providersHeroProps,
  providersHowItWorksProps,
  providersMarketplaceContainerProps,
  providersMarketplaceProps,
  providersProblemContainerProps,
  providersProblemProps,
  providersSecurityContainerProps,
  providersSecurityProps,
  providersSolutionContainerProps,
  providersSolutionProps,
  providersTestimonialsContainerProps,
  providersTestimonialsProps,
  providersUseCasesContainerProps,
  providersUseCasesProps,
} from './ProvidersPageProps'

export default function ProvidersPage() {
  return (
    <main>
      <ProvidersHeroTemplate {...providersHeroProps} />
      <TemplateContainer {...providersProblemContainerProps}>
        <ProblemTemplate {...providersProblemProps} />
      </TemplateContainer>
      <TemplateContainer {...providersSolutionContainerProps}>
        <SolutionTemplate {...providersSolutionProps} />
      </TemplateContainer>
      <StepsSection {...providersHowItWorksProps} />
      <FeatureTabsSection {...providersFeaturesProps} />
      <TemplateContainer {...providersUseCasesContainerProps}>
        <ProvidersUseCasesTemplate {...providersUseCasesProps} />
      </TemplateContainer>
      <TemplateContainer {...providersEarningsContainerProps}>
        <ProvidersEarningsTemplate {...providersEarningsProps} />
      </TemplateContainer>
      <TemplateContainer {...providersMarketplaceContainerProps}>
        <ProvidersMarketplaceTemplate {...providersMarketplaceProps} />
      </TemplateContainer>
      <TemplateContainer {...providersSecurityContainerProps}>
        <ProvidersSecurityTemplate {...providersSecurityProps} />
      </TemplateContainer>
      <TemplateContainer {...providersTestimonialsContainerProps}>
        <ProvidersTestimonialsTemplate {...providersTestimonialsProps} />
      </TemplateContainer>
      <ProvidersCTATemplate {...providersCTAProps} />
    </main>
  )
}
