'use client'

import { ProvidersSecurityCard, TemplateContainer } from '@rbee/ui/molecules'
import { ProvidersCaseCard } from '@rbee/ui/organisms'
import {
  FeaturesTabs,
  HowItWorks,
  ProblemTemplate,
  ProvidersCTA,
  ProvidersEarnings,
  ProvidersHero,
  SolutionTemplate,
  TestimonialsTemplate,
} from '@rbee/ui/templates'
import {
  providersCTAProps,
  providersEarningsContainerProps,
  providersEarningsProps,
  providersFeaturesProps,
  providersHeroProps,
  providersHowItWorksContainerProps,
  providersHowItWorksProps,
  providersMarketplaceContainerProps,
  providersMarketplaceSolutionProps,
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
      <ProvidersHero {...providersHeroProps} />
      <TemplateContainer {...providersProblemContainerProps}>
        <ProblemTemplate {...providersProblemProps} />
      </TemplateContainer>
      <TemplateContainer {...providersSolutionContainerProps}>
        <SolutionTemplate {...providersSolutionProps} />
      </TemplateContainer>
      <TemplateContainer {...providersHowItWorksContainerProps}>
        <HowItWorks {...providersHowItWorksProps} />
      </TemplateContainer>
      <FeaturesTabs {...providersFeaturesProps} />
      <TemplateContainer {...providersUseCasesContainerProps}>
        <div className="mx-auto max-w-[60%] grid gap-6 grid-cols-2">
          {providersUseCasesProps.cases.map((caseData, index) => (
            <ProvidersCaseCard
              key={index}
              icon={caseData.icon}
              title={caseData.title}
              subtitle={caseData.subtitle}
              quote={caseData.quote}
              facts={caseData.facts}
              highlight={caseData.highlight}
              index={index}
            />
          ))}
        </div>
      </TemplateContainer>
      <TemplateContainer {...providersEarningsContainerProps}>
        <ProvidersEarnings {...providersEarningsProps} />
      </TemplateContainer>
      <TemplateContainer {...providersMarketplaceContainerProps}>
        <SolutionTemplate {...providersMarketplaceSolutionProps} />
      </TemplateContainer>
      <TemplateContainer {...providersSecurityContainerProps}>
        <div className="mx-auto max-w-[60%] grid gap-6 grid-cols-2">
          {providersSecurityProps.items.map((item, index) => (
            <ProvidersSecurityCard
              key={index}
              icon={item.icon}
              title={item.title}
              subtitle={item.subtitle}
              body={item.body}
              points={item.points}
              index={index}
            />
          ))}
        </div>
      </TemplateContainer>
      <TemplateContainer {...providersTestimonialsContainerProps}>
        <TestimonialsTemplate {...providersTestimonialsProps} />
      </TemplateContainer>
      <ProvidersCTA {...providersCTAProps} />
    </main>
  )
}
