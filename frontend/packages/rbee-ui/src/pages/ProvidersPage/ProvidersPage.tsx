'use client'

import { ProvidersSecurityCard } from '@rbee/ui/molecules'
import { ProvidersCaseCard, SectionContainer } from '@rbee/ui/organisms'
import {
  CardGridTemplate,
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
      <SectionContainer {...providersProblemContainerProps}>
        <ProblemTemplate {...providersProblemProps} />
      </SectionContainer>
      <SectionContainer {...providersSolutionContainerProps}>
        <SolutionTemplate {...providersSolutionProps} />
      </SectionContainer>
      <SectionContainer {...providersHowItWorksContainerProps}>
        <HowItWorks {...providersHowItWorksProps} />
      </SectionContainer>
      <FeaturesTabs {...providersFeaturesProps} />
      <SectionContainer {...providersUseCasesContainerProps}>
        <CardGridTemplate>
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
        </CardGridTemplate>
      </SectionContainer>
      <SectionContainer {...providersEarningsContainerProps}>
        <ProvidersEarnings {...providersEarningsProps} />
      </SectionContainer>
      <SectionContainer {...providersMarketplaceContainerProps}>
        <SolutionTemplate {...providersMarketplaceSolutionProps} />
      </SectionContainer>
      <SectionContainer {...providersSecurityContainerProps}>
        <CardGridTemplate>
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
        </CardGridTemplate>
      </SectionContainer>
      <SectionContainer {...providersTestimonialsContainerProps}>
        <TestimonialsTemplate {...providersTestimonialsProps} />
      </SectionContainer>
      <ProvidersCTA {...providersCTAProps} />
    </main>
  )
}
