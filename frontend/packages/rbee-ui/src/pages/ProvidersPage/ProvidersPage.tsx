"use client";

import { TemplateContainer } from "@rbee/ui/molecules";
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
} from "@rbee/ui/templates";
import {
  providersCTAProps,
  providersEarningsContainerProps,
  providersEarningsProps,
  providersFeaturesContainerProps,
  providersFeaturesProps,
  providersHeroProps,
  providersHowItWorksContainerProps,
  providersHowItWorksProps,
  providersMarketplaceContainerProps,
  providersMarketplaceSolutionProps,
  providersProblemContainerProps,
  providersProblemProps,
  providersSecurityContainerProps,
  providersSecurityGridProps,
  providersSolutionContainerProps,
  providersSolutionProps,
  providersTestimonialsContainerProps,
  providersTestimonialsProps,
  providersUseCasesContainerProps,
  providersUseCasesGridProps,
} from "./ProvidersPageProps";

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
      <TemplateContainer {...providersFeaturesContainerProps}>
        <FeaturesTabs {...providersFeaturesProps} />
      </TemplateContainer>
      <TemplateContainer {...providersUseCasesContainerProps}>
        <CardGridTemplate {...providersUseCasesGridProps} />
      </TemplateContainer>
      <TemplateContainer {...providersEarningsContainerProps}>
        <ProvidersEarnings {...providersEarningsProps} />
      </TemplateContainer>
      <TemplateContainer {...providersMarketplaceContainerProps}>
        <SolutionTemplate {...providersMarketplaceSolutionProps} />
      </TemplateContainer>
      <TemplateContainer {...providersSecurityContainerProps}>
        <CardGridTemplate {...providersSecurityGridProps} />
      </TemplateContainer>
      <TemplateContainer {...providersTestimonialsContainerProps}>
        <TestimonialsTemplate {...providersTestimonialsProps} />
      </TemplateContainer>
      <ProvidersCTA {...providersCTAProps} />
    </main>
  );
}
