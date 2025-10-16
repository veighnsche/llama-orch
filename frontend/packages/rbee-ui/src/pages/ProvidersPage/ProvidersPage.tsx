"use client";

import { TemplateContainer } from "@rbee/ui/molecules";
import {
  FeaturesTabs,
  HowItWorks,
  ProblemTemplate,
  ProvidersCTA,
  ProvidersEarnings,
  ProvidersHero,
  ProvidersMarketplace,
  ProvidersSecurityTemplate,
  ProvidersTestimonialsTemplate,
  ProvidersUseCasesTemplate,
  SolutionTemplate,
} from "@rbee/ui/templates";
import {
  providersCTAProps,
  providersEarningsContainerProps,
  providersEarningsProps,
  providersFeaturesProps,
  providersHeroProps,
  providersHowItWorksContainerProps,
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
      <FeaturesTabs {...providersFeaturesProps} />
      <TemplateContainer {...providersUseCasesContainerProps}>
        <ProvidersUseCasesTemplate {...providersUseCasesProps} />
      </TemplateContainer>
      <TemplateContainer {...providersEarningsContainerProps}>
        <ProvidersEarnings {...providersEarningsProps} />
      </TemplateContainer>
      <TemplateContainer {...providersMarketplaceContainerProps}>
        <ProvidersMarketplace {...providersMarketplaceProps} />
      </TemplateContainer>
      <TemplateContainer {...providersSecurityContainerProps}>
        <ProvidersSecurityTemplate {...providersSecurityProps} />
      </TemplateContainer>
      <TemplateContainer {...providersTestimonialsContainerProps}>
        <ProvidersTestimonialsTemplate {...providersTestimonialsProps} />
      </TemplateContainer>
      <ProvidersCTA {...providersCTAProps} />
    </main>
  );
}
