"use client";

import { CommissionStructureCard, TemplateContainer } from "@rbee/ui/molecules";
import {
  FeaturesTabs,
  HowItWorks,
  ProblemTemplate,
  ProvidersCTA,
  ProvidersEarnings,
  ProvidersHero,
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
  providersMarketplaceSolutionProps,
  providersMarketplaceCommissionProps,
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
        <SolutionTemplate
          {...providersMarketplaceSolutionProps}
          aside={
            <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
              <CommissionStructureCard {...providersMarketplaceCommissionProps} />
            </div>
          }
        />
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
