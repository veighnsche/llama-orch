'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  CardGridTemplate,
  CTATemplate,
  EmailCapture,
  EnterpriseSecurity,
  FAQTemplate,
  HeroTemplate,
  HowItWorks,
  PricingTemplate,
  ProblemTemplate,
  SolutionTemplate,
  TestimonialsTemplate,
  UseCasesTemplate,
} from '@rbee/ui/templates'
import {
  educationCTAContainerProps,
  educationCTAProps,
  educationCourseLevelsContainerProps,
  educationCourseLevelsProps,
  educationCurriculumContainerProps,
  educationCurriculumProps,
  educationEmailCaptureContainerProps,
  educationEmailCaptureProps,
  educationFAQContainerProps,
  educationFAQProps,
  educationHeroContainerProps,
  educationHeroProps,
  educationLabExercisesContainerProps,
  educationLabExercisesProps,
  educationProblemTemplateContainerProps,
  educationProblemTemplateProps,
  educationResourcesGridContainerProps,
  educationResourcesGridProps,
  educationSolutionContainerProps,
  educationSolutionProps,
  educationStudentTypesContainerProps,
  educationStudentTypesProps,
  educationTestimonialsContainerProps,
  educationTestimonialsData,
} from './EducationPageProps'

export default function EducationPage() {
  return (
    <main>
      <TemplateContainer {...educationHeroContainerProps}>
        <HeroTemplate {...educationHeroProps} />
      </TemplateContainer>

      <TemplateContainer {...educationEmailCaptureContainerProps}>
        <EmailCapture {...educationEmailCaptureProps} />
      </TemplateContainer>

      <TemplateContainer {...educationProblemTemplateContainerProps}>
        <ProblemTemplate {...educationProblemTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...educationSolutionContainerProps}>
        <SolutionTemplate {...educationSolutionProps} />
      </TemplateContainer>

      <TemplateContainer {...educationCourseLevelsContainerProps}>
        <PricingTemplate {...educationCourseLevelsProps} />
      </TemplateContainer>

      <TemplateContainer {...educationCurriculumContainerProps}>
        <EnterpriseSecurity {...educationCurriculumProps} />
      </TemplateContainer>

      <TemplateContainer {...educationLabExercisesContainerProps}>
        <HowItWorks {...educationLabExercisesProps} />
      </TemplateContainer>

      <TemplateContainer {...educationStudentTypesContainerProps}>
        <UseCasesTemplate {...educationStudentTypesProps} />
      </TemplateContainer>

      <TemplateContainer {...educationTestimonialsContainerProps}>
        <TestimonialsTemplate {...educationTestimonialsData} />
      </TemplateContainer>

      <TemplateContainer {...educationResourcesGridContainerProps}>
        <CardGridTemplate {...educationResourcesGridProps} />
      </TemplateContainer>

      <TemplateContainer {...educationFAQContainerProps}>
        <FAQTemplate {...educationFAQProps} />
      </TemplateContainer>

      <TemplateContainer {...educationCTAContainerProps}>
        <CTATemplate {...educationCTAProps} />
      </TemplateContainer>
    </main>
  )
}
