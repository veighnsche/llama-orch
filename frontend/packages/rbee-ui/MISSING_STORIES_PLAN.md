# Missing Stories Plan

## EnterprisePage Props → Stories Mapping

Props in EnterprisePageProps.tsx → Expected Story Names:

1. `enterpriseHeroProps` → EnterpriseHero story should have `Enterprise` story ✅ (has OnEnterprisePage, rename it)
2. `enterpriseEmailCaptureProps` → EmailCapture story should have `Enterprise` story ✅ (already has OnEnterprisePage)
3. `enterpriseProblemTemplateProps` → ProblemTemplate story should have `Enterprise` story ❌ MISSING
4. `enterpriseSolutionProps` → SolutionTemplate story should have `Enterprise` story ❌ MISSING
5. `enterpriseComplianceProps` → EnterpriseCompliance story should have `Enterprise` story ✅ (has OnEnterprisePage, rename it)
6. `enterpriseSecurityProps` → EnterpriseSecurity story should have `Enterprise` story ✅ (has OnEnterprisePage, rename it)
7. `enterpriseHowItWorksProps` → EnterpriseHowItWorks story should have `Enterprise` story ✅ (has OnEnterprisePage, rename it)
8. `enterpriseUseCasesProps` → EnterpriseUseCases story should have `Enterprise` story ✅ (has OnEnterprisePage, rename it)
9. `enterpriseComparisonProps` → ComparisonTemplate story should have `Enterprise` story ❌ MISSING
10. `enterpriseFeaturesData` → (SecurityCard data, not a template)
11. `enterpriseTestimonialsData` → TestimonialsTemplate story should have `Enterprise` story ❌ MISSING
12. `enterpriseCTAProps` → EnterpriseCTA story should have `Enterprise` story ✅ (has OnEnterprisePage, rename it)

## Action Plan

### Rename existing "OnXPage" stories to match props names:
- EnterpriseHero: `OnEnterprisePage` → `Enterprise`
- EnterpriseCompliance: `OnEnterprisePage` → `Enterprise`
- EnterpriseSecurity: `OnEnterprisePage` → `Enterprise`
- EnterpriseHowItWorks: `OnEnterprisePage` → `Enterprise`
- EnterpriseUseCases: `OnEnterprisePage` → `Enterprise`
- EnterpriseCTA: `OnEnterprisePage` → `Enterprise`

### Add missing stories:
- ProblemTemplate: Add `Enterprise` story using `enterpriseProblemTemplateProps`
- SolutionTemplate: Add `Enterprise` story using `enterpriseSolutionProps`
- ComparisonTemplate: Add `Enterprise` story using `enterpriseComparisonProps`
- TestimonialsTemplate: Add `Enterprise` story using `enterpriseTestimonialsData`

## Naming Convention

For props named `{page}{Template}Props`, the story should be named `{Page}`.

Examples:
- `enterpriseHeroProps` → `Enterprise` story
- `developersEmailCaptureProps` → `Developers` story
- `homeHeroProps` → `Home` story
- `pricingHeroProps` → `Pricing` story

This makes it easy to find: "Where is enterpriseHeroProps used? Look for 'Enterprise' story in EnterpriseHero.stories.tsx"
