# Optimal Background Strategy - All Pages

## Background Alternation Pattern

**Rule:** Alternate between `background` and `secondary` to create visual rhythm.
**Exception:** Problem sections always use `destructive-gradient`.

## HomePage

1. HomeHero - Built-in (honeycomb)
2. WhatIsRbee - `background`
3. AudienceSelector - `secondary` (alternation)
4. EmailCapture - `background`
5. Problem - `destructive-gradient` ⚠️
6. Solution - `background`
7. HowItWorks - `secondary` (alternation)
8. FeaturesTabs - `background`
9. UseCases - `secondary` (alternation)
10. Comparison - `background`
11. Pricing - `secondary` (alternation)
12. Testimonials - `background`
13. Technical - `secondary` (alternation)
14. FAQ - `background`
15. CTA - NO CONTAINER (has own styling)

## DevelopersPage

1. DevelopersHero - Built-in
2. EmailCapture - `background`
3. Problem - `destructive-gradient` ⚠️
4. Solution - `background`
5. HowItWorks - `secondary` (alternation)
6. FeaturesTabs - `background`
7. UseCases - `secondary` (alternation)
8. CodeExamples - `background`
9. Pricing - `secondary` (alternation)
10. Testimonials - `background`
11. CTA - NO CONTAINER

## EnterprisePage

1. EnterpriseHero - Built-in
2. EmailCapture - `background` ⚠️ MISSING CONTAINER
3. Problem - `destructive-gradient`
4. Solution - `background`
5. Compliance - `secondary` (alternation)
6. Security - `background`
7. HowItWorks - `secondary` (alternation)
8. UseCases - `background`
9. Comparison - `secondary` (alternation)
10. Features - `background`
11. Testimonials - `secondary` (alternation)
12. CTA - `background`

## FeaturesPage

1. FeaturesHero - Built-in
2. FeaturesTabs - `background` ⚠️ MISSING CONTAINER
3. CrossNodeOrchestration - `secondary` (alternation)
4. IntelligentModelManagement - `background`
5. MultiBackendGpu - `secondary` (alternation)
6. ErrorHandling - `background`
7. RealTimeProgress - `secondary` (alternation)
8. SecurityIsolation - `background`
9. AdditionalFeaturesGrid - `secondary` (alternation)
10. EmailCapture - `background` ⚠️ MISSING CONTAINER

## PricingPage

1. PricingHero - Built-in
2. Pricing - `background`
3. Comparison - `secondary` (alternation)
4. FAQ - `background`
5. EmailCapture - `secondary` (alternation) ⚠️ MISSING CONTAINER

## ProvidersPage

1. ProvidersHero - Built-in
2. Problem - `destructive-gradient`
3. Solution - `background`
4. HowItWorks - `secondary` (alternation)
5. FeaturesTabs - `background` ⚠️ MISSING CONTAINER
6. UseCases - `secondary` (alternation)
7. Earnings - `background`
8. Marketplace - `secondary` (alternation)
9. Security - `background`
10. Testimonials - `secondary` (alternation)
11. CTA - NO CONTAINER

## UseCasesPage

1. UseCasesHero - Built-in
2. Primary - `background`
3. Industry - `secondary` (alternation)
4. EmailCapture - `background` ⚠️ MISSING CONTAINER

## Actions Required

### Add Missing Containers:
1. EnterprisePage - EmailCapture
2. FeaturesPage - EmailCapture + FeaturesTabs
3. PricingPage - EmailCapture
4. ProvidersPage - FeaturesTabs
5. UseCasesPage - EmailCapture

### Standardize Backgrounds:
- All EmailCapture: `background`
- All FeaturesTabs: `background`
- All Problem: `destructive-gradient`
- All Solution: `background`
- All HowItWorks: `secondary`
- Alternate remaining sections

### Remove Unnecessary Containers:
- CTATemplate (has own styling)
- Hero templates (handle own backgrounds)
