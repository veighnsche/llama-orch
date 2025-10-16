# Enterprise Page Refactoring - TemplateContainer Pattern

## Current Status

The Enterprise page templates exist but still have custom header sections that need to be refactored to use the `TemplateContainer` pattern, matching the Providers page implementation.

## Templates That Need Refactoring

### 1. EnterpriseComplianceTemplate
**Current:** Has `eyebrow`, `heading`, `description` props with custom header rendering  
**Needed:** Remove header props, remove `<section>` wrapper, change to `<div>`  
**Container Props Needed:** `kicker`, `title`, `description`, `bgVariant`, etc.

### 2. EnterpriseSecurityTemplate  
**Current:** Has custom header section with kicker/title/description  
**Needed:** Remove header section, keep only security crates grid  
**Container Props Needed:** Standard TemplateContainer props

### 3. EnterpriseHowItWorksTemplate
**Current:** Has custom header section with deployment steps  
**Needed:** Remove header section, keep only steps content  
**Container Props Needed:** Standard TemplateContainer props

### 4. EnterpriseUseCasesTemplate
**Current:** Has custom header section with industry cases  
**Needed:** Remove header section, keep only industry grid  
**Container Props Needed:** Standard TemplateContainer props

### 5. EnterpriseComparisonTemplate
**Current:** Has custom header section with comparison matrix  
**Needed:** Remove header section, keep only matrix/cards  
**Container Props Needed:** Standard TemplateContainer props

### 6. EnterpriseFeaturesTemplate
**Current:** Has custom header section with feature cards  
**Needed:** Remove header section, keep only feature grid  
**Container Props Needed:** Standard TemplateContainer props

### 7. EnterpriseTestimonialsTemplate
**Current:** Has custom header section with testimonials rail  
**Needed:** Remove header section, keep only TestimonialsRail  
**Container Props Needed:** Standard TemplateContainer props

## Templates That Are OK (No Changes Needed)

### EnterpriseHeroTemplate
- Self-contained hero section (like ProvidersHeroTemplate)
- No TemplateContainer needed

### EnterpriseSolutionTemplate
- Uses SolutionSection organism (needs TemplateContainer wrapper in page)
- Similar to ProvidersSolution pattern

### EnterpriseCTATemplate
- Self-contained CTA section (like ProvidersCTATemplate)
- No TemplateContainer needed

## Required Changes

### For Each Template:

1. **Remove from Props:**
   - `eyebrow` / `kicker`
   - `heading` / `title`
   - `description` / `subtitle`
   - Any other header-related props

2. **Change Template Structure:**
   ```tsx
   // Before:
   <section className="...">
     <div className="header">
       <p>{eyebrow}</p>
       <h2>{heading}</h2>
       <p>{description}</p>
     </div>
     {/* content */}
   </section>

   // After:
   <div>
     {/* content only */}
   </div>
   ```

3. **Create Container Props:**
   ```tsx
   export const enterpriseComplianceContainerProps: Omit<TemplateContainerProps, 'children'> = {
     kicker: 'Security & Certifications',
     title: 'Compliance by Design',
     description: 'Built from the ground up to meet GDPR, SOC2, and ISO 27001...',
     bgVariant: 'default',
     paddingY: '2xl',
     maxWidth: '7xl',
     align: 'center',
   }
   ```

4. **Update Page Usage:**
   ```tsx
   // Before:
   <EnterpriseComplianceTemplate {...enterpriseComplianceProps} />

   // After:
   <TemplateContainer {...enterpriseComplianceContainerProps}>
     <EnterpriseComplianceTemplate {...enterpriseComplianceProps} />
   </TemplateContainer>
   ```

## Files to Modify

### Templates (7 files):
- `/src/templates/EnterpriseComplianceTemplate/EnterpriseComplianceTemplate.tsx`
- `/src/templates/EnterpriseSecurityTemplate/EnterpriseSecurityTemplate.tsx`
- `/src/templates/EnterpriseHowItWorksTemplate/EnterpriseHowItWorksTemplate.tsx`
- `/src/templates/EnterpriseUseCasesTemplate/EnterpriseUseCasesTemplate.tsx`
- `/src/templates/EnterpriseComparisonTemplate/EnterpriseComparisonTemplate.tsx`
- `/src/templates/EnterpriseFeaturesTemplate/EnterpriseFeaturesTemplate.tsx`
- `/src/templates/EnterpriseTestimonialsTemplate/EnterpriseTestimonialsTemplate.tsx`

### Page Files:
- `/src/pages/EnterprisePage/EnterprisePage.tsx` - Add TemplateContainer wrappers
- `/src/pages/EnterprisePage/EnterprisePageProps.tsx` - Add container props for each template
- `/src/pages/EnterprisePage/index.ts` - Export new container props

## Benefits

1. **Consistency** - All pages use the same TemplateContainer pattern
2. **Maintainability** - Single source of truth for section headers
3. **Reusability** - Templates are pure presentation components
4. **i18n Ready** - All content in props, ready for translation
5. **CMS Ready** - Easy to integrate with content management systems

## Pattern to Follow

Use the Providers page as the reference implementation:
- `ProvidersUseCasesTemplate` - Pure content template
- `providersUseCasesContainerProps` - Header configuration
- `ProvidersPage.tsx` - Wraps template with TemplateContainer

## Next Steps

1. Refactor each Enterprise template (remove headers)
2. Create container props in EnterprisePageProps.tsx
3. Update EnterprisePage.tsx to use TemplateContainer wrappers
4. Update index.ts to export container props
5. Test that all sections render correctly
6. Update Storybook stories to import props from page file

## Estimated Scope

- **Templates to refactor:** 7
- **Container props to create:** 7
- **Page file updates:** 1
- **Props file updates:** 1
- **Index file updates:** 1
- **Total files:** ~17 files

This matches the scope of the Providers page refactoring that was just completed.
