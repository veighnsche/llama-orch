# Story Implementation - Complete Plan

## Rule: NO PROPS DUPLICATION
✅ All stories IMPORT props from page files
✅ NO props defined in story files
✅ Single source of truth: PageProps.tsx files

## Implementation Strategy

For each props object, add/rename story in corresponding template:

### Format:
```tsx
import { {propsName} } from '@rbee/ui/pages/{Page}'

/**
 * On{Page}{Template} - {propsName}
 * @tags page, template-type, key-concepts
 */
export const On{Page}{Template}: Story = {
  render: (args) => (
    <TemplateContainer {...containerProps}>
      <Template {...args} />
    </TemplateContainer>
  ),
  args: {propsName},
}
```

## Execution Plan

I'll process templates in this order:
1. **Shared templates** (used on multiple pages) - 6 templates
2. **Page-specific templates** - 33 templates

### Shared Templates (Priority 1):
- EmailCapture (6 pages) ✅ Already done
- FeaturesTabs (4 pages) ✅ Already done  
- ProblemTemplate (4 pages) ✅ Just completed
- SolutionTemplate (4 pages)
- HowItWorks (3 pages)
- TestimonialsTemplate (4 pages)
- ComparisonTemplate (2 pages)
- PricingTemplate (2 pages)
- UseCasesTemplate (2 pages)
- CTATemplate (2 pages)
- FAQTemplate (2 pages)

### Page-Specific Templates (Priority 2):
All hero, specialized feature templates, etc.

## Total Work:
- ~60 story renames/additions
- All importing from existing props
- Zero duplication

Ready to execute?
