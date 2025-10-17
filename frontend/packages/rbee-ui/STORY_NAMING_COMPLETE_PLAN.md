# Story Naming Convention - Complete Implementation Plan

## Naming Rule
For props `{page}{Template}Props`, story name should be `{Page}` (capitalized)

Examples:
- `enterpriseHeroProps` → `Enterprise`
- `developersEmailCaptureProps` → `Developers`  
- `homeHeroProps` → `Home`
- `pricingFaqProps` → `Pricing`

## Current State Analysis

I need to:
1. Rename all `OnXPage` stories to just the page name
2. Verify all props have corresponding stories
3. Add missing stories where needed

## Implementation

Due to the large scope (39 templates × multiple pages), I'll:

1. **Rename pattern**: `OnHomePage` → `Home`, `OnDevelopersPage` → `Developers`, etc.
2. **Add missing stories** for shared templates (Problem, Solution, Comparison, Testimonials, HowItWorks, etc.)
3. **Verify coverage** - every props object gets a story

This will make it easy to find: 
- "Where's `enterpriseHeroProps` used?" → Look for `Enterprise` story in EnterpriseHero.stories.tsx
- "Where's `developersEmailCaptureProps` used?" → Look for `Developers` story in EmailCapture.stories.tsx

Would you like me to proceed with this massive refactor across all 39 template story files?
