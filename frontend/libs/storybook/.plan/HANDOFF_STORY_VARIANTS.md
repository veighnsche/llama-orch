# HANDOFF: Story Variants Implementation

**From:** TEAM-FE-008  
**To:** Next Team  
**Date:** 2025-10-11  
**Priority:** Medium  
**Estimated Time:** 4-6 hours

---

## 📋 Task Overview

Enhance all `.story.vue` files for Enterprise and Features page components with proper variants demonstrating different prop configurations.

**Current State:** All `.story.vue` files exist but only have "Default" variant  
**Target State:** Each story should have 2-4 variants showing different use cases

---

## 🎯 Components Requiring Story Variants

### Enterprise Page Components (11)

1. **EnterpriseHero** (`organisms/EnterpriseHero/EnterpriseHero.story.vue`)
   - Default variant ✅ (exists)
   - Without Audit Log variant (set `showAuditLog: false`)
   - Custom Stats variant (different stat values)
   - Custom CTAs variant (different button text)

2. **EnterpriseProblem** (`organisms/EnterpriseProblem/EnterpriseProblem.story.vue`)
   - Default variant ✅ (exists)
   - Custom Problems variant (different problem set)
   - Custom Quote variant (different quote text)

3. **EnterpriseSolution** (`organisms/EnterpriseSolution/EnterpriseSolution.story.vue`)
   - Default variant ✅ (exists)
   - Custom Security Layers variant
   - Custom Benefits variant

4. **EnterpriseHowItWorks** (`organisms/EnterpriseHowItWorks/EnterpriseHowItWorks.story.vue`)
   - Default variant ✅ (exists)
   - Custom Steps variant (3 steps instead of 4)
   - Custom Timeline variant

5. **EnterpriseFeatures** (`organisms/EnterpriseFeatures/EnterpriseFeatures.story.vue`)
   - Default variant ✅ (exists)
   - 2 Features variant (grid with 2 items)
   - Custom Features variant

6. **EnterpriseSecurity** (`organisms/EnterpriseSecurity/EnterpriseSecurity.story.vue`)
   - Default variant ✅ (exists)
   - Custom Crates variant (different security crates)
   - Custom Guarantees variant

7. **EnterpriseCompliance** (`organisms/EnterpriseCompliance/EnterpriseCompliance.story.vue`)
   - Default variant ✅ (exists)
   - Single Standard variant (only GDPR)
   - Custom Standards variant

8. **EnterpriseComparison** (`organisms/EnterpriseComparison/EnterpriseComparison.story.vue`)
   - Default variant ✅ (exists)
   - Fewer Features variant (8 features instead of 12)
   - Custom Competitors variant

9. **EnterpriseUseCases** (`organisms/EnterpriseUseCases/EnterpriseUseCases.story.vue`)
   - Default variant ✅ (exists)
   - 2 Use Cases variant
   - Custom Use Cases variant

10. **EnterpriseTestimonials** (`organisms/EnterpriseTestimonials/EnterpriseTestimonials.story.vue`)
    - Default variant ✅ (exists)
    - Custom Metrics variant
    - Custom Testimonials variant

11. **EnterpriseCTA** (`organisms/EnterpriseCTA/EnterpriseCTA.story.vue`)
    - Default variant ✅ (exists)
    - 2 Options variant
    - Custom Options variant

### Features Page Components (9)

12. **FeaturesHero** (`organisms/FeaturesHero/FeaturesHero.story.vue`)
    - Default variant ✅ (exists)
    - Custom Title variant
    - Custom Highlight variant

13. **CoreFeaturesTabs** (`organisms/CoreFeaturesTabs/CoreFeaturesTabs.story.vue`)
    - Default variant ✅ (exists)
    - No variants needed (complex component, default is sufficient)

14. **MultiBackendGPU** (`organisms/MultiBackendGPU/MultiBackendGPU.story.vue`)
    - Default variant ✅ (exists)
    - No variants needed (informational component)

15. **CrossNodeOrchestration** (`organisms/CrossNodeOrchestration/CrossNodeOrchestration.story.vue`)
    - Default variant ✅ (exists)
    - No variants needed (informational component)

16. **IntelligentModelManagement** (`organisms/IntelligentModelManagement/IntelligentModelManagement.story.vue`)
    - Default variant ✅ (exists)
    - No variants needed (informational component)

17. **RealTimeProgress** (`organisms/RealTimeProgress/RealTimeProgress.story.vue`)
    - Default variant ✅ (exists)
    - No variants needed (informational component)

18. **ErrorHandling** (`organisms/ErrorHandling/ErrorHandling.story.vue`)
    - Default variant ✅ (exists)
    - No variants needed (informational component)

19. **SecurityIsolation** (`organisms/SecurityIsolation/SecurityIsolation.story.vue`)
    - Default variant ✅ (exists)
    - No variants needed (informational component)

20. **AdditionalFeaturesGrid** (`organisms/AdditionalFeaturesGrid/AdditionalFeaturesGrid.story.vue`)
    - Default variant ✅ (exists)
    - 4 Features variant
    - Custom Features variant

---

## 📖 Story Pattern Reference

### Basic Story Structure (Current - Minimal)
```vue
<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<script setup lang="ts">
import ComponentName from './ComponentName.vue'
</script>

<template>
  <Story title="organisms/ComponentName">
    <Variant title="Default">
      <ComponentName />
    </Variant>
  </Story>
</template>
```

### Enhanced Story Structure (Target - With Variants)
```vue
<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-[YOUR-TEAM]: Added story variants -->
<script setup lang="ts">
import ComponentName from './ComponentName.vue'
</script>

<template>
  <Story title="organisms/ComponentName">
    <Variant title="Default">
      <ComponentName />
    </Variant>

    <Variant title="Custom Props">
      <ComponentName
        title="Custom Title"
        description="Custom description text"
        :some-prop="false"
      />
    </Variant>

    <Variant title="Minimal">
      <ComponentName
        title="Minimal Version"
        :show-optional-section="false"
      />
    </Variant>
  </Story>
</template>
```

---

## 🎨 Example: EnterpriseHero Story Enhancement

**Current File:** `organisms/EnterpriseHero/EnterpriseHero.story.vue`

```vue
<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<script setup lang="ts">
import EnterpriseHero from './EnterpriseHero.vue'
</script>

<template>
  <Story title="organisms/EnterpriseHero">
    <Variant title="Default">
      <EnterpriseHero />
    </Variant>
  </Story>
</template>
```

**Enhanced Version:**

```vue
<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-[YOUR-TEAM]: Added story variants -->
<script setup lang="ts">
import EnterpriseHero from './EnterpriseHero.vue'
</script>

<template>
  <Story title="organisms/EnterpriseHero">
    <Variant title="Default">
      <EnterpriseHero />
    </Variant>

    <Variant title="Without Audit Log">
      <EnterpriseHero :show-audit-log="false" />
    </Variant>

    <Variant title="Custom Stats">
      <EnterpriseHero
        stat1-value="99.99%"
        stat1-label="Uptime SLA"
        stat2-value="10 Years"
        stat2-label="Data Retention"
        stat3-value="50+"
        stat3-label="Event Types"
      />
    </Variant>

    <Variant title="Custom CTAs">
      <EnterpriseHero
        primary-cta="Get Started"
        secondary-cta="Learn More"
      />
    </Variant>
  </Story>
</template>
```

---

## 🎨 Example: EnterpriseProblem Story Enhancement

```vue
<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-[YOUR-TEAM]: Added story variants -->
<script setup lang="ts">
import { Globe, FileX } from 'lucide-vue-next'
import EnterpriseProblem from './EnterpriseProblem.vue'
</script>

<template>
  <Story title="organisms/EnterpriseProblem">
    <Variant title="Default">
      <EnterpriseProblem />
    </Variant>

    <Variant title="Custom Problems (2 items)">
      <EnterpriseProblem
        :problems="[
          {
            icon: Globe,
            title: 'Data Residency Issues',
            description: 'Your data crosses international borders without your control.'
          },
          {
            icon: FileX,
            title: 'No Audit Trail',
            description: 'Cannot prove compliance to auditors.'
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Quote">
      <EnterpriseProblem
        quote-text="We need complete control over our AI infrastructure."
        quote-author="— CTO, Fortune 500 Company"
      />
    </Variant>
  </Story>
</template>
```

---

## 📝 Implementation Checklist

For each component story file:

### Step 1: Review Component Props
- [ ] Open the component `.vue` file
- [ ] Review the `interface Props` section
- [ ] Identify which props have defaults
- [ ] Identify which props are optional

### Step 2: Plan Variants
- [ ] Decide on 2-4 meaningful variants
- [ ] Focus on common use cases
- [ ] Show prop flexibility
- [ ] Demonstrate edge cases (if applicable)

### Step 3: Implement Variants
- [ ] Add your team signature comment
- [ ] Add `<Variant>` blocks
- [ ] Test in Histoire (`pnpm story:dev`)
- [ ] Verify visual appearance

### Step 4: Document
- [ ] Add descriptive variant titles
- [ ] Ensure props are correctly typed
- [ ] Test all variants render correctly

---

## 🎯 Priority Order

### High Priority (User-facing, complex props)
1. EnterpriseHero
2. EnterpriseProblem
3. EnterpriseSolution
4. EnterpriseFeatures
5. FeaturesHero
6. AdditionalFeaturesGrid

### Medium Priority (Moderate complexity)
7. EnterpriseHowItWorks
8. EnterpriseSecurity
9. EnterpriseCompliance
10. EnterpriseUseCases
11. EnterpriseTestimonials
12. EnterpriseCTA

### Low Priority (Informational, minimal props)
13-20. All Features page components (most are informational)

---

## 🔍 Testing Instructions

### Run Histoire
```bash
cd frontend/libs/storybook
pnpm story:dev
```

### Verify Each Story
1. Navigate to `organisms/ComponentName`
2. Check all variants appear in sidebar
3. Click each variant
4. Verify component renders correctly
5. Check responsive behavior
6. Verify no console errors

---

## 📚 Reference Materials

### Component Props Documentation
All components have TypeScript interfaces at the top of their `.vue` files:
```typescript
interface Props {
  title?: string
  description?: string
  showSomething?: boolean
  items?: SomeType[]
}
```

### Design Tokens
All components use design tokens. When creating variants, maintain token usage:
- `bg-primary`, `text-primary`
- `bg-background`, `text-foreground`
- `bg-card`, `bg-secondary`
- `text-muted-foreground`
- `border-border`

### Lucide Icons
If variants need custom icons, import from `lucide-vue-next`:
```typescript
import { Shield, Lock, CheckCircle2 } from 'lucide-vue-next'
```

---

## ✅ Acceptance Criteria

- [ ] All 20 component story files have been enhanced
- [ ] Each story has 2-4 variants (or justified as "no variants needed")
- [ ] All variants render correctly in Histoire
- [ ] No TypeScript errors
- [ ] No console errors
- [ ] Responsive behavior verified
- [ ] Team signature added to modified files
- [ ] All variants use design tokens (no hardcoded colors)

---

## 📊 Estimated Effort

| Task | Time | Components |
|------|------|------------|
| High Priority Stories | 2-3 hours | 6 components |
| Medium Priority Stories | 1-2 hours | 6 components |
| Low Priority Stories | 0.5-1 hour | 8 components |
| Testing & QA | 0.5-1 hour | All |
| **Total** | **4-7 hours** | **20 components** |

---

## 🚀 Getting Started

1. **Read this document completely**
2. **Review DevelopersHero.story.vue** (good example with variants)
3. **Start with EnterpriseHero** (high priority, clear props)
4. **Test frequently** in Histoire
5. **Follow the pattern** for remaining components

---

## 💡 Tips

- **Keep it simple** - Don't over-complicate variants
- **Be practical** - Show real use cases, not contrived examples
- **Test as you go** - Run Histoire frequently
- **Copy patterns** - Use DevelopersHero as reference
- **Document decisions** - If a component doesn't need variants, note why
- **Ask questions** - If props are unclear, check the component implementation

---

## 📞 Questions?

If you encounter issues:
1. Check the component's TypeScript interface for available props
2. Review DevelopersHero.story.vue for pattern reference
3. Test in Histoire to see if variants render
4. Check console for TypeScript errors

---

**Status:** 📋 Ready for Next Team  
**Blockers:** None  
**Dependencies:** All components implemented ✅  
**Signature:** TEAM-FE-008
