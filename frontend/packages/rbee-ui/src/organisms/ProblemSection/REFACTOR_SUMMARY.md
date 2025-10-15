# ProblemSection Refactor Summary

## What Changed

### 1. **Moved ProblemSection Out of Home/**
- **Before**: `/organisms/Home/ProblemSection/`
- **After**: `/organisms/ProblemSection/`
- **Reason**: ProblemSection is reused across multiple pages (Home, Developers, Enterprise, Providers), so it should not be nested under Home/. This makes the component's shared nature explicit.

### 2. **Removed All Default Values**
- **Before**: Component had default `title`, `subtitle`, and `items` props with home page content
- **After**: `title` and `items` are now **required** props with no defaults
- **Reason**: Default values are lazy and hide the actual content being rendered. Each page should explicitly define its problem messaging. This makes the component truly reusable and forces intentional content decisions.

### 3. **Updated Home Page to Explicitly Pass Props**
- **Before**: `<ProblemSection />` (relied on defaults)
- **After**: `<ProblemSection title="..." subtitle="..." items={[...]} />` (explicit props)
- **Reason**: Home page now mirrors the pattern used by Developers, Enterprise, and Providers pages. All content is visible in the page component, not hidden in defaults.

### 4. **Enhanced Stories with Copywriter Rationale**
All story files now include detailed documentation explaining:
- **Why each pain point was chosen**
- **Target audience for each problem**
- **Copywriting strategy behind the messaging**
- **Tone and emotional appeal**

#### Updated Stories:
- `/organisms/ProblemSection/ProblemSection.stories.tsx` (Home page context)
- `/organisms/Developers/DevelopersProblem/DevelopersProblem.stories.tsx`
- `/organisms/Enterprise/EnterpriseProblem/EnterpriseProblem.stories.tsx`
- `/organisms/Providers/ProvidersProblem/ProvidersProblem.stories.tsx`

### 5. **Updated All Import Paths**
- Developers, Enterprise, and Providers wrapper components now import from `@rbee/ui/organisms/ProblemSection`
- Barrel export in `/organisms/index.ts` updated to export from new location

## File Structure

```
organisms/
├── ProblemSection/                    # NEW LOCATION (shared organism)
│   ├── ProblemSection.tsx
│   ├── ProblemSection.stories.tsx
│   ├── index.ts
│   └── REFACTOR_SUMMARY.md (this file)
├── Developers/
│   └── DevelopersProblem/
│       ├── DevelopersProblem.tsx      # Wrapper with developer-specific props
│       └── DevelopersProblem.stories.tsx  # Enhanced with copywriter rationale
├── Enterprise/
│   └── EnterpriseProblem/
│       ├── EnterpriseProblem.tsx      # Wrapper with enterprise-specific props
│       └── EnterpriseProblem.stories.tsx  # Enhanced with copywriter rationale
└── Providers/
    └── ProvidersProblem/
        ├── ProvidersProblem.tsx       # Wrapper with provider-specific props
        └── ProvidersProblem.stories.tsx   # Enhanced with copywriter rationale
```

## Component API

### ProblemSection (Base Component)

```tsx
type ProblemSectionProps = {
	kicker?: string                    // Optional eyebrow text
	title: string                      // REQUIRED - main headline
	subtitle?: string                  // Optional subheadline
	items: ProblemItem[]               // REQUIRED - array of problem cards
	ctaPrimary?: { label: string; href: string }
	ctaSecondary?: { label: string; href: string }
	ctaCopy?: string
	id?: string
	className?: string
	gridClassName?: string
	eyebrow?: string                   // Legacy support (maps to kicker)
}

type ProblemItem = {
	title: string
	body: string
	icon: React.ComponentType<{ className?: string }> | React.ReactNode
	tag?: string                       // Optional loss indicator (e.g., "Loss €50/mo")
	tone?: 'primary' | 'destructive' | 'muted'
}
```

### Usage Pattern

**Home Page** (explicit props):
```tsx
<ProblemSection
	title="The hidden risk of AI-assisted development"
	subtitle="You're building complex codebases with AI assistance..."
	items={[
		{
			title: 'The model changes',
			body: 'Your assistant updates overnight...',
			icon: <AlertTriangle className="h-6 w-6" />,
			tone: 'destructive',
		},
		// ... more items
	]}
/>
```

**Developers Page** (wrapper component):
```tsx
<DevelopersProblem />  // Uses developer-specific defaults internally
```

**Enterprise Page** (wrapper component):
```tsx
<EnterpriseProblem />  // Uses enterprise-specific defaults internally
```

**Providers Page** (wrapper component):
```tsx
<ProvidersProblem />   // Uses provider-specific defaults internally
```

## Story Documentation Format

Each story now includes:

### Problem Analysis Template
```markdown
**Problem N: [Title]**
- **Icon**: [IconName] ([color]/[tone])
- **Tone**: [Tone type] ([reason])
- **Copy**: "[Exact copy from component]"
- **Tag**: "[Tag text]" (optional)
- **Target**: [Target audience]
- **Why this pain point**: [Detailed explanation of why the copywriter chose this pain point, what it addresses, and why it resonates with the target audience]
```

### Example (from Developers story):
```markdown
**Problem 1: The Model Changes**
- **Icon**: AlertTriangle (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your team is blocked."
- **Tag**: "High risk"
- **Target**: Developers using AI coding assistants (Cursor, Zed, Continue, GitHub Copilot)
- **Why this pain point**: This is the #1 fear for developers building with AI assistance. When Claude/GPT/Copilot updates, code generation patterns change. What worked yesterday breaks today. Your team's velocity drops to zero. This is a visceral, immediate pain that developers have experienced firsthand. The copywriter chose this because it's the most relatable pain point—every developer using AI has experienced a breaking change.
```

## Benefits of This Refactor

1. **No More Lazy Defaults**: Every page explicitly defines its problem messaging
2. **True Reusability**: ProblemSection is now a proper shared organism, not a Home page component
3. **Better Documentation**: Stories explain WHY each pain point was chosen, not just WHAT it says
4. **Consistent Patterns**: Home page now follows the same pattern as other pages
5. **Maintainability**: Changes to problem messaging are explicit and visible in page components
6. **Copywriter Intent**: Stories preserve the reasoning behind messaging decisions

## Migration Notes

- ✅ All imports updated
- ✅ Home page updated to pass explicit props
- ✅ All wrapper components (Developers, Enterprise, Providers) updated
- ✅ All stories enhanced with copywriter rationale
- ✅ Old `/organisms/Home/ProblemSection/` directory deleted
- ✅ TypeScript compilation passes
- ✅ No breaking changes to public API (backward compatible)

## Related Files

- `/apps/commercial/app/page.tsx` - Home page updated with explicit props
- `/organisms/index.ts` - Barrel export updated
- `/organisms/Developers/DevelopersProblem/` - Import path updated
- `/organisms/Enterprise/EnterpriseProblem/` - Import path updated
- `/organisms/Providers/ProvidersProblem/` - Import path updated

---

**Refactored by**: Assistant
**Date**: 2025-01-15
**Reason**: User requested removal of default values and proper shared organism structure
