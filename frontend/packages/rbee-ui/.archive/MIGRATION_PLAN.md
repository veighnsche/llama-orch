# Migration Plan: Adopting @rbee/ui

## Phase 1: Foundation ✅
- [x] Create @rbee/ui package structure
- [x] Extract design tokens to shared CSS
- [x] Implement core atoms (Button, Badge)
- [x] Implement core molecules (Card)
- [x] Add to pnpm workspace
- [x] Integrate into user-docs
- [x] Integrate into commercial site

## Phase 2: Component Migration (Next Steps)

### Priority 1: Atoms
- [ ] Extract Input component
- [ ] Extract Label component
- [ ] Extract Checkbox component
- [ ] Extract Radio component
- [ ] Extract Switch component
- [ ] Extract Textarea component

### Priority 2: Molecules
- [ ] Extract Alert component
- [ ] Extract Dialog/Modal component
- [ ] Extract Dropdown component
- [ ] Extract Tabs component
- [ ] Extract Accordion component
- [ ] Extract Tooltip component

### Priority 3: Documentation Components
- [ ] Create Callout component (for Nextra docs)
- [ ] Create CodeBlock component
- [ ] Create Table component
- [ ] Create Heading components with anchors

## Phase 3: Advanced Components

### Organisms (if shared between apps)
- [ ] Navigation patterns
- [ ] Section containers
- [ ] Hero patterns
- [ ] Footer patterns

## Guidelines for Migration

### When to Move to @rbee/ui
1. **Component is used in both apps** - Definite candidate
2. **Component is purely presentational** - Good candidate
3. **Component has no app-specific logic** - Good candidate
4. **Component uses only design tokens** - Good candidate

### When to Keep App-Specific
1. **Business logic embedded** - Keep in app
2. **API/data fetching** - Keep in app
3. **Routing-specific** - Keep in app
4. **One-off custom styling** - Keep in app

### Migration Process
1. **Identify component** for migration
2. **Extract to @rbee/ui** with proper types
3. **Update imports** in both apps
4. **Test visual consistency** in both apps
5. **Remove old component** from app
6. **Update documentation** in @rbee/ui

## Testing Checklist

For each migrated component:
- [ ] Renders correctly in commercial site (light mode)
- [ ] Renders correctly in commercial site (dark mode)
- [ ] Renders correctly in user-docs (light mode)
- [ ] Renders correctly in user-docs (dark mode)
- [ ] All variants work as expected
- [ ] All sizes work as expected
- [ ] Props are properly typed
- [ ] Accessibility features preserved
- [ ] No visual regressions

## Benefits

### Immediate
- ✅ Consistent design tokens across apps
- ✅ Single source of truth for colors/spacing
- ✅ Dark mode works consistently

### Near-term
- Reduced code duplication
- Faster component development
- Easier design updates
- Better type safety

### Long-term
- Scalable design system
- Easier to add new applications
- Better developer experience
- Clearer component ownership

## Non-Goals

- **Not a component library for external use** - This is internal tooling
- **Not replacing Radix UI** - We still use Radix as the foundation for complex components
- **Not a CSS framework** - We still use Tailwind for utility classes
- **Not enforcing strict patterns** - Apps can still have custom components

## Next Actions

1. **Commercial site**: Start using `@rbee/ui/styles` in addition to existing globals.css
2. **User-docs**: Already integrated, test shared components work
3. **New components**: Build new components in @rbee/ui when they're needed by both apps
4. **Gradual migration**: Slowly replace duplicated components over time

## Questions & Decisions

### Q: Should we migrate all commercial site components?
**A:** No. Only migrate when:
- Component is needed by user-docs, OR
- Component is a perfect candidate for sharing (pure atoms/molecules)

### Q: What about commercial-specific components?
**A:** Keep them in the commercial site. Not everything needs to be shared.

### Q: How do we handle app-specific styling?
**A:** Use className prop to override styles on a per-app basis.

### Q: Should we version @rbee/ui?
**A:** Not yet. It's a workspace package. If we extract it later, we'll version it properly.
