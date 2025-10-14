# Storybook Master Index

**Complete guide to all Storybook documentation and planning**

---

## 📋 Planning Documents

### 1. **STORYBOOK_STORIES_PLAN.md** ⭐ PRIMARY PLAN
**Purpose:** Complete execution plan for creating all Storybook stories  
**Use When:** Starting the project, tracking progress, understanding phases  
**Key Sections:**
- Phase-by-phase breakdown (0-7)
- Timeline and estimates
- Success criteria
- Quick start for engineers

### 2. **STORYBOOK_COMPONENT_DISCOVERY.md**
**Purpose:** Complete inventory of components needing stories  
**Use When:** Picking which component to work on, checking what's used  
**Key Sections:**
- Full component list (41 components)
- Priority classification (P0-P3)
- Usage locations
- Verification commands

### 3. **STORYBOOK_DOCUMENTATION_STANDARD.md** ⭐ MANDATORY REFERENCE
**Purpose:** Documentation requirements and standards for all stories  
**Use When:** Writing any story, during code review  
**Key Sections:**
- Component description templates
- ArgTypes configuration
- Story variant requirements
- Mock data standards
- Quality checklist

### 4. **STORYBOOK_QUICK_START.md** ⭐ ENGINEER GUIDE
**Purpose:** Step-by-step guide to create a story in 5 minutes  
**Use When:** First time writing a story, need quick reference  
**Key Sections:**
- 10-step process
- Code templates
- Common patterns
- Troubleshooting

---

## 🎯 How to Use These Documents

### If You're Starting Fresh
1. Read **STORYBOOK_STORIES_PLAN.md** (overview)
2. Check **STORYBOOK_COMPONENT_DISCOVERY.md** (pick component)
3. Follow **STORYBOOK_QUICK_START.md** (write story)
4. Reference **STORYBOOK_DOCUMENTATION_STANDARD.md** (ensure quality)

### If You're Writing a Story
1. **STORYBOOK_QUICK_START.md** - Follow steps 1-10
2. **STORYBOOK_DOCUMENTATION_STANDARD.md** - Check requirements
3. **STORYBOOK_COMPONENT_DISCOVERY.md** - Verify component priority

### If You're Reviewing a Story
1. **STORYBOOK_DOCUMENTATION_STANDARD.md** - Check against standard
2. Section 8: Quality Checklist - Verify all items

### If You're Tracking Progress
1. **STORYBOOK_STORIES_PLAN.md** - Check phase completion
2. **STORYBOOK_COMPONENT_DISCOVERY.md** - Mark components complete

---

## 📊 Project Overview

### Scope
- **Total Components:** 41
- **Atoms:** 2
- **Organisms:** 39
- **Estimated Time:** 35-40 hours
- **Timeline:** 2-3 weeks

### Priorities
- **P0 (Critical):** 4 components - Navigation, Footer, Icons
- **P1 (High):** 5 components - Hero, CTA, Pricing, FAQ, EmailCapture
- **P2 (Medium):** 11 components - Marketing sections
- **P3 (Low):** 21 components - Page-specific variants

### Phases
1. **Phase 0:** Discovery (✅ Complete)
2. **Phase 1:** Foundation setup (⏳ In Progress)
3. **Phase 2:** P0 components
4. **Phase 3:** P1 components
5. **Phase 4:** P2 components
6. **Phase 5:** P3 components
7. **Phase 6:** Quality assurance
8. **Phase 7:** Documentation

---

## 🔧 Technical Setup

### Dark Mode Configuration
- ✅ Updated `.storybook/preview.ts` with theme decorator
- ✅ Theme toggle in toolbar (no manual theme props)
- ✅ Automatic dark class application

### Commands
```bash
# Start Storybook
pnpm storybook

# Build Storybook
pnpm build-storybook

# Verify component usage
grep -r "ComponentName" /home/vince/Projects/llama-orch/frontend/bin/commercial/app/
```

---

## ✅ Success Criteria

### Documentation
- [ ] All 41 components have stories
- [ ] All stories follow documentation standard
- [ ] All props documented in argTypes
- [ ] Realistic mock data used

### Quality
- [ ] All stories render in light mode
- [ ] All stories render in dark mode
- [ ] No console errors
- [ ] Responsive tested where applicable

### Deliverables
- [ ] 41 story files created
- [ ] All phases complete
- [ ] QA passed
- [ ] Documentation updated

---

## 📝 Story Requirements Summary

### Minimum Stories per Component
- **Atoms:** 2 stories (Default + 1 variant)
- **Molecules:** 3 stories (Default + 2 variants)
- **Organisms:** 3 stories (Default + 2 variants)

### Required Documentation
- Overview
- When to Use
- Variants
- Examples
- Accessibility (organisms)

### Required Testing
- Light mode
- Dark mode
- Responsive (if applicable)
- Interactive elements
- No console errors

---

## 🚀 Quick Reference

### File Locations
```
frontend/libs/rbee-ui/
├── .storybook/
│   ├── main.ts
│   └── preview.ts (✅ Updated with theme decorator)
├── src/
│   ├── atoms/
│   │   └── ComponentName/
│   │       ├── ComponentName.tsx
│   │       └── ComponentName.stories.tsx
│   ├── organisms/
│   │   └── ComponentName/
│   │       ├── ComponentName.tsx
│   │       └── ComponentName.stories.tsx
│   └── __mocks__/
│       └── componentData.ts
└── STORYBOOK_*.md (Planning docs)
```

### Story Template Locations
- Atom template: See `STORYBOOK_QUICK_START.md` Step 3
- Organism template: See `STORYBOOK_QUICK_START.md` Step 3
- Full examples: `src/atoms/Button.stories.tsx`, `src/atoms/Badge.stories.tsx`

### Key Standards
- ❌ NO separate dark/light mode stories
- ✅ Use toolbar theme toggle
- ✅ Realistic mock data (no Lorem ipsum)
- ✅ Minimum 2 stories per component
- ✅ Document all props in argTypes

---

## 📚 Additional Resources

### Existing Stories (Examples)
- `src/atoms/Button.stories.tsx` - Complete atom example
- `src/atoms/Badge.stories.tsx` - Atom with variants
- `src/organisms/HeroSection/HeroSection.stories.tsx` - Organism example

### External Documentation
- [Storybook Docs](https://storybook.js.org/docs)
- [Storybook ArgTypes](https://storybook.js.org/docs/react/api/argtypes)
- [Storybook Decorators](https://storybook.js.org/docs/react/writing-stories/decorators)

---

## 🎯 Next Actions

### Immediate (Phase 1)
1. ✅ Update `.storybook/preview.ts` with theme decorator
2. ⏳ Test theme toggle in existing stories
3. ⏳ Create mock data directory structure

### Phase 2 (P0 Components)
1. GitHubIcon story
2. DiscordIcon story
3. Navigation story
4. Footer story

### Phase 3 (P1 Components)
1. HeroSection story
2. EmailCapture story
3. CTASection story
4. PricingSection story
5. FAQSection story

---

## 📞 Support

### Questions?
1. Check **STORYBOOK_QUICK_START.md** for common issues
2. Review existing stories for examples
3. Reference **STORYBOOK_DOCUMENTATION_STANDARD.md** for requirements

### Issues?
1. Check Storybook console for errors
2. Verify component imports
3. Test in isolation
4. Check dark mode configuration

---

## 🎉 Summary

**You have everything you need to start writing stories:**

1. **The Plan** - STORYBOOK_STORIES_PLAN.md
2. **The List** - STORYBOOK_COMPONENT_DISCOVERY.md
3. **The Standard** - STORYBOOK_DOCUMENTATION_STANDARD.md
4. **The Guide** - STORYBOOK_QUICK_START.md

**Pick a component, follow the quick start, and start building world-class documentation! 🚀**
