# EducationPage Fix Guide

**Status**: 40 TypeScript errors requiring structural fixes  
**Recommendation**: Complete rewrite following HomelabPage pattern

## 🔧 Required Fixes

### 1. BeeArchitecture Topology (Line 196)
**Current (WRONG)**:
```tsx
<BeeArchitecture topology="homelab" />
```

**Should be**:
```tsx
<BeeArchitecture 
  topology={{
    mode: 'single-pc',
    hostLabel: 'Student Laptop',
    workers: [
      { id: 'w1', label: 'GPU 1', kind: 'cuda' },
      { id: 'w2', label: 'GPU 2', kind: 'cuda' },
    ]
  }}
/>
```

### 2. PricingTemplate Features (Lines 220-262)
**Current (WRONG)**:
```tsx
features: [
  { text: 'What is distributed AI?', included: true },
  { text: 'Beehive architecture basics', included: true },
  // ...
]
```

**Should be** (simple strings):
```tsx
features: [
  'What is distributed AI?',
  'Beehive architecture basics',
  'Simple orchestration',
  'CLI fundamentals',
  'Basic Rust concepts',
]
```

Also need to fix tier structure:
- `name` → `title`
- `body` → remove (not in PricingTierData)
- Add `ctaText`, `ctaHref` from the `cta` object
- `cta.variant` → `ctaVariant`

### 3. SecurityTemplate (Lines 290-345)
**Current (WRONG)**:
```tsx
securityCards: [
  {
    icon: <Shield className="size-6" />,
    title: 'Process Isolation',
    body: 'Learn how rbee isolates...',  // ← WRONG
    bullets: [...],
  }
]
```

**Should be**:
```tsx
securityCards: [
  {
    icon: <Shield className="size-6" />,
    title: 'Process Isolation',
    subtitle: 'Sandboxed Execution',
    intro: 'Learn how rbee isolates...',  // ← Use intro, not body
    bullets: [...],
    docsHref: '/docs/security/isolation',
  }
]
```

### 4. HowItWorks Steps (Lines 372-415)
**Current (WRONG)**:
```tsx
steps: [
  {
    title: 'Install rbee',  // ← WRONG
    block: { kind: 'terminal', content: '...' }
  }
]
```

**Should be**:
```tsx
steps: [
  {
    label: 'Install rbee',  // ← Use label, not title
    block: { kind: 'terminal', content: '...' }
  }
]
```

### 5. UseCases (Line 443)
**Current (WRONG)**:
```tsx
useCases: [...]  // ← WRONG property name
```

**Should be**:
```tsx
items: [...]  // ← Correct property name
```

### 6. Testimonials (Lines 517-533)
**Current (WRONG)**:
```tsx
testimonials: [
  {
    quote: '...',
    author: 'Emma de Vries',
    role: 'CS Student',
    company: 'TU Delft',  // ← WRONG
  }
]
```

**Should be** (remove company or restructure):
```tsx
testimonials: [
  {
    quote: '...',
    author: 'Emma de Vries',
    role: 'CS Student, TU Delft',  // ← Combine into role
  }
]
```

### 7. Stats (Lines 541-551)
**Current (WRONG)**:
```tsx
stats: [
  {
    icon: <Users className="w-6 h-6" />,
    value: '2,000+',
    label: 'Students',
    body: 'Learning with rbee',  // ← WRONG
  }
]
```

**Should be** (remove body):
```tsx
stats: [
  {
    icon: <Users className="w-6 h-6" />,
    value: '2,000+',
    label: 'Students Learning',  // ← Combine label + body
  }
]
```

### 8. FAQ Items (Lines 571-598, 622)
**Current (WRONG)**:
```tsx
{
  icon: <BookOpen className="size-5" />,
  question: '...',
  answer: '...',
}
```

And property name:
```tsx
faqs: [...]  // ← WRONG
```

**Should be**:
```tsx
faqItems: [  // ← Correct property name
  {
    question: '...',
    answer: '...',
    // Remove icon property
  }
]
```

### 9. CTA Template (Line 669)
**Current (WRONG)**:
```tsx
headline: 'Start Learning Today'  // ← WRONG
```

**Should be**:
```tsx
title: 'Start Learning Today'  // ← Use title, not headline
```

## 📋 Recommended Approach

1. **Reference HomelabPage** - It's 100% type-safe and uses similar templates
2. **Fix section by section** - Don't try to fix everything at once
3. **Test after each section** - Run `pnpm tsc --noEmit` frequently
4. **Use correct template interfaces** - Check the template source files for correct prop names

## 🎯 Priority Order

1. **PricingTemplate** (15 errors) - Most errors, simplest fix
2. **SecurityTemplate** (6 errors) - body → intro + subtitle
3. **HowItWorks** (4 errors) - title → label
4. **FAQ** (5 errors) - faqs → faqItems, remove icons
5. **Stats** (3 errors) - remove body property
6. **Testimonials** (3 errors) - remove/restructure company
7. **UseCases** (1 error) - useCases → items
8. **CTA** (1 error) - headline → title
9. **BeeArchitecture** (1 error) - Fix topology structure

## ✅ Success Criteria

Run `pnpm tsc --noEmit` and verify:
- 0 errors in EducationPage
- Page follows same patterns as HomelabPage
- All templates use correct prop names
