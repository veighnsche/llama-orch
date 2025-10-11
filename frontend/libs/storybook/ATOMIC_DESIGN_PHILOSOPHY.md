# Atomic Design Philosophy for orchyra-storybook

**Created by:** TEAM-FE-000  
**Date:** 2025-10-11  
**Status:** Design System Guidelines

---

## 🧬 What is Atomic Design?

Atomic Design is a methodology for creating design systems by breaking interfaces down into their fundamental building blocks, then combining them to form increasingly complex components.

**Created by Brad Frost**, this methodology uses chemistry as a metaphor:

```
Atoms → Molecules → Organisms → Templates → Pages
```

---

## 📐 The Five Levels

### 1️⃣ Atoms (Foundational Elements)

**Definition:** The basic building blocks of matter. In interfaces, these are the smallest functional units.

**Examples:**
- Buttons
- Input fields
- Labels
- Icons
- Typography elements (headings, paragraphs)
- Color swatches
- Spacing units

**Characteristics:**
- ✅ Cannot be broken down further without losing meaning
- ✅ Highly reusable
- ✅ Abstract and context-free
- ✅ Defined by design tokens

**In orchyra-storybook:**
```vue
<!-- Button.vue - An Atom -->
<template>
  <button :class="buttonClass">
    <slot />
  </button>
</template>
```

**Story Example:**
```typescript
// Button.story.ts
export const Primary = () => ({
  components: { Button },
  template: '<Button variant="primary">Click Me</Button>',
})
```

---

### 2️⃣ Molecules (Simple Combinations)

**Definition:** Groups of atoms bonded together to form relatively simple, functional units.

**Examples:**
- Search form (input + button)
- Form field (label + input + error message)
- Card header (avatar + name + timestamp)
- Navigation item (icon + label)

**Characteristics:**
- ✅ Composed of 2-3 atoms
- ✅ Single, focused purpose
- ✅ Still relatively simple
- ✅ Reusable across contexts

**In orchyra-storybook:**
```vue
<!-- FormField.vue - A Molecule -->
<template>
  <div class="form-field">
    <Label :for="id">{{ label }}</Label>
    <Input :id="id" v-model="value" />
    <ErrorMessage v-if="error">{{ error }}</ErrorMessage>
  </div>
</template>
```

**Story Example:**
```typescript
// FormField.story.ts
export const WithError = () => ({
  components: { FormField },
  template: '<FormField label="Email" error="Invalid email" />',
})
```

---

### 3️⃣ Organisms (Complex Components)

**Definition:** Relatively complex UI components composed of groups of molecules and/or atoms.

**Examples:**
- Navigation bar (logo + nav items + search + user menu)
- Product card (image + title + description + price + button)
- Comment thread (avatar + name + timestamp + content + actions)
- Pricing table (header + features list + CTA)

**Characteristics:**
- ✅ Composed of multiple molecules/atoms
- ✅ Forms a distinct section of an interface
- ✅ Can be reused across templates
- ✅ Has clear boundaries and purpose

**In orchyra-storybook:**
```vue
<!-- NavBar.vue - An Organism -->
<template>
  <nav class="navbar">
    <Brand :logo="logo" />
    <NavItems :items="items" />
    <SearchBar />
    <UserMenu :user="user" />
  </nav>
</template>
```

**Story Example:**
```typescript
// NavBar.story.ts
export const LoggedIn = () => ({
  components: { NavBar },
  template: '<NavBar :user="{ name: \'John\' }" />',
})
```

---

### 4️⃣ Templates (Page Layouts)

**Definition:** Page-level objects that place components into a layout and articulate the design's underlying content structure.

**Examples:**
- Homepage layout
- Dashboard layout
- Article layout
- Checkout flow layout

**Characteristics:**
- ✅ Focus on content structure, not content itself
- ✅ Show component relationships
- ✅ Define responsive behavior
- ✅ Use placeholder content

**In orchyra-storybook:**
```vue
<!-- HomeTemplate.vue - A Template -->
<template>
  <div class="home-template">
    <NavBar />
    <Hero :title="title" :subtitle="subtitle" />
    <Features :items="features" />
    <CTA />
    <Footer />
  </div>
</template>
```

**Story Example:**
```typescript
// HomeTemplate.story.ts
export const Default = () => ({
  components: { HomeTemplate },
  template: '<HomeTemplate title="Welcome" />',
})
```

---

### 5️⃣ Pages (Real Content)

**Definition:** Specific instances of templates with real representative content.

**Examples:**
- Homepage with actual copy
- Product detail page with real product
- User profile with actual user data

**Characteristics:**
- ✅ Templates filled with real content
- ✅ What users actually see
- ✅ Used for testing and stakeholder review
- ✅ Highest level of fidelity

**Note:** Pages are typically **not** in the storybook library. They live in the application (`commercial-frontend-v2/src/views/`).

---

## 🎯 Why Atomic Design?

### Benefits

1. **Consistency**
   - Reusing atoms/molecules ensures visual consistency
   - Design tokens propagate changes everywhere

2. **Efficiency**
   - Build once, use everywhere
   - Changes to atoms cascade to all molecules/organisms

3. **Scalability**
   - Easy to add new pages using existing components
   - Component library grows organically

4. **Collaboration**
   - Designers and developers speak the same language
   - Clear component hierarchy

5. **Testing**
   - Test atoms in isolation
   - Compose tested atoms into molecules
   - Confidence in complex organisms

6. **Documentation**
   - Storybook shows all components at all levels
   - Living style guide

---

## 📋 Guidelines for orchyra-storybook

### 1. Start with Design Tokens (Subatomic)

Before atoms, define your design tokens:

```css
/* tokens.css */
:root {
  /* Colors */
  --color-primary: #0066cc;
  --color-secondary: #6c757d;
  --color-success: #28a745;
  --color-danger: #dc3545;
  
  /* Typography */
  --font-family-base: 'Inter', sans-serif;
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  
  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;
  
  /* Borders */
  --border-radius-sm: 0.25rem;
  --border-radius-md: 0.5rem;
  --border-radius-lg: 1rem;
}
```

**Rule:** All components must use design tokens, never hardcoded values.

---

### 2. Build Atoms First

**Atoms should be:**
- ✅ Single-purpose
- ✅ Highly configurable via props
- ✅ Style-agnostic (use design tokens)
- ✅ Accessible (ARIA, keyboard nav)

**Example Atoms:**
```
stories/
├── Button/
│   ├── Button.vue
│   └── Button.story.ts
├── Input/
│   ├── Input.vue
│   └── Input.story.ts
├── Label/
│   ├── Label.vue
│   └── Label.story.ts
└── Icon/
    ├── Icon.vue
    └── Icon.story.ts
```

**Button.vue (Atom):**
```vue
<script setup lang="ts">
interface Props {
  variant?: 'primary' | 'secondary' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  disabled: false,
})
</script>

<template>
  <button
    :class="[
      'btn',
      `btn--${variant}`,
      `btn--${size}`,
      { 'btn--disabled': disabled }
    ]"
    :disabled="disabled"
  >
    <slot />
  </button>
</template>

<style scoped>
.btn {
  font-family: var(--font-family-base);
  border: none;
  border-radius: var(--border-radius-md);
  cursor: pointer;
  transition: all 0.2s;
}

.btn--primary {
  background: var(--color-primary);
  color: white;
}

.btn--sm {
  padding: var(--spacing-xs) var(--spacing-sm);
  font-size: var(--font-size-sm);
}

.btn--md {
  padding: var(--spacing-sm) var(--spacing-md);
  font-size: var(--font-size-base);
}
</style>
```

---

### 3. Compose Molecules

**Molecules should:**
- ✅ Combine 2-3 atoms
- ✅ Serve a single purpose
- ✅ Be context-independent
- ✅ Handle their own state (if simple)

**Example Molecules:**
```
stories/
├── FormField/
│   ├── FormField.vue
│   └── FormField.story.ts
├── SearchBar/
│   ├── SearchBar.vue
│   └── SearchBar.story.ts
└── NavItem/
    ├── NavItem.vue
    └── NavItem.story.ts
```

**FormField.vue (Molecule):**
```vue
<script setup lang="ts">
import { Label, Input, ErrorMessage } from '../atoms'

interface Props {
  label: string
  modelValue: string
  error?: string
  required?: boolean
}

const props = defineProps<Props>()
const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()
</script>

<template>
  <div class="form-field">
    <Label :required="required">{{ label }}</Label>
    <Input
      :model-value="modelValue"
      :error="!!error"
      @update:model-value="emit('update:modelValue', $event)"
    />
    <ErrorMessage v-if="error">{{ error }}</ErrorMessage>
  </div>
</template>

<style scoped>
.form-field {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}
</style>
```

---

### 4. Build Organisms

**Organisms should:**
- ✅ Combine multiple molecules/atoms
- ✅ Form a distinct UI section
- ✅ Be reusable across pages
- ✅ Handle complex interactions

**Example Organisms:**
```
stories/
├── NavBar/
│   ├── NavBar.vue
│   └── NavBar.story.ts
├── ProductCard/
│   ├── ProductCard.vue
│   └── ProductCard.story.ts
└── PricingTable/
    ├── PricingTable.vue
    └── PricingTable.story.ts
```

**NavBar.vue (Organism):**
```vue
<script setup lang="ts">
import { Brand, NavItem, Button } from '../atoms'
import { SearchBar } from '../molecules'

interface NavLink {
  label: string
  href: string
}

interface Props {
  logo: string
  links: NavLink[]
  showSearch?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showSearch: true,
})
</script>

<template>
  <nav class="navbar">
    <Brand :logo="logo" />
    <div class="navbar__links">
      <NavItem
        v-for="link in links"
        :key="link.href"
        :href="link.href"
      >
        {{ link.label }}
      </NavItem>
    </div>
    <SearchBar v-if="showSearch" />
    <Button variant="primary">Get Started</Button>
  </nav>
</template>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  gap: var(--spacing-lg);
  padding: var(--spacing-md);
  background: var(--color-background);
  border-bottom: 1px solid var(--color-border);
}

.navbar__links {
  display: flex;
  gap: var(--spacing-md);
  flex: 1;
}
</style>
```

---

### 5. Create Templates (Optional in Storybook)

Templates are usually in the application, but you can create them in storybook for documentation:

```vue
<!-- HomeTemplate.vue -->
<script setup lang="ts">
import { NavBar, Hero, Features, Footer } from '../organisms'
</script>

<template>
  <div class="home-template">
    <NavBar />
    <main>
      <Hero />
      <Features />
    </main>
    <Footer />
  </div>
</template>
```

---

## 📝 Story Writing Guidelines

### 1. One Story File Per Component

```
Button/
├── Button.vue          # Component
└── Button.story.ts     # Stories
```

### 2. Show All Variants

```typescript
// Button.story.ts
export const Primary = () => ({
  components: { Button },
  template: '<Button variant="primary">Primary</Button>',
})

export const Secondary = () => ({
  components: { Button },
  template: '<Button variant="secondary">Secondary</Button>',
})

export const Danger = () => ({
  components: { Button },
  template: '<Button variant="danger">Danger</Button>',
})

export const Disabled = () => ({
  components: { Button },
  template: '<Button disabled>Disabled</Button>',
})
```

### 3. Show All States

```typescript
// Input.story.ts
export const Default = () => ({
  components: { Input },
  template: '<Input placeholder="Enter text" />',
})

export const WithValue = () => ({
  components: { Input },
  template: '<Input model-value="Hello World" />',
})

export const WithError = () => ({
  components: { Input },
  template: '<Input error="This field is required" />',
})

export const Disabled = () => ({
  components: { Input },
  template: '<Input disabled />',
})
```

### 4. Show Composition

For molecules/organisms, show how atoms compose:

```typescript
// FormField.story.ts
export const Default = () => ({
  components: { FormField },
  template: '<FormField label="Email" />',
})

export const WithError = () => ({
  components: { FormField },
  template: '<FormField label="Email" error="Invalid email" />',
})

export const Required = () => ({
  components: { FormField },
  template: '<FormField label="Email" required />',
})
```

### 5. Use Controls (Histoire)

```typescript
// Button.story.ts
export const Playground = () => ({
  components: { Button },
  template: '<Button :variant="variant" :size="size">{{ text }}</Button>',
  state: () => ({
    variant: 'primary',
    size: 'md',
    text: 'Click Me',
  }),
})
```

---

## 🗂️ File Organization

```
storybook/
├── styles/
│   ├── tokens.css           # Design tokens (subatomic)
│   └── reset.css            # CSS reset
├── stories/
│   ├── atoms/               # Level 1: Atoms
│   │   ├── Button/
│   │   ├── Input/
│   │   ├── Label/
│   │   └── Icon/
│   ├── molecules/           # Level 2: Molecules
│   │   ├── FormField/
│   │   ├── SearchBar/
│   │   └── NavItem/
│   ├── organisms/           # Level 3: Organisms
│   │   ├── NavBar/
│   │   ├── ProductCard/
│   │   └── PricingTable/
│   ├── templates/           # Level 4: Templates (optional)
│   │   ├── HomeTemplate/
│   │   └── DashboardTemplate/
│   └── index.ts             # Barrel export
├── ATOMIC_DESIGN_PHILOSOPHY.md  # This file
└── package.json
```

---

## ✅ Checklist for Creating Stories

### For Atoms

- [ ] Component uses design tokens (no hardcoded values)
- [ ] Component is accessible (ARIA, keyboard nav)
- [ ] All variants shown in stories
- [ ] All states shown (default, hover, active, disabled)
- [ ] All sizes shown (if applicable)
- [ ] Props documented in story controls
- [ ] Component is single-purpose

### For Molecules

- [ ] Composed of 2-3 atoms
- [ ] Single, focused purpose
- [ ] Context-independent
- [ ] All composition variations shown
- [ ] State management clear
- [ ] Props documented

### For Organisms

- [ ] Composed of multiple molecules/atoms
- [ ] Forms a distinct UI section
- [ ] Reusable across pages
- [ ] All layout variations shown
- [ ] Responsive behavior documented
- [ ] Complex interactions demonstrated

---

## 🚫 Anti-Patterns to Avoid

### ❌ Don't Skip Levels

**Bad:**
```vue
<!-- Button.vue trying to be an organism -->
<template>
  <button>
    <Icon />
    <Label />
    <Badge />
    <Tooltip />
  </button>
</template>
```

**Good:**
```vue
<!-- Button.vue as a simple atom -->
<template>
  <button>
    <slot />
  </button>
</template>
```

### ❌ Don't Hardcode Values

**Bad:**
```css
.button {
  padding: 12px 24px;
  background: #0066cc;
  border-radius: 8px;
}
```

**Good:**
```css
.button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-primary);
  border-radius: var(--border-radius-md);
}
```

### ❌ Don't Mix Levels

**Bad:**
```vue
<!-- Molecule trying to include an organism -->
<template>
  <div class="search-bar">
    <Input />
    <Button />
    <NavBar /> <!-- ❌ Organism in a molecule -->
  </div>
</template>
```

**Good:**
```vue
<!-- Molecule with only atoms -->
<template>
  <div class="search-bar">
    <Input />
    <Button />
  </div>
</template>
```

### ❌ Don't Create "God Components"

**Bad:**
```vue
<!-- SuperCard.vue trying to do everything -->
<template>
  <div>
    <Image />
    <Title />
    <Description />
    <Tags />
    <Author />
    <Comments />
    <RelatedPosts />
    <ShareButtons />
  </div>
</template>
```

**Good:**
```vue
<!-- ProductCard.vue with focused purpose -->
<template>
  <div>
    <Image />
    <Title />
    <Price />
    <Button />
  </div>
</template>
```

---

## 🎓 Learning Resources

- **Brad Frost's Atomic Design:** https://atomicdesign.bradfrost.com/
- **Pattern Lab:** https://patternlab.io/
- **Storybook Best Practices:** https://storybook.js.org/docs/vue/writing-stories/introduction

---

## 📌 Summary

**Atomic Design Hierarchy:**
```
Design Tokens (Subatomic)
    ↓
Atoms (Button, Input, Label)
    ↓
Molecules (FormField, SearchBar)
    ↓
Organisms (NavBar, ProductCard)
    ↓
Templates (HomeTemplate, DashboardTemplate)
    ↓
Pages (Actual application views)
```

**Key Principles:**
1. ✅ Start with design tokens
2. ✅ Build atoms first
3. ✅ Compose upward (atoms → molecules → organisms)
4. ✅ Keep components focused and single-purpose
5. ✅ Show all variants and states in stories
6. ✅ Use design tokens, never hardcode
7. ✅ Make components accessible
8. ✅ Document everything in storybook

---

**Created by:** TEAM-FE-000  
**For:** orchyra-storybook design system  
**Purpose:** Guide component creation and story writing
