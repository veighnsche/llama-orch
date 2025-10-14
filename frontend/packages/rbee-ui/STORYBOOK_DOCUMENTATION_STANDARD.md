# Storybook Documentation Standard

**Version:** 1.0  
**Date:** 2025-10-14  
**Status:** MANDATORY

---

## Purpose

This document defines the **mandatory documentation standard** for all Storybook stories in `@rbee/ui`. Every story MUST follow this standard to ensure consistency, completeness, and usefulness.

---

## Documentation Structure

Every story file MUST include:

1. **Component Description** (in meta.parameters.docs.description.component)
2. **ArgTypes Configuration** (for all props)
3. **Story Variants** (minimum 2)
4. **Usage Examples** (in component description)

---

## 1. Component Description Template

### For Atoms

```typescript
const meta = {
  title: 'Atoms/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
[1-2 sentence description of what this component is and does]

## When to Use
- [Use case 1]
- [Use case 2]
- [Use case 3]

## Variants
- **[Variant 1]**: [Description]
- **[Variant 2]**: [Description]

## Examples
\`\`\`tsx
import { ComponentName } from '@rbee/ui/atoms/ComponentName'

<ComponentName prop="value" />
\`\`\`

## Accessibility
- [Key accessibility feature 1]
- [Key accessibility feature 2]
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    // See section 2 below
  },
} satisfies Meta<typeof ComponentName>;
```

### For Molecules

```typescript
const meta = {
  title: 'Molecules/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'centered', // or 'padded'
    docs: {
      description: {
        component: `
## Overview
[2-3 sentence description of the molecule and its composition]

## Composition
This molecule is composed of:
- **[Atom 1]**: [Purpose]
- **[Atom 2]**: [Purpose]

## When to Use
- [Use case 1]
- [Use case 2]

## Variants
- **[Variant 1]**: [Description]
- **[Variant 2]**: [Description]

## Examples
\`\`\`tsx
import { ComponentName } from '@rbee/ui/molecules/ComponentName'

<ComponentName 
  prop1="value1"
  prop2="value2"
/>
\`\`\`

## Related Components
- [Related Component 1]
- [Related Component 2]
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    // See section 2 below
  },
} satisfies Meta<typeof ComponentName>;
```

### For Organisms

```typescript
const meta = {
  title: 'Organisms/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'fullscreen', // or 'padded' for smaller organisms
    docs: {
      description: {
        component: `
## Overview
[3-4 sentence description of the organism, its purpose, and complexity]

## Composition
This organism contains:
- **[Section 1]**: [Description and purpose]
- **[Section 2]**: [Description and purpose]
- **[Section 3]**: [Description and purpose]

## When to Use
- [Use case 1]
- [Use case 2]
- [Use case 3]

## Content Requirements
- **[Content Type 1]**: [Requirements and guidelines]
- **[Content Type 2]**: [Requirements and guidelines]

## Variants
- **[Variant 1]**: [Description and when to use]
- **[Variant 2]**: [Description and when to use]

## Examples
\`\`\`tsx
import { ComponentName } from '@rbee/ui/organisms/ComponentName'

<ComponentName 
  headline="Your Headline"
  description="Your description"
  cta={{
    text: "Get Started",
    href: "/signup"
  }}
/>
\`\`\`

## Used In
- Home page (\`/\`)
- [Other page] (\`/path\`)

## Related Components
- [Related Organism 1]
- [Related Organism 2]

## Accessibility
- [Key accessibility feature 1]
- [Key accessibility feature 2]
- [Keyboard navigation details]
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    // See section 2 below
  },
} satisfies Meta<typeof ComponentName>;
```

---

## 2. ArgTypes Configuration

Every prop MUST be documented with:

```typescript
argTypes: {
  propName: {
    control: 'text' | 'select' | 'boolean' | 'number' | 'object' | 'color',
    description: 'Clear description of what this prop does',
    table: {
      type: { summary: 'string' | 'number' | 'boolean' | 'object' },
      defaultValue: { summary: 'default value' },
      category: 'Appearance' | 'Content' | 'Behavior' | 'State',
    },
    options: ['option1', 'option2'], // if control is 'select'
  },
}
```

### Control Types by Prop Type

| Prop Type | Control Type | Example |
|-----------|--------------|---------|
| String | `'text'` | `title: string` |
| Enum/Union | `'select'` | `variant: 'default' \| 'secondary'` |
| Boolean | `'boolean'` | `disabled: boolean` |
| Number | `'number'` | `count: number` |
| Object | `'object'` | `data: { key: value }` |
| Color | `'color'` | `backgroundColor: string` |

### Categories

Group props by category:

- **Appearance**: Visual styling props (variant, size, color)
- **Content**: Content props (title, description, children)
- **Behavior**: Interaction props (onClick, onSubmit)
- **State**: State props (disabled, loading, error)

### Example: Complete ArgTypes

```typescript
argTypes: {
  // Appearance
  variant: {
    control: 'select',
    options: ['default', 'secondary', 'outline', 'ghost', 'destructive'],
    description: 'Visual style variant of the button',
    table: {
      type: { summary: 'string' },
      defaultValue: { summary: 'default' },
      category: 'Appearance',
    },
  },
  size: {
    control: 'select',
    options: ['sm', 'md', 'lg', 'icon'],
    description: 'Size of the button',
    table: {
      type: { summary: 'string' },
      defaultValue: { summary: 'md' },
      category: 'Appearance',
    },
  },
  
  // Content
  children: {
    control: 'text',
    description: 'Button text or content',
    table: {
      type: { summary: 'ReactNode' },
      category: 'Content',
    },
  },
  
  // State
  disabled: {
    control: 'boolean',
    description: 'Whether the button is disabled',
    table: {
      type: { summary: 'boolean' },
      defaultValue: { summary: 'false' },
      category: 'State',
    },
  },
  loading: {
    control: 'boolean',
    description: 'Whether the button is in loading state',
    table: {
      type: { summary: 'boolean' },
      defaultValue: { summary: 'false' },
      category: 'State',
    },
  },
  
  // Behavior
  onClick: {
    action: 'clicked',
    description: 'Callback fired when button is clicked',
    table: {
      type: { summary: '() => void' },
      category: 'Behavior',
    },
  },
}
```

---

## 3. Story Variants

### Minimum Required Stories

**For Atoms:**
- `Default` - Standard configuration
- At least 1 variant story (e.g., `Secondary`, `Large`, `Disabled`)

**For Molecules:**
- `Default` - Standard configuration
- At least 2 variant stories showing different configurations

**For Organisms:**
- `Default` - Standard configuration with realistic data
- At least 2 variant stories (e.g., `WithImage`, `Minimal`, `Mobile`)

### Story Naming Conventions

Use descriptive, PascalCase names:

✅ **Good:**
- `Default`
- `WithIcon`
- `LargeSize`
- `DisabledState`
- `AllVariants`
- `MobileView`

❌ **Bad:**
- `Story1`
- `test`
- `example`
- `variant_1`

### Story Structure

```typescript
export const StoryName: Story = {
  args: {
    // Props for this story
  },
  parameters: {
    // Story-specific parameters (optional)
    docs: {
      description: {
        story: 'Optional description of this specific variant',
      },
    },
  },
};
```

### Showcase Stories

For components with many variants, create showcase stories:

```typescript
export const AllVariants: Story = {
  render: () => (
    <div className="flex gap-4 flex-wrap">
      <ComponentName variant="default" />
      <ComponentName variant="secondary" />
      <ComponentName variant="outline" />
      <ComponentName variant="ghost" />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'All available visual variants displayed together',
      },
    },
  },
};

export const AllSizes: Story = {
  render: () => (
    <div className="flex gap-4 items-center">
      <ComponentName size="sm">Small</ComponentName>
      <ComponentName size="md">Medium</ComponentName>
      <ComponentName size="lg">Large</ComponentName>
    </div>
  ),
};
```

---

## 4. Mock Data Standards

### Create Realistic Mock Data

**DO:**
- ✅ Use realistic, production-like data
- ✅ Create reusable mock data files
- ✅ Use consistent naming conventions
- ✅ Include edge cases (long text, empty states)

**DON'T:**
- ❌ Use "Lorem ipsum" or placeholder text
- ❌ Use "Test 1", "Test 2" as content
- ❌ Hardcode mock data in every story
- ❌ Use unrealistic data

### Mock Data File Structure

```
src/
├── __mocks__/
│   ├── testimonials.ts
│   ├── pricing.ts
│   ├── features.ts
│   └── index.ts
```

**Example:** `src/__mocks__/testimonials.ts`

```typescript
export const mockTestimonials = [
  {
    id: '1',
    name: 'Sarah Chen',
    role: 'CTO',
    company: 'TechCorp',
    avatar: '/avatars/sarah.jpg',
    content: 'rbee transformed how we deploy LLMs. Setup took minutes, not days.',
    rating: 5,
  },
  {
    id: '2',
    name: 'Marcus Rodriguez',
    role: 'Lead Engineer',
    company: 'DataFlow',
    avatar: '/avatars/marcus.jpg',
    content: 'The orchestration layer is brilliant. We scaled from 2 to 20 GPUs seamlessly.',
    rating: 5,
  },
];

export const mockTestimonialLong = {
  id: '3',
  name: 'Dr. Emily Watson',
  role: 'Research Director',
  company: 'AI Research Institute',
  avatar: '/avatars/emily.jpg',
  content: 'After evaluating multiple solutions, rbee stood out for its simplicity and power. The ability to manage our own infrastructure while maintaining the ease of a managed service is exactly what we needed. Our team was productive within hours of setup.',
  rating: 5,
};
```

### Using Mock Data in Stories

```typescript
import { mockTestimonials } from '@/mocks/testimonials';

export const Default: Story = {
  args: {
    testimonials: mockTestimonials,
  },
};

export const SingleTestimonial: Story = {
  args: {
    testimonials: [mockTestimonials[0]],
  },
};

export const LongContent: Story = {
  args: {
    testimonials: [mockTestimonialLong],
  },
};
```

---

## 5. Dark Mode Standards

### ❌ WRONG: Separate Dark Mode Stories

```typescript
// DON'T DO THIS
export const LightMode: Story = {
  args: { theme: 'light' },
};

export const DarkMode: Story = {
  args: { theme: 'dark' },
};
```

### ✅ CORRECT: Use Storybook Toolbar

Dark mode is controlled via the Storybook toolbar. No theme prop needed in stories.

**Preview configuration handles this automatically:**

```typescript
// .storybook/preview.ts
const withTheme: Decorator = (Story, context) => {
  const theme = context.globals.theme || 'light';
  useEffect(() => {
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(theme);
  }, [theme]);
  return <Story />;
};
```

**Users toggle theme via toolbar:**
- Click theme icon in Storybook toolbar
- Component automatically re-renders with new theme
- No manual theme prop needed

---

## 6. Responsive Standards

### When to Show Responsive Variants

Show responsive variants when:
- Layout changes significantly at different breakpoints
- Mobile has different UI (e.g., hamburger menu)
- Content reflows or stacks differently

### How to Show Responsive Variants

**Option 1: Viewport Parameters**

```typescript
export const MobileView: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
};

export const TabletView: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
  },
};
```

**Option 2: Container Width**

```typescript
export const MobileWidth: Story = {
  render: () => (
    <div className="max-w-sm">
      <ComponentName />
    </div>
  ),
};
```

---

## 7. Accessibility Documentation

Every organism story MUST include accessibility information:

```typescript
docs: {
  description: {
    component: `
...

## Accessibility
- **Keyboard Navigation**: Tab through interactive elements, Enter/Space to activate
- **Screen Readers**: All images have alt text, buttons have descriptive labels
- **Focus Management**: Focus is trapped in modal when open
- **ARIA**: Uses \`aria-label\`, \`aria-describedby\` for context
- **Color Contrast**: All text meets WCAG AA standards (4.5:1 minimum)
    `,
  },
}
```

---

## 8. Quality Checklist

Before marking a story as complete, verify:

### Documentation
- [ ] Component description is clear and complete
- [ ] All sections filled out (Overview, When to Use, Examples, etc.)
- [ ] All props documented in argTypes
- [ ] Props grouped by category
- [ ] Usage examples provided

### Stories
- [ ] Minimum required stories created
- [ ] Story names are descriptive
- [ ] Realistic mock data used
- [ ] Edge cases covered (long text, empty states)

### Visual
- [ ] Renders correctly in light mode
- [ ] Renders correctly in dark mode
- [ ] No layout breaks or overflow
- [ ] Typography is readable
- [ ] Colors have good contrast

### Functional
- [ ] Interactive elements work
- [ ] No console errors or warnings
- [ ] Responsive behavior works (if applicable)

### Accessibility
- [ ] Keyboard navigation works
- [ ] Focus states visible
- [ ] Accessibility section documented

---

## 9. Examples

### Complete Atom Story Example

See: `src/atoms/Button/Button.stories.tsx`

### Complete Molecule Story Example

See: `src/molecules/Card/Card.stories.tsx` (to be created)

### Complete Organism Story Example

See: `src/organisms/HeroSection/HeroSection.stories.tsx`

---

## 10. Enforcement

This standard is **MANDATORY** for all stories in `@rbee/ui`.

**Code reviews will check:**
- [ ] All sections of component description present
- [ ] All props documented with descriptions
- [ ] Minimum story count met
- [ ] Realistic mock data used
- [ ] No separate dark mode stories
- [ ] Quality checklist passed

**Stories that don't meet this standard will be rejected.**

---

## Summary

Every story MUST have:
1. ✅ Complete component description (Overview, When to Use, Examples, etc.)
2. ✅ All props documented in argTypes with descriptions
3. ✅ Minimum 2 story variants with realistic data
4. ✅ Dark mode via toolbar (no separate stories)
5. ✅ Accessibility documentation (for organisms)
6. ✅ Quality checklist passed

**This ensures our Storybook is a world-class component library documentation system.**
