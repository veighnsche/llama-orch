# Storybook Quick Start Guide for Engineers

**Get started writing stories in 5 minutes**

---

## Prerequisites

```bash
cd /home/vince/Projects/llama-orch/frontend/libs/rbee-ui
pnpm install
```

---

## Step 1: Pick Your Component

Check `STORYBOOK_COMPONENT_DISCOVERY.md` for the list of components needing stories.

**Start with your assigned priority:**
- P0: Navigation, Footer, Icons
- P1: HeroSection, EmailCapture, CTASection, PricingSection, FAQSection
- P2: Other marketing sections
- P3: Page-specific variants

---

## Step 2: Understand the Component

### Read the Component Source

```bash
# Example: Button component
cat src/atoms/Button/Button.tsx
```

**Look for:**
- Props interface
- Variants (if using CVA)
- Default values
- Dependencies

### Check Where It's Used

```bash
# Find usage in commercial app
grep -r "ComponentName" /home/vince/Projects/llama-orch/frontend/bin/commercial/app/
```

---

## Step 3: Create the Story File

### File Location

Place story next to component:

```
src/atoms/Button/
├── Button.tsx
├── Button.stories.tsx  ← Create this
└── index.ts
```

### Copy Template

**For Atoms:**
```typescript
import type { Meta, StoryObj } from '@storybook/react';
import { ComponentName } from './ComponentName';

const meta = {
  title: 'Atoms/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'Brief description here.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    // Document props here
  },
} satisfies Meta<typeof ComponentName>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};
```

**For Organisms:**
```typescript
import type { Meta, StoryObj } from '@storybook/react';
import { ComponentName } from './ComponentName';

const meta = {
  title: 'Organisms/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
[Description]

## When to Use
- Use case 1
- Use case 2
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ComponentName>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};
```

---

## Step 4: Configure ArgTypes

Document each prop:

```typescript
argTypes: {
  variant: {
    control: 'select',
    options: ['default', 'secondary', 'outline'],
    description: 'Visual style variant',
    table: {
      type: { summary: 'string' },
      defaultValue: { summary: 'default' },
      category: 'Appearance',
    },
  },
  children: {
    control: 'text',
    description: 'Button content',
    table: {
      type: { summary: 'ReactNode' },
      category: 'Content',
    },
  },
  disabled: {
    control: 'boolean',
    description: 'Whether button is disabled',
    table: {
      type: { summary: 'boolean' },
      defaultValue: { summary: 'false' },
      category: 'State',
    },
  },
}
```

**Categories:**
- Appearance: variant, size, color
- Content: children, title, description
- Behavior: onClick, onSubmit
- State: disabled, loading, error

---

## Step 5: Create Story Variants

### Minimum Required

**Atoms:** 2 stories (Default + 1 variant)  
**Molecules:** 3 stories (Default + 2 variants)  
**Organisms:** 3 stories (Default + 2 variants)

### Example Stories

```typescript
export const Default: Story = {
  args: {
    variant: 'default',
    children: 'Button',
  },
};

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    children: 'Secondary',
  },
};

export const Large: Story = {
  args: {
    size: 'lg',
    children: 'Large Button',
  },
};

export const Disabled: Story = {
  args: {
    disabled: true,
    children: 'Disabled',
  },
};

// Showcase story
export const AllVariants: Story = {
  render: () => (
    <div className="flex gap-4">
      <ComponentName variant="default">Default</ComponentName>
      <ComponentName variant="secondary">Secondary</ComponentName>
      <ComponentName variant="outline">Outline</ComponentName>
    </div>
  ),
};
```

---

## Step 6: Add Realistic Mock Data

### Create Mock Data File (if needed)

```typescript
// src/__mocks__/componentData.ts
export const mockData = {
  title: "Realistic Title",
  description: "Realistic description that would appear in production",
  items: [
    { id: '1', label: 'Item 1', value: 'value1' },
    { id: '2', label: 'Item 2', value: 'value2' },
  ],
};
```

### Use in Stories

```typescript
import { mockData } from '@/__mocks__/componentData';

export const Default: Story = {
  args: {
    ...mockData,
  },
};
```

---

## Step 7: Test in Storybook

### Start Storybook

```bash
pnpm storybook
```

Opens at `http://localhost:6006`

### Test Checklist

- [ ] Story appears in sidebar
- [ ] Component renders correctly
- [ ] Toggle light/dark mode (toolbar button)
- [ ] Test all story variants
- [ ] Adjust props in Controls panel
- [ ] Check documentation tab
- [ ] No console errors

---

## Step 8: Write Documentation

### Required Sections

```typescript
docs: {
  description: {
    component: `
## Overview
[What is this component?]

## When to Use
- [Use case 1]
- [Use case 2]

## Variants
- **Default**: [Description]
- **Secondary**: [Description]

## Examples
\`\`\`tsx
import { ComponentName } from '@rbee/ui/atoms/ComponentName'

<ComponentName variant="default">
  Content
</ComponentName>
\`\`\`

## Accessibility
- [Keyboard navigation]
- [Screen reader support]
    `,
  },
}
```

**See `STORYBOOK_DOCUMENTATION_STANDARD.md` for complete requirements.**

---

## Step 9: Quality Check

### Before Committing

- [ ] All props documented in argTypes
- [ ] Minimum story count met
- [ ] Realistic mock data (no "Lorem ipsum")
- [ ] Documentation complete
- [ ] Light mode tested
- [ ] Dark mode tested
- [ ] No console errors
- [ ] Responsive tested (if applicable)

---

## Step 10: Commit

```bash
git add src/atoms/ComponentName/ComponentName.stories.tsx
git commit -m "docs(storybook): add ComponentName story"
```

---

## Common Patterns

### Pattern 1: Icon Component

```typescript
const meta = {
  title: 'Atoms/Icons/IconName',
  component: IconName,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
  },
} satisfies Meta<typeof IconName>;

export const Default: Story = {};

export const Large: Story = {
  args: { className: 'w-12 h-12' },
};

export const Colored: Story = {
  render: () => (
    <div className="flex gap-4">
      <IconName className="text-blue-600" />
      <IconName className="text-green-600" />
    </div>
  ),
};
```

### Pattern 2: Section Component

```typescript
const meta = {
  title: 'Organisms/SectionName',
  component: SectionName,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SectionName>;

export const Default: Story = {
  args: {
    headline: "Realistic Headline",
    description: "Realistic description from actual usage",
    items: mockItems,
  },
};

export const MobileView: Story = {
  args: { ...Default.args },
  parameters: {
    viewport: { defaultViewport: 'mobile1' },
  },
};
```

### Pattern 3: Form Component

```typescript
export const Default: Story = {
  args: {
    onSubmit: (data) => console.log('Submitted:', data),
  },
};

export const WithError: Story = {
  args: {
    error: 'Please enter a valid email address',
  },
};

export const Loading: Story = {
  args: {
    loading: true,
  },
};

export const Success: Story = {
  args: {
    success: true,
    message: 'Form submitted successfully!',
  },
};
```

---

## Troubleshooting

### Story Doesn't Appear

**Check:**
- File ends with `.stories.tsx`
- File is in `src/` directory
- Storybook is running
- No syntax errors

### Dark Mode Doesn't Work

**Check:**
- Preview decorator is configured (`.storybook/preview.ts`)
- Component uses Tailwind dark mode classes (`dark:bg-gray-900`)
- No hardcoded theme prop in story

### Component Doesn't Render

**Check:**
- All required props provided
- Mock data is valid
- No console errors
- Component imports correctly

### Controls Don't Work

**Check:**
- ArgTypes configured for prop
- Control type matches prop type
- Prop is not excluded in argTypes

---

## Resources

- **Full Plan:** `STORYBOOK_STORIES_PLAN.md`
- **Component List:** `STORYBOOK_COMPONENT_DISCOVERY.md`
- **Documentation Standard:** `STORYBOOK_DOCUMENTATION_STANDARD.md`
- **Existing Examples:** `src/atoms/Button.stories.tsx`, `src/atoms/Badge.stories.tsx`

---

## Need Help?

1. Check existing stories for examples
2. Read the documentation standard
3. Test in Storybook frequently
4. Ask for review early

**Now go write some stories! 🚀**
