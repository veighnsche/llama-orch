# Storybook Setup

Storybook is configured for the `@rbee/ui` component library to provide interactive component documentation and testing.

## Running Storybook

```bash
# Start Storybook development server
pnpm storybook

# Build Storybook for production
pnpm build-storybook
```

Storybook will be available at `http://localhost:6006`.

## What's Included

### Interactive Component Preview
- Live component rendering
- Interactive controls for all props
- Real-time prop updates
- Multiple story variants per component

### Documentation
- Auto-generated prop tables
- Usage examples
- Component descriptions
- Code snippets

### Features
- **Dark Mode Toggle**: Test components in both light and dark themes
- **Responsive Preview**: Test different viewport sizes
- **Accessibility**: Built-in accessibility testing
- **Controls Panel**: Modify props interactively
- **Actions Panel**: View component events

## Stories Created

### Atoms
- **Button** (`src/atoms/Button.stories.tsx`)
  - Default, Secondary, Outline, Ghost, Destructive variants
  - Small, Medium, Large, Icon sizes
  - All Variants showcase
  - All Sizes showcase
  - Disabled state

- **Badge** (`src/atoms/Badge.stories.tsx`)
  - Default, Secondary, Outline, Destructive variants
  - All Variants showcase
  - Use Cases examples

### Molecules
- **Card** (`src/molecules/Card.stories.tsx`)
  - Default card
  - With Footer
  - With Badge
  - Feature Grid (2x2 grid example)
  - Minimal Card
  - Long Content

### Introduction
- **Welcome Page** (`src/Introduction.mdx`)
  - Design philosophy
  - Primary color showcase
  - Component categories
  - Usage guide

## Story Structure

Each story file follows this pattern:

```tsx
import type { Meta, StoryObj } from '@storybook/react';
import { ComponentName } from './ComponentName';

const meta = {
  title: 'Category/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    // Prop controls configuration
  },
} satisfies Meta<typeof ComponentName>;

export default meta;
type Story = StoryObj<typeof meta>;

export const StoryName: Story = {
  args: {
    // Default props
  },
};
```

## Adding New Stories

1. Create a `.stories.tsx` file next to your component
2. Import the component
3. Define the meta configuration
4. Export story variants
5. Storybook will automatically discover and display it

Example:

```tsx
// src/atoms/NewComponent.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { NewComponent } from './NewComponent';

const meta = {
  title: 'Atoms/NewComponent',
  component: NewComponent,
  tags: ['autodocs'],
} satisfies Meta<typeof NewComponent>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    prop: 'value',
  },
};
```

## Configuration

### Main Config (`.storybook/main.ts`)
- Story file patterns
- Addons configuration
- Framework settings (React + Vite)

### Preview Config (`.storybook/preview.ts`)
- Global styles import
- Parameter defaults
- Theme toggle configuration
- Background options

### Vite Config (`vite.config.ts`)
- Tailwind CSS v4 integration
- Build optimization

## Best Practices

### Story Naming
- Use descriptive names: `Default`, `WithIcon`, `LargeSize`
- Group related variants: `AllVariants`, `AllSizes`
- Show use cases: `FeatureGrid`, `UseCases`

### Documentation
- Add component descriptions
- Document all props with argTypes
- Include usage examples
- Show multiple variants

### Organization
- Keep stories next to components
- One story file per component
- Group by atomic design category (Atoms/Molecules/Organisms)

### Variants
- Show all visual variants
- Include edge cases
- Demonstrate disabled/loading states
- Showcase responsive behavior

## Addons Included

- **@storybook/addon-essentials**: Core addons (docs, controls, actions, etc.)
- **@storybook/addon-interactions**: User interaction testing
- **@storybook/addon-links**: Link between stories
- **@chromatic-com/storybook**: Visual regression testing integration

## Dark Mode

Dark mode is implemented via the theme toggle in the toolbar. The preview automatically applies the `.dark` class to the root element, which activates dark mode CSS variables.

To test dark mode:
1. Click the theme toggle button in the Storybook toolbar
2. Components will re-render with dark mode styles
3. All CSS variables update automatically

## Responsive Testing

Use the viewport toolbar to test components at different screen sizes:
- Mobile
- Tablet
- Desktop
- Custom sizes

## Next Steps

1. **Add more stories** as new components are created
2. **Add interaction tests** using `@storybook/test`
3. **Set up visual regression testing** with Chromatic
4. **Document complex patterns** with MDX stories
5. **Add accessibility checks** using addon-a11y
