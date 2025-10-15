# TechnologyStack

A reusable molecule component that displays a list of technologies with optional open source CTA and architecture documentation link.

## Usage

```tsx
import { TechnologyStack } from "@rbee/ui/molecules";
import type { TechItem } from "@rbee/ui/molecules";

const technologies: TechItem[] = [
  {
    name: "Rust",
    description: "Performance + memory safety.",
    ariaLabel: "Tech: Rust",
  },
  {
    name: "Candle ML",
    description: "Rust-native inference.",
    ariaLabel: "Tech: Candle ML",
  },
];

export function MyComponent() {
  return (
    <TechnologyStack
      technologies={technologies}
      githubUrl="https://github.com/yourusername/rbee"
      license="MIT License"
      architectureUrl="/docs/architecture"
    />
  );
}
```

## Props

### `technologies` (required)
Array of technology items to display. Each item should have:
- `name`: Technology name
- `description`: Brief description
- `ariaLabel`: Accessible label for screen readers

### `showOpenSourceCTA` (optional)
Whether to show the open source CTA card. Default: `true`

### `githubUrl` (optional)
GitHub repository URL. Default: `"https://github.com/yourusername/rbee"`

### `license` (optional)
License type to display. Default: `"MIT License"`

### `showArchitectureLink` (optional)
Whether to show the architecture docs link. Default: `true`

### `architectureUrl` (optional)
Architecture docs URL. Default: `"/docs/architecture"`

### `className` (optional)
Optional className for the container.

## Features

- Animated card entrance with staggered delays
- Hover effects on technology cards
- Accessible with proper ARIA labels
- Responsive design
- Customizable CTA and links
