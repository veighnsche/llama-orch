# ArchitectureHighlights

A reusable molecule component that displays a list of architecture highlights with titles and supporting details.

## Usage

```tsx
import { ArchitectureHighlights } from "@rbee/ui/molecules";
import type { ArchitectureHighlight } from "@rbee/ui/molecules";

const highlights: ArchitectureHighlight[] = [
  {
    title: "BDD-Driven Development",
    details: [
      "42/62 scenarios passing (68% complete)",
      "Live CI coverage",
    ],
  },
  {
    title: "Process Isolation",
    details: ["Worker-level sandboxes. Zero cross-leak."],
  },
];

export function MyComponent() {
  return <ArchitectureHighlights highlights={highlights} />;
}
```

## Props

### `highlights` (required)
Array of architecture highlight items. Each item should have:
- `title`: The main highlight title
- `details`: Array of supporting detail strings

### `className` (optional)
Optional className for the container.

## Features

- Clean, semantic list structure
- Consistent typography hierarchy
- Supports multiple detail lines per highlight
- Responsive design
