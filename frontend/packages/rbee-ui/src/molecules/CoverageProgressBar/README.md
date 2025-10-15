# CoverageProgressBar

A reusable molecule component that displays test coverage progress with a visual progress bar.

## Usage

```tsx
import { CoverageProgressBar } from "@rbee/ui/molecules";

export function MyComponent() {
  return (
    <CoverageProgressBar
      label="BDD Coverage"
      passing={42}
      total={62}
    />
  );
}
```

## Props

### `label` (optional)
Label for the progress bar. Default: `"BDD Coverage"`

### `passing` (required)
Number of passing scenarios.

### `total` (required)
Total number of scenarios.

### `className` (optional)
Optional className for the container.

## Features

- Automatic percentage calculation
- Dynamic progress bar width
- Displays passing/total count
- Clean, accessible design
- Responsive layout
