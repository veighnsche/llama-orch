# Testing Guide

## Setup

Install dependencies:
```bash
pnpm install
```

## Running Tests

### Run all tests
```bash
pnpm test
```

### Run tests in watch mode
```bash
pnpm test --watch
```

### Run tests with UI
```bash
pnpm test:ui
```

### Run tests with coverage
```bash
pnpm test:coverage
```

### Run specific test file
```bash
pnpm test parse-inline-markdown
```

## Test Structure

Tests are colocated with source files using the `.test.tsx` or `.test.ts` suffix.

Example:
```
src/utils/
├── parse-inline-markdown.tsx
└── parse-inline-markdown.test.tsx
```

## Writing Tests

### Basic test structure
```typescript
import { describe, it, expect } from 'vitest'

describe('MyComponent', () => {
  it('should do something', () => {
    expect(true).toBe(true)
  })
})
```

### Testing React components
```typescript
import { render } from '@testing-library/react'
import { MyComponent } from './MyComponent'

it('should render correctly', () => {
  const { container } = render(<MyComponent />)
  expect(container.textContent).toBe('Hello')
})
```

## Coverage

Coverage reports are generated in the `coverage/` directory.

View HTML coverage report:
```bash
pnpm test:coverage
open coverage/index.html
```

## CI Integration

Tests run automatically in CI on:
- Pull requests
- Commits to main branch

## Troubleshooting

### Tests not running
1. Make sure dependencies are installed: `pnpm install`
2. Check that vitest is in devDependencies
3. Verify vitest.config.ts exists

### Import errors
Make sure the path alias is configured in vitest.config.ts:
```typescript
resolve: {
  alias: {
    '@rbee/ui': path.resolve(__dirname, './src'),
  },
}
```

## Related Files

- `vitest.config.ts` - Vitest configuration
- `src/test/setup.ts` - Test setup and global configuration
- `package.json` - Test scripts
