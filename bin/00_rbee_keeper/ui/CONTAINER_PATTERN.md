# Container Pattern (TEAM-340)

## Rule Zero: Components Are Self-Contained

**Components handle their own data fetching. Pages just compose components.**

## The Pattern

### ✅ CORRECT: Self-Contained Component

```tsx
// Component handles its own data fetching
function MyCardContent() {
  const { data } = useMyStore();
  return <Card>...</Card>;
}

export function MyCard() {
  return (
    <DaemonContainer
      cacheKey="my-card"
      metadata={{ name: "My Card", description: "..." }}
      fetchFn={() => useMyStore.getState().fetchData()}
    >
      <MyCardContent />
    </DaemonContainer>
  );
}
```

### ✅ CORRECT: Page Composition

```tsx
// Page just composes self-contained components
export default function MyPage() {
  return (
    <PageContainer title="My Page" description="...">
      <MyCard />
      <AnotherCard />
    </PageContainer>
  );
}
```

### ❌ WRONG: Page-Level Container

```tsx
// DON'T wrap components in DaemonContainer at page level
export default function MyPage() {
  return (
    <PageContainer title="My Page" description="...">
      <DaemonContainer fetchFn={...}>  {/* ❌ WRONG */}
        <MyCard />
      </DaemonContainer>
    </PageContainer>
  );
}
```

## Why This Matters

1. **Single Responsibility**: Components own their data fetching logic
2. **Reusability**: Components work anywhere without external wrappers
3. **Consistency**: Same pattern everywhere, no confusion
4. **Maintainability**: Data fetching logic lives with the component

## Current Implementation

### Self-Contained Components

- ✅ `QueenCard` - Wraps itself with DaemonContainer
- ✅ `HiveCard` - Wraps itself with DaemonContainer
- ✅ `InstalledHiveList` - Wraps itself with DaemonContainer (fetches list only)
- ✅ `QueenIframe` - Wraps itself with DaemonContainer

### Page Composition

- ✅ `ServicesPage` - Composes QueenCard, InstalledHiveList, InstallHiveCard
- ✅ `QueenPage` - Composes QueenIframe
- ✅ `HelpPage` - Static content, no containers needed
- ✅ `SettingsPage` - Static content, no containers needed

## Exception: List Components

`InstalledHiveList` uses DaemonContainer to fetch the **list** of hives, then renders individual `HiveCard` components. Each `HiveCard` fetches its **own status** independently.

This is correct because:
- List fetching is a separate concern from individual item status
- Each HiveCard is self-contained and can be used independently
- Parallel data fetching (list + individual statuses) improves performance

## Migration Checklist

When creating a new component that needs data:

- [ ] Create `ComponentContent` function that uses store hooks
- [ ] Create exported `Component` function that wraps with DaemonContainer
- [ ] Use component directly in pages without additional wrappers
- [ ] Verify component works in isolation (Storybook, etc.)

## Anti-Patterns to Avoid

1. ❌ Wrapping components with DaemonContainer in pages
2. ❌ Passing fetch functions as props from pages to components
3. ❌ Using useEffect in components when DaemonContainer handles it
4. ❌ Creating "container" components that just wrap other components

## Benefits

- **No prop drilling**: Components get data from stores, not props
- **Better loading states**: Each component shows its own loading/error UI
- **Parallel fetching**: Multiple components fetch data simultaneously
- **Easier testing**: Components are self-contained units
- **Clear ownership**: Data fetching logic lives with the component
