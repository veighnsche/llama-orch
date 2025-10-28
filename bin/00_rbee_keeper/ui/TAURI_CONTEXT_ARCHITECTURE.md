# Tauri Context Architecture

## Overview

Global context to detect if the app is running in Tauri or browser mode, allowing components to adapt their behavior accordingly.

## Implementation

### 1. **TauriContext** (`src/contexts/TauriContext.tsx`)

```typescript
const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
```

- Checks for `__TAURI_INTERNALS__` at mount time
- Provides `isTauri` boolean to all child components
- Single source of truth for environment detection

### 2. **TauriProvider** (Root Level)

Wraps the entire app in `main.tsx`:

```tsx
<TauriProvider>
  <ThemeProvider>
    <App />
  </ThemeProvider>
</TauriProvider>
```

### 3. **useTauri() Hook**

Any component can access the Tauri state:

```typescript
const { isTauri } = useTauri();
```

## Usage in CustomTitlebar

### Before (Local Detection)
```typescript
const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
```
- Checked in every component
- Duplicated logic
- Hard to test

### After (Context)
```typescript
const { isTauri } = useTauri();
```
- Single check at root
- Consistent across app
- Easy to mock for testing

## Behavior

### In Tauri Mode (`isTauri = true`)
- Shows window controls (minimize, maximize, close)
- All Tauri APIs available
- Full native functionality

### In Browser Mode (`isTauri = false`)
- Hides window controls completely
- Shows "(Browser Mode)" label
- No Tauri API calls attempted
- Graceful degradation

## Benefits

1. **Single Source of Truth** - Environment checked once at root
2. **No Duplication** - All components use same context
3. **Testable** - Easy to mock `TauriProvider` in tests
4. **Type-Safe** - TypeScript ensures correct usage
5. **Performance** - Check happens once, not on every render
6. **Clean UI** - Buttons hidden instead of disabled in browser

## Future Use Cases

The `useTauri()` hook can be used anywhere in the app:

```typescript
// Conditional rendering
const { isTauri } = useTauri();
if (isTauri) {
  // Use Tauri-specific features
}

// Conditional API calls
const handleSave = async () => {
  if (isTauri) {
    await invoke("save_file", { path });
  } else {
    // Use browser localStorage or API
  }
};

// Feature detection
{isTauri ? <TauriFeature /> : <BrowserFallback />}
```

## Testing

### Mock TauriProvider for Tests

```typescript
// In test setup
const MockTauriProvider = ({ children, isTauri = false }) => (
  <TauriContext.Provider value={{ isTauri }}>
    {children}
  </TauriContext.Provider>
);

// Test browser mode
<MockTauriProvider isTauri={false}>
  <CustomTitlebar />
</MockTauriProvider>

// Test Tauri mode
<MockTauriProvider isTauri={true}>
  <CustomTitlebar />
</MockTauriProvider>
```

## Error Handling

Still wrapped in `CustomTitlebarErrorBoundary`:
- Catches runtime errors
- Shows fallback UI
- Logs to console
- Prevents app crashes

## Architecture Diagram

```
main.tsx
  └─ TauriProvider (detects environment once)
      └─ ThemeProvider
          └─ App
              └─ CustomTitlebar
                  └─ useTauri() → { isTauri: boolean }
```

## Files

- `src/contexts/TauriContext.tsx` - Context definition
- `src/main.tsx` - Provider setup
- `src/components/CustomTitlebar.tsx` - Consumer example
