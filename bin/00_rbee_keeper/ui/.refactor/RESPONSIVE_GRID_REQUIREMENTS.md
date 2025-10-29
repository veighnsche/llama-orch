# Responsive Grid Requirements (TEAM-369)

## The 4 Scenarios

### 1. Half-width (~960px) + Narration Open
- **Window:** ~960px
- **Narration panel:** ~300px
- **Content area:** ~660px
- **Requirement:** 1 column (stacked)

### 2. Half-width (~960px) + Narration Closed
- **Window:** ~960px
- **Content area:** ~960px (minus sidebar ~150px = ~810px)
- **Requirement:** 2 columns side-by-side

### 3. Full-screen (1920px) + Narration Open
- **Window:** 1920px
- **Narration panel:** ~300px
- **Content area:** ~1620px
- **Requirement:** 3 columns side-by-side

### 4. Full-screen (1920px) + Narration Closed
- **Window:** 1920px
- **Content area:** ~1920px (minus sidebar ~150px = ~1770px)
- **Requirement:** 4 columns side-by-side

---

## Tailwind Breakpoints (Window Width)

- `sm`: 640px
- `md`: 768px
- `lg`: 1024px
- `xl`: 1280px
- `2xl`: 1536px

---

## Solution

**Problem:** Tailwind breakpoints are based on WINDOW width, not content area width!

**Narration Open:**
- Window < 1280px → 1 column (covers scenario 1)
- Window >= 1280px → 3 columns (covers scenario 3)

**Narration Closed:**
- Window < 640px → 1 column
- Window >= 640px → 2 columns (covers scenario 2)
- Window >= 1536px → 4 columns (covers scenario 4)

**The key:** Skip 3 columns when narration is closed! Go straight from 2 to 4.

---

## Implementation

```typescript
showNarration
  ? "grid-cols-1 xl:grid-cols-3"
  : "grid-cols-1 sm:grid-cols-2 2xl:grid-cols-4"
```

**Narration Open:** 1 col → 3 col at xl (1280px)
**Narration Closed:** 1 col → 2 col at sm (640px) → 4 col at 2xl (1536px)
