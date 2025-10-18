# Component Consolidation Plan: CheckListItem → BulletListItem

**Date**: 2025-10-15  
**Status**: ✅ Ready to Execute  
**Impact**: Low (CheckListItem not used in production)

## Analysis

### Current State

We have two similar list item components:

#### CheckListItem
- **Location**: `src/molecules/CheckListItem/`
- **Icon**: Hardcoded Lucide `Check` icon
- **Props**: `text`, `variant` (success/primary/muted), `size` (sm/md/lg)
- **Features**: Simple check icon + text
- **Usage**: ❌ **NOT USED** in any pages or organisms

#### BulletListItem
- **Location**: `src/molecules/BulletListItem/`
- **Icon**: Configurable (`dot`, `check`, `arrow`)
- **Props**: `title`, `description`, `meta`, `variant` (dot/check/arrow), `color` (primary/chart-1-5)
- **Features**: Flexible bullet + title + optional description + optional meta
- **Usage**: ✅ Used in production (likely in feature lists)

### Overlap

| Feature | CheckListItem | BulletListItem |
|---------|---------------|----------------|
| Check icon | ✅ (hardcoded) | ✅ (variant="check") |
| Dot bullet | ❌ | ✅ (variant="dot") |
| Arrow bullet | ❌ | ✅ (variant="arrow") |
| Custom colors | ✅ (3 variants) | ✅ (6 colors) |
| Size variants | ✅ (sm/md/lg) | ❌ |
| Description | ❌ | ✅ |
| Meta text | ❌ | ✅ |

**Conclusion**: BulletListItem is more flexible and can replace CheckListItem entirely.

---

## Recommendation

**Deprecate CheckListItem** and enhance BulletListItem to cover all CheckListItem use cases.

### Why?

1. **No production usage**: CheckListItem is not used anywhere in apps or organisms
2. **Feature superset**: BulletListItem already has `variant="check"` which does the same thing
3. **More flexible**: BulletListItem supports description and meta text
4. **Reduce maintenance**: One component instead of two

---

## Migration Path

### Option 1: Delete CheckListItem (Recommended)

**Pros**:
- Clean codebase
- No confusion about which component to use
- Less maintenance

**Cons**:
- None (not used in production)

**Steps**:
1. ✅ Verify no usage in apps/organisms (DONE - confirmed not used)
2. Delete `src/molecules/CheckListItem/` directory
3. Remove from `src/molecules/index.ts` exports
4. Update documentation to recommend BulletListItem

### Option 2: Keep CheckListItem as Wrapper

**Pros**:
- Backward compatibility (if needed in future)
- Simpler API for simple check lists

**Cons**:
- Maintains two components
- Potential confusion

**Implementation**:
```tsx
// CheckListItem.tsx becomes a thin wrapper
export function CheckListItem({ text, variant, size, className }: CheckListItemProps) {
  const colorMap = {
    success: 'chart-3',
    primary: 'primary',
    muted: 'chart-1',
  }
  
  return (
    <BulletListItem
      title={text}
      variant="check"
      color={colorMap[variant]}
      className={className}
    />
  )
}
```

---

## Recommended Action: Option 1 (Delete)

Since CheckListItem is **not used anywhere**, we should delete it to keep the codebase clean.

### Execution Steps

1. **Delete CheckListItem directory**:
   ```bash
   rm -rf src/molecules/CheckListItem/
   ```

2. **Remove from exports**:
   ```tsx
   // src/molecules/index.ts
   // Remove: export * from './CheckListItem'
   ```

3. **Update BulletListItem docs** to mention it can be used for check lists:
   ```markdown
   ## When to Use
   - In feature lists (use variant="check")
   - In pricing plan features (use variant="check")
   - In benefit descriptions
   - In step-by-step instructions (use variant="arrow")
   - In comparison tables
   ```

4. **Add size support to BulletListItem** (optional enhancement):
   ```tsx
   export interface BulletListItemProps {
     // ... existing props
     size?: 'sm' | 'md' | 'lg'  // NEW
   }
   ```

---

## Alternative: Enhance BulletListItem First

If you want to preserve CheckListItem's size variants, enhance BulletListItem first:

### Add Size Prop to BulletListItem

```tsx
export interface BulletListItemProps {
  title: string
  description?: string
  meta?: string
  color?: string
  variant?: 'dot' | 'check' | 'arrow'
  size?: 'sm' | 'md' | 'lg'  // NEW
  className?: string
}

export function BulletListItem({
  title,
  description,
  meta,
  color = 'chart-3',
  variant = 'dot',
  size = 'md',  // NEW
  className,
}: BulletListItemProps) {
  // Size mappings
  const bulletSizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
  }
  
  const titleSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm font-medium',
    lg: 'text-base font-medium',
  }
  
  const descriptionSizeClasses = {
    sm: 'text-[11px]',
    md: 'text-sm',
    lg: 'text-base',
  }
  
  // Apply size classes to bullet and text
  // ...
}
```

Then delete CheckListItem.

---

## Decision Matrix

| Criteria | Delete Now | Enhance First |
|----------|-----------|---------------|
| **Simplicity** | ✅ Fastest | ⚠️ More work |
| **Feature parity** | ⚠️ Loses size variants | ✅ Full parity |
| **Risk** | ✅ None (not used) | ✅ None (not used) |
| **Future-proof** | ⚠️ May need sizes later | ✅ Covers all cases |

---

## Recommendation: Enhance First, Then Delete

**Best approach**:
1. Add `size` prop to BulletListItem (5 minutes)
2. Update BulletListItem stories to show size variants (5 minutes)
3. Delete CheckListItem (2 minutes)
4. Update docs (3 minutes)

**Total time**: ~15 minutes  
**Benefit**: Clean codebase with full feature coverage

---

## Next Steps

**Choose one**:

### A. Quick Delete (No Enhancement)
```bash
# 1. Delete CheckListItem
rm -rf src/molecules/CheckListItem/

# 2. Remove from exports
# Edit src/molecules/index.ts and remove CheckListItem export

# 3. Done!
```

### B. Enhance + Delete (Recommended)
```bash
# 1. Add size prop to BulletListItem
# 2. Update BulletListItem stories
# 3. Delete CheckListItem
# 4. Remove from exports
```

---

**Status**: ✅ Ready to execute  
**Risk**: ✅ Low (component not used)  
**Recommendation**: **Option B** (Enhance BulletListItem, then delete CheckListItem)
