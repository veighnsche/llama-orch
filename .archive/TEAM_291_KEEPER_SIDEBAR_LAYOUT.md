# TEAM-291: Bee Keeper Sidebar Layout

**Status:** ✅ COMPLETE

**Mission:** Redesign Bee Keeper page with a second sidebar for commands and main area for output, replacing the 2x2 grid layout.

## Problem

The 2x2 grid layout with 4 cards was cluttered and didn't provide good space for command output.

## Solution

Two-panel layout with:
1. **Commands Sidebar** (left) - 256px width, similar to main navigation
2. **Command Output Area** (right) - Full remaining width

## New Layout

```
┌──────────┬─────────────┬────────────────────────────────┐
│          │             │                                │
│  Main    │  Commands   │  Command Output                │
│  Nav     │  Sidebar    │  (Terminal-style)              │
│          │             │                                │
│  256px   │  256px      │  Flex (remaining width)        │
│          │             │                                │
└──────────┴─────────────┴────────────────────────────────┘
```

## Commands Sidebar

### Structure
```
┌─────────────────┐
│ Commands        │ ← Header
│ CLI operations  │
├─────────────────┤
│ Queen           │ ← Section label
│ • Start Queen   │
│ • Stop Queen    │
│ • Restart Queen │
│ • Queen Status  │
│ • Build Queen   │
├─────────────────┤
│ Hive (Local)    │
│ • Start Hive    │
│ • Stop Hive     │
│ • ...           │
├─────────────────┤
│ Hive (SSH)      │
│ • Install Hive  │
│ • Start Hive    │
│ • ...           │
├─────────────────┤
│ Git (SSH)       │
│ • Clone Repo    │
│ • Pull Updates  │
│ • Check Status  │
└─────────────────┘
```

### Features
- ✅ 256px fixed width (matches main sidebar)
- ✅ Scrollable content
- ✅ Grouped by operation type
- ✅ Separators between groups
- ✅ Ghost button style (hover effect)
- ✅ Full height
- ✅ Same background as main sidebar

### Sections
1. **Queen** (5 commands)
2. **Hive (Local)** (5 commands)
3. **Hive (SSH)** (6 commands)
4. **Git (SSH)** (3 commands)

**Total:** 19 commands

## Command Output Area

### Structure
```
┌────────────────────────────────────┐
│ Command Output                     │ ← Header
│ Real-time command execution results│
├────────────────────────────────────┤
│                                    │
│ Click a command to execute...      │
│                                    │
│ [Terminal output will appear here] │
│                                    │
│                                    │
│                                    │
└────────────────────────────────────┘
```

### Features
- ✅ Full remaining width
- ✅ Full height
- ✅ Scrollable content
- ✅ Monospace font
- ✅ Card with border
- ✅ Header with title and description

## CSS Classes Used

### Commands Sidebar
```tsx
className="w-64 border-r bg-sidebar flex flex-col h-full"
```
- `w-64` - 256px width
- `border-r` - Right border
- `bg-sidebar` - Sidebar background color
- `flex flex-col` - Vertical flex layout
- `h-full` - Full height

### Command Buttons
```tsx
className="w-full justify-start h-8 px-2 text-sm"
variant="ghost"
```
- `w-full` - Full width
- `justify-start` - Left-aligned text
- `h-8` - 32px height
- `px-2` - Horizontal padding
- `text-sm` - Small text
- `ghost` - Transparent with hover effect

### Output Area
```tsx
className="flex-1 flex flex-col h-full"
```
- `flex-1` - Take remaining space
- `flex flex-col` - Vertical flex layout
- `h-full` - Full height

## Comparison

### Before (2x2 Grid)
```
┌────────────┬────────────┐
│ Queen (5)  │ Hive (5)   │
├────────────┼────────────┤
│ Hive SSH   │ Git SSH    │
│ (6)        │ (3)        │
└────────────┴────────────┘
┌──────────────────────────┐
│ Command Output (small)   │
└──────────────────────────┘
```

**Issues:**
- ❌ Cluttered 2x2 grid
- ❌ Small output area
- ❌ Hard to scan commands
- ❌ Wasted space

### After (Sidebar + Output)
```
┌──────────┬────────────────────────┐
│ Commands │ Command Output         │
│          │                        │
│ Queen    │ [Large terminal area]  │
│ Hive     │                        │
│ Git      │                        │
│          │                        │
│ (scroll) │ (scroll)               │
└──────────┴────────────────────────┘
```

**Benefits:**
- ✅ Clean sidebar layout
- ✅ Large output area
- ✅ Easy to scan commands
- ✅ Efficient use of space

## Responsive Behavior

### Desktop (≥1280px)
- Commands sidebar: 256px fixed
- Output area: Remaining width
- Both areas scrollable

### Tablet (768px - 1280px)
- Commands sidebar: 256px fixed
- Output area: Smaller but usable
- Both areas scrollable

### Mobile (<768px)
- Would need separate mobile layout
- Consider tabs or accordion
- Not implemented yet

## Height Calculation

```tsx
className="h-[calc(100vh-2rem)]"
```

- `100vh` - Full viewport height
- `-2rem` - Subtract padding (32px)
- Result: Full height minus padding

## Scrolling

### Commands Sidebar
```tsx
<ScrollArea className="flex-1">
```
- Scrolls when commands exceed viewport
- Smooth scrolling
- Custom scrollbar styling

### Output Area
```tsx
<ScrollArea className="flex-1 p-4">
```
- Scrolls when output exceeds viewport
- Monospace font preserved
- Padding for readability

## Visual Hierarchy

### Level 1: Section Labels
```tsx
className="px-2 py-1.5 text-xs font-semibold text-muted-foreground"
```
- Small, bold text
- Muted color
- Clear separation

### Level 2: Command Buttons
```tsx
variant="ghost"
className="w-full justify-start h-8 px-2 text-sm"
```
- Normal text size
- Left-aligned
- Hover effect

### Level 3: Output Content
```tsx
className="font-mono text-sm"
```
- Monospace font
- Terminal-style appearance

## Interaction States

### Command Buttons
- **Default:** Transparent, normal text
- **Hover:** Light background, slightly brighter text
- **Active:** (Future) Highlighted background, bold text
- **Disabled:** (Future) Muted text, no hover

### Output Area
- **Empty:** Placeholder text
- **Running:** (Future) Streaming output
- **Complete:** (Future) Full output with exit code
- **Error:** (Future) Red text for errors

## Future Enhancements

### Commands Sidebar
1. **Active state** - Highlight currently running command
2. **Icons** - Add icons for each command type
3. **Badges** - Show running/completed status
4. **Search** - Filter commands
5. **Favorites** - Pin frequently used commands

### Output Area
1. **Tabs** - Multiple command outputs
2. **Clear button** - Clear output
3. **Copy button** - Copy output to clipboard
4. **Download** - Save output to file
5. **Syntax highlighting** - Color code output
6. **Timestamps** - Show when commands ran
7. **Exit codes** - Show success/failure

## Files Changed

### Modified
1. **`src/app/keeper/page.tsx`**
   - Complete redesign from 2x2 grid to sidebar layout
   - Removed Card components
   - Added ScrollArea components
   - Added Separator components
   - Changed button styling to ghost variant
   - Added proper height calculations

## Engineering Rules Compliance

- ✅ Layout only (no logic)
- ✅ Clean, professional design
- ✅ Consistent with main sidebar
- ✅ Efficient use of space
- ✅ Scrollable areas
- ✅ Team signature (TEAM-291)

---

**TEAM-291 COMPLETE** - Bee Keeper page redesigned with sidebar layout for better UX.
