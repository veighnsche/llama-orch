# TEAM-291: Commands Sidebar Component

**Status:** ✅ COMPLETE

**Mission:** Extract the commands sidebar into a separate reusable component.

## Implementation

### New Component

**File:** `src/components/CommandsSidebar.tsx`

**Purpose:** Reusable sidebar component for CLI commands in the Bee Keeper page.

**Structure:**
```tsx
<Sidebar collapsible="none">
  <SidebarHeader>
    Commands
  </SidebarHeader>
  <SidebarContent>
    <SidebarGroup>Queen</SidebarGroup>
    <SidebarGroup>Hive</SidebarGroup>
    <SidebarGroup>Git (SSH)</SidebarGroup>
  </SidebarContent>
</Sidebar>
```

### Commands Included

**Queen (5 commands):**
- Start Queen
- Stop Queen
- Restart Queen
- Queen Status
- Build Queen

**Hive (5 commands):**
- Start Hive
- Stop Hive
- Restart Hive
- Hive Status
- Build Hive

**Git (SSH) (3 commands):**
- Clone Repository
- Pull Updates
- Check Status

**Total:** 13 commands

## Updated Keeper Page

**Before:**
```tsx
export default function KeeperPage() {
  return (
    <SidebarProvider>
      <div>
        <Sidebar>
          {/* 100+ lines of sidebar code */}
        </Sidebar>
        <div>{/* Output area */}</div>
      </div>
    </SidebarProvider>
  );
}
```

**After:**
```tsx
export default function KeeperPage() {
  return (
    <SidebarProvider>
      <div>
        <CommandsSidebar />
        <div>{/* Output area */}</div>
      </div>
    </SidebarProvider>
  );
}
```

## Benefits

### Code Organization
- ✅ Cleaner keeper page (reduced from ~120 LOC to ~40 LOC)
- ✅ Reusable component
- ✅ Single responsibility
- ✅ Easier to maintain

### Maintainability
- ✅ Commands defined in one place
- ✅ Easy to add/remove commands
- ✅ Easy to modify styling
- ✅ Can be tested independently

### Future Enhancements
- ✅ Easy to add click handlers
- ✅ Easy to add active states
- ✅ Easy to add icons
- ✅ Can accept props for customization

## Files

### Created
1. **`src/components/CommandsSidebar.tsx`** (100 LOC)
   - Sidebar component with all commands
   - Uses proper Sidebar components from `@rbee/ui/atoms`
   - Grouped by operation type

### Modified
2. **`src/app/keeper/page.tsx`** (40 LOC, was ~120 LOC)
   - Imports CommandsSidebar component
   - Simplified layout
   - Focused on page structure only

## Component API

### Current (No Props)
```tsx
<CommandsSidebar />
```

### Future (With Props)
```tsx
<CommandsSidebar 
  onCommandClick={(command) => handleCommand(command)}
  activeCommand="Start Queen"
  disabledCommands={["Stop Queen"]}
/>
```

## Styling

Uses the same Sidebar components as main navigation:
- `Sidebar` - Main container
- `SidebarHeader` - Header section
- `SidebarContent` - Scrollable content
- `SidebarGroup` - Command groups
- `SidebarGroupLabel` - Group labels
- `SidebarGroupContent` - Group content
- `SidebarMenu` - Menu container
- `SidebarMenuItem` - Menu items
- `SidebarMenuButton` - Command buttons

## Consistency

**Main Navigation Sidebar:**
- Uses Sidebar components ✅
- Grouped navigation ✅
- Scrollable content ✅

**Commands Sidebar:**
- Uses Sidebar components ✅
- Grouped commands ✅
- Scrollable content ✅

**Both sidebars use the SAME components!**

## Future Enhancements

### Phase 1: Interactivity
```tsx
export function CommandsSidebar({ onCommandClick }) {
  return (
    <Sidebar>
      <SidebarMenuItem>
        <SidebarMenuButton onClick={() => onCommandClick('start-queen')}>
          Start Queen
        </SidebarMenuButton>
      </SidebarMenuItem>
    </Sidebar>
  );
}
```

### Phase 2: Active States
```tsx
export function CommandsSidebar({ activeCommand }) {
  return (
    <SidebarMenuItem>
      <SidebarMenuButton isActive={activeCommand === 'start-queen'}>
        Start Queen
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}
```

### Phase 3: Icons
```tsx
import { PlayIcon, StopIcon, RefreshIcon } from 'lucide-react';

<SidebarMenuButton>
  <PlayIcon />
  <span>Start Queen</span>
</SidebarMenuButton>
```

### Phase 4: Command Data
```tsx
const commands = [
  { id: 'start-queen', label: 'Start Queen', icon: PlayIcon, group: 'Queen' },
  { id: 'stop-queen', label: 'Stop Queen', icon: StopIcon, group: 'Queen' },
  // ...
];

export function CommandsSidebar({ commands, onCommandClick }) {
  return (
    <Sidebar>
      {commands.map(cmd => (
        <SidebarMenuItem key={cmd.id}>
          <SidebarMenuButton onClick={() => onCommandClick(cmd.id)}>
            <cmd.icon />
            <span>{cmd.label}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
    </Sidebar>
  );
}
```

## Engineering Rules Compliance

- ✅ Reusable component
- ✅ Single responsibility
- ✅ Uses proper Sidebar components
- ✅ Consistent with main navigation
- ✅ Clean separation of concerns
- ✅ Team signature (TEAM-291)

---

**TEAM-291 COMPLETE** - Commands sidebar extracted into reusable component.
