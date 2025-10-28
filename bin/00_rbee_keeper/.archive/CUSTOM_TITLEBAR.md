# Custom Titlebar

**TEAM-334:** Replaced native window decorations with custom React titlebar

## What Changed

### 1. Tauri Configuration

**File:** `tauri.conf.json`

```json
{
  "app": {
    "windows": [{
      "decorations": false,  // ← Removes native titlebar
      "transparent": false
    }]
  }
}
```

### 2. Custom Titlebar Component

**File:** `ui/src/components/CustomTitlebar.tsx`

Features:
- **Drag area** - `data-tauri-drag-region` attribute allows window dragging
- **Window controls** - Minimize, Maximize, Close buttons using `Button` component
- **App branding** - `BrandLogo` component from `@rbee/ui/molecules`
- **Styled** - Uses design system components (Button, BrandLogo)
- **Height** - `h-10` (40px) for proper button sizing

### 3. Shell Component

**File:** `ui/src/components/Shell.tsx`

Wraps the entire app with proper layout structure:

```tsx
<div className="fixed inset-0 flex flex-col">
  <CustomTitlebar />  {/* ← Titlebar at top */}
  
  <div className="flex-1 flex overflow-hidden">
    <KeeperSidebar />  {/* ← Sidebar on left */}
    <main className="flex-1 overflow-auto">
      {children}  {/* ← Page content */}
    </main>
  </div>
</div>
```

**Key points:**
- `fixed inset-0` - Fills entire window without needing h-screen/w-screen
- `flex flex-col` - Vertical layout (titlebar on top, content below)
- `flex-1` on content area - Takes remaining space after titlebar
- `overflow-auto` on main - Pages can scroll independently

### 4. App Integration

**File:** `ui/src/App.tsx`

```tsx
<Shell>
  <Routes>
    <Route path="/" element={<KeeperPage />} />
    <Route path="/settings" element={<SettingsPage />} />
    <Route path="/help" element={<HelpPage />} />
  </Routes>
</Shell>
```

Clean and simple - Shell handles all layout concerns.

## Benefits

✅ **Full control** - Customize titlebar appearance, colors, layout  
✅ **Consistent UI** - Titlebar matches app theme  
✅ **Cross-platform** - Same look on Linux, macOS, Windows  
✅ **React components** - Can add menus, search, tabs, etc.

## Customization

### Change Colors

Edit `CustomTitlebar.tsx`:

```tsx
className="h-8 bg-primary text-primary-foreground ..."
```

### Add Menu Items

```tsx
<div className="flex items-center gap-2">
  <span className="text-lg">🐝</span>
  <span className="text-sm font-medium">rbee Keeper</span>
  
  {/* Add menu items */}
  <button>File</button>
  <button>Edit</button>
  <button>View</button>
</div>
```

### Add Tabs

```tsx
<div className="flex-1 flex items-center justify-center gap-2">
  <button className="px-3 py-1 rounded">Services</button>
  <button className="px-3 py-1 rounded">Settings</button>
  <button className="px-3 py-1 rounded">Help</button>
</div>
```

### Change Height

```tsx
className="h-10 bg-background ..."  // Taller titlebar
```

## Window Controls

The titlebar includes standard window controls:

- **Minimize** - `appWindow.minimize()`
- **Maximize** - `appWindow.toggleMaximize()`
- **Close** - `appWindow.close()`

All use Tauri's `getCurrentWindow()` API.

## Drag Region

The entire titlebar is draggable via `data-tauri-drag-region` attribute. This allows users to move the window by clicking and dragging anywhere on the titlebar (except buttons).

## Testing

```bash
# Rebuild UI
cd ui && npm run build

# Rebuild app
cd .. && cargo build --bin rbee-keeper

# Run
./rbee
```

You should see:
- No native titlebar
- Custom titlebar at top with 🐝 icon
- Minimize, Maximize, Close buttons on the right
- Ability to drag window by titlebar

## Troubleshooting

**Titlebar not draggable?**
- Check `data-tauri-drag-region` attribute is present
- Make sure it's on the container div, not individual elements

**Buttons not working?**
- Check browser console for errors
- Verify `@tauri-apps/api` is installed

**Titlebar too tall/short?**
- Adjust `h-8` class (h-6, h-10, h-12, etc.)
- Update padding/margins as needed
