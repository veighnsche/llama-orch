# TEAM-334: Desktop Entry Status

**Date:** Oct 28, 2025  
**Status:** ⚠️ NOT WORKING YET

## What Was Done

Created a desktop entry for rbee development build that should launch the GUI from the application menu.

## Files Modified

1. **`~/.local/share/applications/rbee-dev.desktop`**
   - Desktop entry file (shows in application menu)
   - Has comments explaining the issue
   - Location documented for other developers

2. **`DESKTOP_ENTRY.md`**
   - Full documentation
   - Debugging instructions
   - Status clearly marked as not working

3. **`rbee` (root script)**
   - Added TEAM-334 comments
   - Documents desktop entry integration
   - Points to DESKTOP_ENTRY.md

4. **`xtask/src/tasks/rbee.rs`**
   - Added TEAM-334 comments
   - Explains the flow
   - Notes that it works directly but not from desktop entry

5. **`bin/00_rbee_keeper/src/main.rs`**
   - Added comments to `launch_gui()` function
   - Documents the desktop entry flow
   - Lists possible issues (Display, Wayland/X11, permissions)

## The Flow (Theory)

```
Desktop Entry
    ↓
./rbee (root script)
    ↓
xtask rbee (auto-build check)
    ↓
target/debug/rbee-keeper (no args)
    ↓
launch_gui() in main.rs
    ↓
Tauri GUI (should appear)
```

## Current Issue

- ✅ Desktop entry is valid (`desktop-file-validate` passes)
- ✅ Process starts successfully
- ✅ Works when called directly: `./rbee`
- ❌ GUI window doesn't appear when launched from desktop entry

## Debugging Commands

```bash
# Test desktop entry manually
gtk-launch rbee-dev.desktop

# Check if process started
ps aux | grep rbee-keeper

# Check system logs
journalctl --user -n 50 | grep rbee

# Validate desktop entry
desktop-file-validate ~/.local/share/applications/rbee-dev.desktop

# Test rbee script directly (this works!)
cd /home/vince/Projects/llama-orch && ./rbee
```

## Root Cause: Niri (Wayland Compositor) Incompatibility

**Confirmed:** User is running Niri, a scrollable-tiling Wayland compositor.

Tauri apps (which use WebKit/GTK) may not work properly with certain Wayland compositors, especially tiling window managers like Niri.

## Attempted Fixes

1. **XWayland fallback** - Desktop entry now uses `GDK_BACKEND=x11` to force XWayland
2. **Terminal mode** - Set `Terminal=true` to see errors
3. **Working directory** - Added `Path=` to desktop entry
4. **Environment variables** - Tried WAYLAND_DISPLAY, GDK_BACKEND

## Possible Solutions

1. **Use XWayland** - `GDK_BACKEND=x11 ./rbee` (current desktop entry setting)
2. **Niri window rules** - Add rules in `~/.config/niri/config.kdl` for rbee-keeper
3. **Run from terminal** - `./rbee` works fine when run directly
4. **Different compositor** - Test with Sway, Hyprland, or GNOME to confirm it's Niri-specific
5. **Tauri configuration** - May need Tauri-specific Wayland settings

## Next Steps for Future Developer

1. Check if GUI appears with `Terminal=true` (currently set)
2. Add explicit environment variables to desktop entry:
   ```ini
   Env=DISPLAY=:0
   Env=WAYLAND_DISPLAY=wayland-0
   ```
3. Try absolute path for Exec instead of relative
4. Check Tauri logs/errors when launched from desktop entry
5. Compare environment when launched directly vs from desktop entry:
   ```bash
   # Direct launch
   env > /tmp/direct.env
   
   # Desktop entry launch (add to rbee script)
   env > /tmp/desktop.env
   
   # Compare
   diff /tmp/direct.env /tmp/desktop.env
   ```

## Documentation

All code has comments pointing to:
- Desktop entry location: `~/.local/share/applications/rbee-dev.desktop`
- Documentation: `DESKTOP_ENTRY.md`
- Status: ⚠️ NOT WORKING YET

Future developers will know:
1. Where the desktop entry is
2. What the flow should be
3. That it's not working yet
4. How to debug it
