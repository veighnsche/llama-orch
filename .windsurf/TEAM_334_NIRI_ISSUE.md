# TEAM-334: Niri Compatibility Issue

**Date:** Oct 28, 2025  
**Status:** ⚠️ KNOWN ISSUE - Desktop entry doesn't work with Niri

## The Problem

Desktop entry for rbee doesn't work when launched from the application menu on Niri (Wayland tiling compositor).

## Root Cause

**Niri** is a scrollable-tiling Wayland compositor. Tauri apps (which use WebKit/GTK) may not work properly with certain Wayland compositors, especially tiling window managers.

## What Works

✅ **Running directly from terminal:** `./rbee` works perfectly

## What Doesn't Work

❌ **Desktop entry launch:** Clicking "rbee (Development)" in app menu doesn't show window

## Current Desktop Entry Configuration

```ini
[Desktop Entry]
Exec=env GDK_BACKEND=x11 /home/vince/Projects/llama-orch/rbee
Terminal=true
Path=/home/vince/Projects/llama-orch
```

**Attempted fix:** Using XWayland (`GDK_BACKEND=x11`) instead of native Wayland

## Possible Solutions

### Option 1: Use Terminal (Current Workaround)

```bash
# Just run from terminal
cd /home/vince/Projects/llama-orch
./rbee
```

This works perfectly and is the recommended approach for now.

### Option 2: Niri Window Rules

Add rules to `~/.config/niri/config.kdl`:

```kdl
window-rule {
    match app-id="rbee-keeper"
    default-column-width { proportion 0.5; }
    open-on-output "your-monitor-name"
}
```

### Option 3: Test Different Backend

Try forcing different backends:

```bash
# XWayland
GDK_BACKEND=x11 ./rbee

# Native Wayland
GDK_BACKEND=wayland ./rbee

# Broadway (HTML5 backend - probably won't work)
GDK_BACKEND=broadway ./rbee
```

### Option 4: Test on Different Compositor

To confirm it's Niri-specific, test on:
- Sway
- Hyprland
- GNOME (Mutter)
- KDE (KWin)

### Option 5: Tauri Configuration

May need to add Tauri-specific Wayland configuration to `tauri.conf.json`:

```json
{
  "app": {
    "windows": [{
      "decorations": true,
      "transparent": false,
      "alwaysOnTop": false
    }]
  }
}
```

## Recommendation

**For now: Just use `./rbee` from terminal.** It works perfectly.

The desktop entry is nice-to-have but not essential. The app functions correctly when launched directly.

## Documentation

All files have been updated with comments explaining the Niri issue:

- `~/.local/share/applications/rbee-dev.desktop` - Desktop entry with comments
- `DESKTOP_ENTRY.md` - Full documentation
- `rbee` script - Comments about desktop entry
- `xtask/src/tasks/rbee.rs` - Comments about desktop entry
- `bin/00_rbee_keeper/src/main.rs` - Comments in launch_gui()

## For Future Developers

If you want to fix this:

1. Research Niri + Tauri compatibility
2. Check Niri GitHub issues for GTK/WebKit apps
3. Test with different GDK backends
4. Try Niri window rules
5. Consider filing issue with Tauri or Niri projects

But honestly, just running `./rbee` from terminal is fine for a development build.
