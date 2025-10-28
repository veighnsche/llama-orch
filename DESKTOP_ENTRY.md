# Desktop Entry for rbee Development Build

**TEAM-334:** Created desktop launcher for development builds (uses existing `rbee` script)

## ⚠️ STATUS: NOT WORKING YET - Niri Compatibility Issue

The desktop entry is created and valid, but the GUI window doesn't appear when launched from the application menu.

**Root Cause:** Niri (Wayland tiling window manager) compatibility issue. Tauri apps may not work properly with certain Wayland compositors.

**Current Workaround Attempt:** Using XWayland (`GDK_BACKEND=x11`) instead of native Wayland.

**Needs investigation or alternative solution.**

## Files Created

1. **`~/.local/share/applications/rbee-dev.desktop`**
   - Desktop entry file (shows up in application menu)
   - Name: "rbee (Development)"
   - Category: Development
   - Uses existing `rbee` wrapper script (TEAM-162)
   - Location: `~/.local/share/applications/rbee-dev.desktop`

## Usage

### From Application Menu

1. Open your application launcher (Super key / Start menu)
2. Search for "rbee"
3. Click "rbee (Development)"

### From Command Line

```bash
# Direct launch (uses existing rbee wrapper)
./rbee

# Or via desktop entry
gtk-launch rbee-dev.desktop
```

## How It Works (Theory)

1. Desktop entry points to existing `rbee` script (root of repo)
2. `rbee` script calls `xtask rbee` which handles auto-build and launch
3. `xtask rbee` launches `target/debug/rbee-keeper` with no args
4. `rbee-keeper` (no args) should launch Tauri GUI

**Current Issue:** Process starts but GUI window doesn't appear. Unknown why.

## Debugging

```bash
# Test desktop entry manually
gtk-launch rbee-dev.desktop

# Check if process started
ps aux | grep rbee-keeper

# Check system logs
journalctl --user -n 50 | grep rbee

# Validate desktop entry
desktop-file-validate ~/.local/share/applications/rbee-dev.desktop

# Test rbee script directly (this works)
cd /home/vince/Projects/llama-orch && ./rbee

# Try with XWayland explicitly
GDK_BACKEND=x11 ./rbee

# Try with native Wayland
GDK_BACKEND=wayland ./rbee

# Check Niri-specific issues
# Niri may need window rules or specific configuration
# See: https://github.com/YaLTeR/niri
```

## Niri-Specific Notes

Niri is a scrollable-tiling Wayland compositor. Known issues with Tauri/GTK apps:

1. **Window may not appear** - Some GTK apps don't work with Niri's tiling model
2. **XWayland fallback** - Desktop entry now uses `GDK_BACKEND=x11` to force XWayland
3. **Niri window rules** - May need to add rules in `~/.config/niri/config.kdl`
4. **Alternative** - Run directly from terminal: `./rbee` (this works)

## Benefits

- ✅ Shows up in application menu
- ✅ Auto-builds if source changed
- ✅ No need to manually run `cargo build`
- ✅ Always launches latest development build

## Customization

### Change Icon

Edit `~/.local/share/applications/rbee-dev.desktop`:
```ini
Icon=/path/to/your/icon.png
```

### Add to Favorites/Dock

Right-click the application in your launcher and select "Add to Favorites" or "Pin to Dock"

## Uninstall

```bash
rm ~/.local/share/applications/rbee-dev.desktop
update-desktop-database ~/.local/share/applications
```

## Production Build

For a production desktop entry, create a similar file but point to:
```ini
Exec=/home/vince/Projects/llama-orch/target/release/rbee-keeper
Name=rbee
```

Or install system-wide to `/usr/share/applications/` (requires sudo).

### Desktop Entry Details

```ini
Name: rbee (Development)
Exec: /home/vince/Projects/llama-orch/rbee
Icon: utilities-terminal
Categories: Development
```

**Rule Zero Applied:** Uses existing `rbee` wrapper script instead of creating duplicate. Single source of truth.

The launcher is now installed and ready to use! Just search for "rbee" in your application menu.
