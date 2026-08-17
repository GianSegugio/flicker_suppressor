# Flicker Suppressor GUI build

## Development

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_gui_dev.ps1
```

## Portable Windows build

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_windows.ps1
```

The portable folder is produced under `build\gui_main.dist`.

## Logo

The GUI expects:

- `assets\logo.png` - full-resolution logo used by About;
- `assets\logo_256.png` - window/taskbar/build icon source.

The build script creates `assets\app.ico` automatically. Replacing the two PNGs is enough; no source edit is required.
