# Packaging notes

The PyInstaller build can fail with a `PermissionError: [WinError 5]` when it tries to replace an existing executable (for example `D:\\soratool\\dist\\AI_Director_Pro.exe`). This usually happens because Windows is holding a lock on the file (e.g., the app is still running, antivirus/Defender is scanning it, or the folder is open in Explorer).

To resolve the issue:

1. Close every running instance of `AI_Director_Pro.exe` and any Explorer window showing the `dist` folder.
2. Manually delete the old `dist` and `build` folders before rebuilding, or run PyInstaller with the `--clean` flag to clear previous outputs.
3. Re-run the build from an elevated command prompt ("Run as administrator") to avoid permission problems.
4. If antivirus is blocking the file, temporarily whitelist the project folder while building.

Example clean rebuild command on Windows:

```bash
pyinstaller --clean --noconfirm AI_Director_Pro.spec
```
