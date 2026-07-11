# Windows DOCX rendering QA

Use the project-local stable renderer instead of calling LibreOffice directly or
using the generic renderer when working on this Windows workstation.

## Why

The generic document renderer calls `soffice` by command name. On this machine,
LibreOffice is installed at:

```text
C:\Program Files\LibreOffice\program\soffice.exe
```

and may not be available on `PATH`. Direct LibreOffice calls can also produce
non-fatal profile or internal messages in the terminal. The stable renderer
uses:

- the explicit LibreOffice executable path;
- a clean temporary LibreOffice profile for each render;
- captured stdout/stderr, printed only on real failure;
- the bundled Poppler `pdftoppm.exe` for PNG page rendering.

## Command

```powershell
$py = 'C:\Users\11844\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$script = 'D:\mimic-iv-3.1\postop_aki_dynamic_prediction\scripts\render_docx_windows_stable.py'
$docx = 'D:\mimic-iv-3.1\postop_aki_dynamic_prediction\outputs\manuscript_package_v8_critical_care\critical_care_main_manuscript_zh_review.docx'
$out = 'D:\mimic-iv-3.1\postop_aki_dynamic_prediction\outputs\manuscript_package_v8_critical_care\_qa_render_zh_stable'

& $py $script $docx --output-dir $out --keep-pdf
```

Expected clean output looks like:

```text
Rendered 16 page(s): ...\_qa_render_zh_stable
```

## Notes

- `_qa_render_*` folders are temporary QA artifacts and should not be submitted.
- The manuscript package ZIP builder excludes `_qa*` and `_render*` folders.
- If LibreOffice is moved or reinstalled, pass an explicit path:

```powershell
& $py $script $docx --output-dir $out --soffice 'C:\Path\To\soffice.exe'
```
