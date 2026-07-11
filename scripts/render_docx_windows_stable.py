"""Stable DOCX render helper for this Windows project.

Why this exists:
- The bundled generic renderer invokes ``soffice`` by command name. On this
  workstation LibreOffice is installed at ``C:\\Program Files\\LibreOffice`` and
  is not always on PATH, which causes FileNotFoundError.
- LibreOffice can also fail if the headless user profile path is malformed or
  reused while locked.
- LibreOffice sometimes prints non-fatal internal messages to stdout/stderr.

This helper uses an explicit soffice.exe path, a clean per-render profile, and
captures noisy process output. It only prints logs if conversion actually fails.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_SOFFICE_CANDIDATES = [
    Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
    Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    Path.home() / r"AppData\Local\Programs\LibreOffice\program\soffice.exe",
]

DEFAULT_PDFTOPPM_CANDIDATES = [Path(os.environ["PDFTOPPM_PATH"])] if os.environ.get("PDFTOPPM_PATH") else []


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def find_exe(explicit: str | None, candidates: list[Path], command_name: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit {command_name} path does not exist: {p}")

    for p in candidates:
        if p.exists():
            return p

    found = shutil.which(command_name)
    if found:
        return Path(found)

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {command_name}. Searched:\n{searched}")


def run_quiet(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def convert_docx_to_pdf(docx: Path, out_dir: Path, soffice: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lo_profile_codex_", dir=str(Path(tempfile.gettempdir()))) as profile:
        profile_dir = Path(profile)
        cmd = [
            str(soffice),
            f"-env:UserInstallation={file_uri(profile_dir)}",
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--nolockcheck",
            "--nodefault",
            "--norestore",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(out_dir),
            str(docx),
        ]
        proc = run_quiet(cmd)

    expected = out_dir / f"{docx.stem}.pdf"
    pdfs = sorted(out_dir.glob("*.pdf"))
    if proc.returncode != 0 or not expected.exists():
        if pdfs and pdfs[0].stat().st_size > 0:
            return pdfs[0]
        print("LibreOffice conversion failed.", file=sys.stderr)
        print("Command:", " ".join(cmd), file=sys.stderr)
        print("Return code:", proc.returncode, file=sys.stderr)
        if proc.stdout.strip():
            print("STDOUT:\n" + proc.stdout.strip(), file=sys.stderr)
        if proc.stderr.strip():
            print("STDERR:\n" + proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Failed to convert DOCX to PDF: {docx}")

    return expected


def rasterize_pdf(pdf: Path, out_dir: Path, pdftoppm: Path, dpi: int) -> None:
    prefix = out_dir / "page"
    cmd = [str(pdftoppm), "-png", "-r", str(dpi), str(pdf), str(prefix)]
    proc = run_quiet(cmd)
    pages = sorted(out_dir.glob("page-*.png"))
    if proc.returncode != 0 or not pages:
        print("PDF rasterization failed.", file=sys.stderr)
        print("Command:", " ".join(cmd), file=sys.stderr)
        print("Return code:", proc.returncode, file=sys.stderr)
        if proc.stdout.strip():
            print("STDOUT:\n" + proc.stdout.strip(), file=sys.stderr)
        if proc.stderr.strip():
            print("STDERR:\n" + proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Failed to rasterize PDF: {pdf}")


def render_one(docx: Path, output_dir: Path, soffice: Path, pdftoppm: Path, dpi: int, keep_pdf: bool) -> tuple[Path, int]:
    if not docx.exists():
        raise FileNotFoundError(f"DOCX not found: {docx}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = convert_docx_to_pdf(docx, output_dir, soffice)
    rasterize_pdf(pdf, output_dir, pdftoppm, dpi)
    pages = sorted(output_dir.glob("page-*.png"))
    if not keep_pdf:
        pdf.unlink(missing_ok=True)
    return output_dir, len(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render DOCX to PDF/PNG using stable Windows LibreOffice settings.")
    parser.add_argument("docx", help="Input DOCX file")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered PDF/PNG files")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--keep-pdf", action="store_true")
    parser.add_argument("--soffice", default=None, help="Explicit soffice.exe path")
    parser.add_argument("--pdftoppm", default=None, help="Explicit pdftoppm.exe path")
    args = parser.parse_args()

    soffice = find_exe(args.soffice, DEFAULT_SOFFICE_CANDIDATES, "soffice.exe")
    pdftoppm = find_exe(args.pdftoppm, DEFAULT_PDFTOPPM_CANDIDATES, "pdftoppm.exe")
    out_dir, page_count = render_one(
        Path(args.docx),
        Path(args.output_dir),
        soffice=soffice,
        pdftoppm=pdftoppm,
        dpi=args.dpi,
        keep_pdf=args.keep_pdf,
    )
    print(f"Rendered {page_count} page(s): {out_dir}")


if __name__ == "__main__":
    main()
