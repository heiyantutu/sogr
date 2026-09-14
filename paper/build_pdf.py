"""Compile paper/SOGR.typ -> paper/SOGR.pdf with Typst (DejaVu layout)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TYP = ROOT / "SOGR.typ"
PDF = ROOT / "SOGR.pdf"
FONTS = ROOT / "fonts"

TYPST_CANDIDATES = [
    shutil.which("typst"),
    str(
        Path.home()
        / "AppData/Local/Microsoft/WinGet/Packages"
        / "Typst.Typst_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "typst-x86_64-pc-windows-msvc/typst.exe"
    ),
]


def typst_bin() -> str:
    for p in TYPST_CANDIDATES:
        if p and Path(p).exists():
            return p
    raise SystemExit("typst not found; install with: winget install Typst.Typst")


def main() -> int:
    cmd = [
        typst_bin(),
        "compile",
        "--font-path",
        str(FONTS),
        str(TYP),
        str(PDF),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
