"""Generate a deterministic SHA-256 manifest for the tracked release tree.

Run from any directory inside the Git checkout:

    python scripts/generate_release_manifest.py --write

Only Git-tracked regular files are included. ``RELEASE_MANIFEST.sha256``
excludes itself, making the result deterministic for a given committed tree.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"


def tracked_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    rels = [Path(p.decode("utf-8")) for p in proc.stdout.split(b"\0") if p]
    return sorted(p for p in rels if p.as_posix() != MANIFEST.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest() -> str:
    lines: list[str] = []
    for rel in tracked_files():
        absolute = ROOT / rel
        if not absolute.is_file():
            raise FileNotFoundError(f"tracked path is not a regular file: {rel}")
        lines.append(f"{sha256(absolute)}  ./{rel.as_posix()}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace RELEASE_MANIFEST.sha256 instead of printing it",
    )
    args = parser.parse_args()
    content = build_manifest()
    if args.write:
        MANIFEST.write_text(content, encoding="utf-8")
        print(f"Wrote {MANIFEST} with {len(content.splitlines())} entries")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
