from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, List, Optional

from .config import resolve_project_path


SUPPORTED_DATA_SUFFIXES = {".csv", ".parquet", ".sav", ".sas7bdat", ".xpt"}


def require_package(package: str, install_hint: Optional[str] = None) -> None:
    try:
        __import__(package)
    except ImportError as exc:
        hint = install_hint or f"pip install {package}"
        raise RuntimeError(f"Missing required package '{package}'. Install it with: {hint}") from exc


def find_matching_files(raw_dir: str | Path, patterns: Iterable[str]) -> List[Path]:
    base = resolve_project_path(raw_dir)
    if not base.exists():
        return []
    matches: List[Path] = []
    for path in base.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DATA_SUFFIXES:
            continue
        for pattern in patterns:
            if fnmatch.fnmatch(path.name.lower(), pattern.lower()):
                matches.append(path)
                break
    return sorted(matches)


def choose_single_file(files: List[Path], label: str) -> Path:
    if not files:
        raise FileNotFoundError(f"No {label} file found.")
    if len(files) > 1:
        names = "\n".join(f"  - {path}" for path in files)
        raise RuntimeError(f"Multiple {label} files found; keep one or configure explicitly:\n{names}")
    return files[0]


def load_table(path: str | Path):
    require_package("pandas", "pip install -r requirements.txt")
    import pandas as pd

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".sav":
        return pd.read_spss(file_path)
    if suffix in {".sas7bdat", ".xpt"}:
        return pd.read_sas(file_path, format="sas7bdat" if suffix == ".sas7bdat" else "xport")
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def write_table(df, path: str | Path) -> None:
    require_package("pandas", "pip install -r requirements.txt")
    file_path = resolve_project_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(file_path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(file_path, index=False)
        return
    raise ValueError(f"Unsupported output file type: {file_path.suffix}")
