"""Shared paths for isolated benchmark inputs and evaluation attempts."""
from pathlib import Path

BENCHMARKS = ("verified", "multilingual", "swepro")


def input_directory(root: Path, benchmark: str) -> Path:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")
    return root / "evaluation" / benchmark / "input"


def evaluation_directory(root: Path, benchmark: str, step: int, repetition: int = 1) -> Path:
    if step < 1 or repetition < 1:
        raise ValueError("Evaluation step and repetition must be positive")
    return input_directory(root, benchmark).parent / f"step_{step:04d}" / f"run_{repetition:02d}"


def resolve_portable_python(explicit: Path | None = None) -> Path | None:
    """An unset path disables adaptation; a supplied invalid runtime is an error."""
    import os

    value = explicit if explicit is not None else os.environ.get("HARNESSADAPT_PORTABLE_PYTHON", "").strip()
    if not value:
        return None
    directory = Path(value).expanduser().resolve()
    executable = directory / "bin" / "python3"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Portable Python must contain an executable bin/python3: {directory}")
    if not any((directory / "lib").glob("python*/encodings/__init__.py")):
        raise ValueError(f"Portable Python must include its standard library (lib/python*/encodings): {directory}")
    return directory
