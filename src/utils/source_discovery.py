"""Safe, explainable discovery of a LegalIR source bundle for notebook launchers."""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any


SOURCE_SIGNATURE = (
    Path("scripts/run_preflight.py"),
    Path("scripts/run_smoke_test.py"),
    Path("src/config.py"),
    Path("configs/legalir_kaggle_t4_smoke.yaml"),
)


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_source_dir(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    missing = [str(relative) for relative in SOURCE_SIGNATURE if not (root / relative).is_file()]
    return {"path": str(root), "kind": "directory", "accepted": not missing, "missing_signature": missing}


def _zip_prefixes(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    prefixes: set[str] = set()
    marker = "scripts/run_preflight.py"
    for name in names:
        if name.endswith(marker):
            prefixes.add(name[: -len(marker)])
    return sorted(prefixes)


def inspect_source_zip(path: str | Path) -> dict[str, Any]:
    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        return {"path": str(archive_path), "kind": "zip", "accepted": False, "reason": f"invalid zip: {exc}"}
    prefixes = _zip_prefixes(archive_path)
    accepted_prefixes = []
    rejected = []
    for prefix in prefixes:
        missing = [str(item) for item in SOURCE_SIGNATURE if prefix + str(item) not in names]
        if missing:
            rejected.append({"prefix": prefix, "missing_signature": missing})
        else:
            accepted_prefixes.append(prefix)
    return {
        "path": str(archive_path),
        "kind": "zip",
        "accepted": len(accepted_prefixes) == 1,
        "accepted_prefixes": accepted_prefixes,
        "rejected_prefixes": rejected,
        "reason": None if len(accepted_prefixes) == 1 else "zip must contain exactly one LegalIR source root",
    }


def scan_input(input_root: str | Path) -> list[dict[str, Any]]:
    root = Path(input_root)
    results: list[dict[str, Any]] = []
    if not root.is_dir():
        return results
    seen: set[Path] = set()
    for marker in root.rglob("run_preflight.py"):
        candidate = marker.parent.parent
        if candidate not in seen:
            seen.add(candidate)
            results.append(inspect_source_dir(candidate))
    for archive in root.rglob("*.zip"):
        results.append(inspect_source_zip(archive))
    return results


def safe_extract_source_zip(zip_path: str | Path, work_root: str | Path) -> Path:
    archive = Path(zip_path)
    inspection = inspect_source_zip(archive)
    if not inspection["accepted"]:
        raise ValueError(f"Cannot extract unverified source zip: {inspection}")
    digest = sha256_file(archive)
    destination = Path(work_root) / f"legalir_source_{digest[:12]}"
    if destination.exists():
        # A rerun reuses only an already validated extraction.  Never add a
        # second copy and never delete a working directory automatically.
        if inspect_source_dir(destination)["accepted"]:
            return destination
        raise RuntimeError(f"Existing extraction is not a LegalIR source: {destination}; choose a new working path explicitly")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe zip member rejected: {member.filename}")
        staging = destination.with_name(destination.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        source.extractall(staging)
    prefix = inspection["accepted_prefixes"][0]
    source_root = staging / prefix
    if not inspect_source_dir(source_root)["accepted"]:
        shutil.rmtree(staging)
        raise RuntimeError(f"Extracted source signature failed: {source_root}")
    source_root.rename(destination)
    if staging.exists():
        staging.rmdir()
    return destination


def select_source(
    input_root: str | Path,
    work_root: str | Path,
    source_root: str | Path | None = None,
    source_zip: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Select an explicit source, or exactly one verified candidate.  No guessing."""
    diagnostics: dict[str, Any] = {"input_root": str(input_root), "candidates": []}
    if source_root:
        inspected = inspect_source_dir(source_root)
        diagnostics["explicit_source_root"] = inspected
        if not inspected["accepted"]:
            raise ValueError(f"SOURCE_ROOT is not a LegalIR source: {inspected}")
        return Path(source_root), diagnostics
    if source_zip:
        inspected = inspect_source_zip(source_zip)
        diagnostics["explicit_source_zip"] = inspected
        if not inspected["accepted"]:
            raise ValueError(f"SOURCE_ZIP is not a LegalIR source bundle: {inspected}")
        return safe_extract_source_zip(source_zip, work_root), diagnostics
    diagnostics["candidates"] = scan_input(input_root)
    accepted = [entry for entry in diagnostics["candidates"] if entry.get("accepted")]
    if len(accepted) == 0:
        raise FileNotFoundError(
            "No verified LegalIR source was found. Attach an unzipped source/bundle, or set LEGALIR_SOURCE_ROOT / LEGALIR_SOURCE_ZIP. "
            f"Diagnostics: {diagnostics}"
        )
    if len(accepted) > 1:
        raise RuntimeError(
            "Multiple verified LegalIR sources were found; set LEGALIR_SOURCE_ROOT to the intended one. "
            f"Candidates: {accepted}"
        )
    selected = accepted[0]
    if selected["kind"] == "directory":
        return Path(selected["path"]), diagnostics
    return safe_extract_source_zip(selected["path"], work_root), diagnostics
