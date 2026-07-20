#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ALLOWED_SOPHIA_OVERWRITES = {
    "gpt2nano/train_unified.py",
    "scripts/bootstrap_upstreams.sh",
    "scripts/prepare_openwebtext.sh",
    "scripts/use_existing_openwebtext_bins.sh",
    "scripts/check_openwebtext_bins.sh",
    "scripts/audit_sophia_official.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def find_repo_root(root: Path) -> Path:
    if (root / "gpt2nano").is_dir():
        return root
    candidates = [path.parent for path in root.rglob("gpt2nano") if path.is_dir()]
    candidates = sorted(set(candidates), key=lambda p: len(p.parts))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one repository root, found: {candidates}")
    return candidates[0]


def inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def overlay(source: Path, destination: Path, *, allowed_overwrites: set[str] | None = None) -> list[str]:
    allowed_overwrites = allowed_overwrites or set()
    overwritten: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(source))
        target = destination / relative
        if target.exists():
            if relative not in allowed_overwrites:
                raise RuntimeError(f"Unexpected overwrite refused: {relative}")
            overwritten.append(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return overwritten


def zip_tree(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a clean v11 repository with the minimal Sophia patch and additive gauge notebooks")
    parser.add_argument("--base-zip", required=True, type=Path, help="Clean v11 zip or already-clean v24 full repository zip")
    parser.add_argument("--sophia-patch", type=Path, help="Optional research1_sophia_v11_minimal_patch_v24_replacement.zip")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-zip", type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    staging = args.output_dir.parent / (args.output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    extract(args.base_zip.resolve(), staging)
    repo = find_repo_root(staging)
    before = inventory(repo)

    sophia_overwrites: list[str] = []
    if args.sophia_patch:
        patch_tmp = staging / "__sophia_patch__"
        patch_tmp.mkdir()
        extract(args.sophia_patch.resolve(), patch_tmp)
        patch_root = find_repo_root(patch_tmp) if (patch_tmp / "gpt2nano").exists() or list(patch_tmp.rglob("gpt2nano")) else patch_tmp
        # Replacement archives normally contain files at the archive root.
        if not (patch_root / "gpt2nano").exists() and (patch_tmp / "scripts").exists():
            patch_root = patch_tmp
        sophia_overwrites = overlay(patch_root, repo, allowed_overwrites=ALLOWED_SOPHIA_OVERWRITES)
        shutil.rmtree(patch_tmp)

    # The installer lives inside the additive patch. The patch root is three
    # parents above this file: patch/scripts/gauge_theory/install_clean_repo.py.
    additive_root = Path(__file__).resolve().parents[2]
    overlay(additive_root, repo, allowed_overwrites=set())

    after = inventory(repo)
    missing_original = sorted(set(before) - set(after))
    if missing_original:
        raise RuntimeError(f"Original files disappeared: {missing_original}")

    final_parent = args.output_dir.parent
    final_parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(repo), str(args.output_dir))
    shutil.rmtree(staging, ignore_errors=True)

    report = {
        "base_zip": str(args.base_zip.resolve()),
        "base_zip_sha256": sha256(args.base_zip.resolve()),
        "sophia_patch": str(args.sophia_patch.resolve()) if args.sophia_patch else None,
        "sophia_patch_sha256": sha256(args.sophia_patch.resolve()) if args.sophia_patch else None,
        "sophia_overwrites": sophia_overwrites,
        "original_file_count": len(before),
        "final_file_count": len(after),
        "missing_original_files": missing_original,
    }
    (args.output_dir / "docs" / "CLEAN_REBUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_zip = args.output_zip or args.output_dir.with_suffix(".zip")
    if output_zip.exists():
        output_zip.unlink()
    zip_tree(args.output_dir, output_zip)
    print(json.dumps(report, indent=2))
    print("Final repository:", args.output_dir)
    print("Final zip:", output_zip)


if __name__ == "__main__":
    main()
