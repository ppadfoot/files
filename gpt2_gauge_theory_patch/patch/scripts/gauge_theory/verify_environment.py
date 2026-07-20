#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(os.environ.get("GPT2_NANO_REPO", Path.cwd())).expanduser().resolve()
    required = [
        root / "gpt2nano" / "train_unified.py",
        root / "scripts" / "bootstrap_upstreams.sh",
        root / "scripts" / "check_openwebtext_bins.sh",
        root / "notebooks" / "gauge_covariant_theory" / "gauge_theory_utils.py",
        root / "third_party" / "Sophia" / "data" / "openwebtext" / "train.bin",
        root / "third_party" / "Sophia" / "data" / "openwebtext" / "val.bin",
    ]
    missing = [str(path) for path in required if not path.exists()]
    checkpoints = sorted((root / "runs").rglob("ckpt*.pt")) if (root / "runs").exists() else []
    zero = [str(path) for path in checkpoints if path.stat().st_size == 0]
    report = {
        "repo_root": str(root),
        "missing_required_files": missing,
        "checkpoint_count": len(checkpoints),
        "zero_size_checkpoints": zero,
        "dataset": {
            str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256(path)}
            for path in required[-2:]
            if path.exists()
        },
        "passed": not missing and not zero and len(checkpoints) > 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
