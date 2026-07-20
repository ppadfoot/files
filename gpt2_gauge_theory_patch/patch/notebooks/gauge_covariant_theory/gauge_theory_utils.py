from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_csv(name: str, default: Sequence[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def smoke_mode() -> bool:
    return env_bool("GAUGE_SMOKE_TEST", False)


def find_repo_root(start: Path | None = None) -> Path:
    explicit = os.environ.get("GPT2_NANO_REPO") or os.environ.get("REPO_ROOT")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    start = (start or Path.cwd()).resolve()
    candidates.extend([start, *start.parents])
    candidates.extend([
        Path.home() / "project" / "gpt2_nano_optimizer_pipeline2",
        Path.home() / "project" / "gpt2_nano_optimizer_pipeline",
    ])
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "gpt2nano").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    if smoke_mode():
        return Path.cwd().resolve()
    raise FileNotFoundError(
        "Repository root with gpt2nano/ and scripts/ was not found. "
        "Launch Jupyter from the repository or set GPT2_NANO_REPO."
    )


def output_dir(experiment: str, repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    out = root / "analysis_outputs" / "gauge_covariant_theory" / experiment
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save_figure(fig: Any, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")


def robust_scale(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.4826 * np.median(np.abs(x - np.median(x)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(x))
    return float(scale)


def empirical_ccdf(values: Sequence[float], n_grid: int = 160) -> tuple[np.ndarray, np.ndarray]:
    x = np.abs(np.asarray(values, dtype=float).reshape(-1))
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)
    lo = max(float(np.quantile(x, 0.02)), np.finfo(float).tiny)
    hi = float(np.quantile(x, min(0.999, 1 - 1 / max(x.size, 2))))
    if not hi > lo:
        hi = float(x.max())
    grid = np.geomspace(lo, max(hi, lo * (1 + 1e-6)), n_grid)
    ccdf = np.array([(x > threshold).mean() for threshold in grid])
    good = ccdf > 0
    return grid[good], ccdf[good]


def effective_slope(values: Sequence[float], n_grid: int = 120, smooth: int = 9) -> tuple[np.ndarray, np.ndarray]:
    x, ccdf = empirical_ccdf(values, n_grid=n_grid)
    if x.size < 7:
        return np.array([], dtype=float), np.array([], dtype=float)
    lx = np.log(x)
    ly = np.log(ccdf)
    slope = -np.gradient(ly, lx)
    if smooth > 1 and slope.size >= smooth:
        kernel = np.ones(smooth, dtype=float) / smooth
        slope = np.convolve(slope, kernel, mode="same")
    return x, slope


def log_shape_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.abs(np.asarray(a, dtype=float).reshape(-1))
    b = np.abs(np.asarray(b, dtype=float).reshape(-1))
    a = a[np.isfinite(a) & (a > 0)]
    b = b[np.isfinite(b) & (b > 0)]
    if a.size < 4 or b.size < 4:
        return math.nan
    la = np.log(a / np.median(a))
    lb = np.log(b / np.median(b))
    qs = np.linspace(0.02, 0.98, 97)
    return float(np.mean(np.abs(np.quantile(la, qs) - np.quantile(lb, qs))))


def participation_ratio(weights: Sequence[float]) -> float:
    w = np.abs(np.asarray(weights, dtype=float))
    denom = float(np.sum(w * w))
    if denom <= 0:
        return math.nan
    return float(np.sum(w) ** 2 / denom)


def parse_iteration(path: Path) -> int | None:
    values = re.findall(r"(?:iter|step|ckpt)[_-]?(\d+)", path.stem.lower())
    if values:
        return int(values[-1])
    if path.name == "ckpt_last.pt":
        return 10**18
    return None


# -----------------------------------------------------------------------------
# Manifest and probe bank
# -----------------------------------------------------------------------------


DEFAULT_RUN_DIRS: dict[str, str] = {
    "adamw": "runs/final_stage2/adamw_best",
    "lion": "runs/final_stage2/lion_best",
    "muon": "runs/final_stage2/muon_best",
    "sgd": "runs/final_stage2/sgd_best",
    "sophia_rho_0p05": "runs/research1_sophia_rho/sophia_official_rho0p05_lr6e-4_wd0p2_clip1",
    "sophia_rho_0p2": "runs/research1_sophia_rho/sophia_official_rho0p2_lr6e-4_wd0p2_clip1",
    "sophia_rho_0p8": "runs/research1_sophia_rho/sophia_official_rho0p8_lr6e-4_wd0p2_clip1",
    "sophia_rho_1p5": "runs/research1_sophia_rho/sophia_official_rho1p5_lr6e-4_wd0p2_clip1",
}


def checkpoint_candidates(run_dir: Path) -> list[Path]:
    patterns = ["ckpt_last.pt", "ckpt*.pt", "checkpoint*.pt"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(run_dir.glob(pattern))
    return sorted(set(path.resolve() for path in found if path.is_file()))


def choose_checkpoints(run_dir: Path, policy: str) -> list[Path]:
    candidates = checkpoint_candidates(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    numeric = [(parse_iteration(path), path) for path in candidates]
    numeric = [(it, path) for it, path in numeric if it is not None]
    numeric.sort(key=lambda pair: pair[0])
    if policy == "last":
        exact = run_dir / "ckpt_last.pt"
        return [exact.resolve()] if exact.exists() else [numeric[-1][1]]
    finite = [(it, path) for it, path in numeric if it < 10**18]
    if not finite:
        return [numeric[-1][1]]
    if policy == "mid":
        return [finite[len(finite) // 2][1]]
    if policy == "early_mid_late":
        ids = sorted(set([0, len(finite) // 2, len(finite) - 1]))
        return [finite[index][1] for index in ids]
    raise ValueError(f"Unknown checkpoint policy: {policy}")


def build_resolved_manifest(repo_root: Path, out: Path) -> dict[str, Any]:
    labels = env_csv("GAUGE_RUN_FILTER", list(DEFAULT_RUN_DIRS))
    policy = os.environ.get("GAUGE_CHECKPOINT_POLICY", "mid")
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for label in labels:
        relative = DEFAULT_RUN_DIRS.get(label, label)
        run_dir = (repo_root / relative).resolve()
        if not run_dir.is_dir():
            missing.append(label)
            continue
        for checkpoint in choose_checkpoints(run_dir, policy):
            entries.append({
                "label": label,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "iteration": parse_iteration(checkpoint),
            })
    if missing:
        raise FileNotFoundError(
            "Requested runs are missing: " + ", ".join(missing) + ". "
            "Set GAUGE_RUN_FILTER to available labels only or migrate the missing runs."
        )
    data_dir = repo_root / "third_party" / "Sophia" / "data" / "openwebtext"
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    for path in (train_bin, val_bin):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file is missing: {path}")
    payload = {
        "schema_version": 1,
        "repo_root": str(repo_root),
        "checkpoint_policy": policy,
        "entries": entries,
        "data": {
            "data_dir": str(data_dir),
            "train_bin": str(train_bin),
            "train_sha256": sha256_file(train_bin),
            "val_bin": str(val_bin),
            "val_sha256": sha256_file(val_bin),
        },
    }
    save_json(out / "resolved_manifest.json", payload)
    return payload


def make_probe_bank(manifest: Mapping[str, Any], out: Path) -> dict[str, Any]:
    train_path = Path(manifest["data"]["train_bin"])
    train = np.memmap(train_path, dtype=np.uint16, mode="r")
    block_size = env_int("GAUGE_BLOCK_SIZE", 256)
    n_sequences = env_int("GAUGE_PROBE_SEQUENCES", 64 if smoke_mode() else 4096)
    seed = env_int("GAUGE_SEED", 20260720)
    rng = np.random.default_rng(seed)
    # Sample from a block grid to avoid overlapping windows.
    max_block = max(1, (len(train) - block_size - 1) // (block_size + 1))
    if n_sequences > max_block:
        raise ValueError(f"Requested {n_sequences} non-overlapping windows, only {max_block} are available")
    block_ids = rng.choice(max_block, size=n_sequences, replace=False)
    starts = np.sort(block_ids * (block_size + 1)).astype(np.int64)
    pair_order = rng.permutation(n_sequences).astype(np.int64)
    random_seeds = rng.integers(0, 2**31 - 1, size=64, dtype=np.int64)
    np.savez_compressed(
        out / "probe_bank.npz",
        sequence_starts=starts,
        pair_order=pair_order,
        random_seeds=random_seeds,
        block_size=np.array([block_size], dtype=np.int64),
    )
    payload = {
        "seed": seed,
        "block_size": block_size,
        "n_sequences": n_sequences,
        "probe_bank_sha256": sha256_file(out / "probe_bank.npz"),
        "manifest_sha256": sha256_file(out / "resolved_manifest.json"),
    }
    save_json(out / "probe_bank.json", payload)
    return payload


def load_manifest_and_bank(repo_root: Path | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = repo_root or find_repo_root()
    base = root / "analysis_inputs" / "gauge_covariant_theory"
    manifest_path = base / "resolved_manifest.json"
    bank_path = base / "probe_bank.npz"
    if not manifest_path.exists() or not bank_path.exists():
        raise FileNotFoundError(
            "Run 00_manifest_and_probe_bank.ipynb first. "
            f"Expected {manifest_path} and {bank_path}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_npz = np.load(bank_path)
    bank = {key: bank_npz[key] for key in bank_npz.files}
    return manifest, bank


# -----------------------------------------------------------------------------
# GPT repository adapter (all torch imports are lazy, so CI smoke tests do not
# need torch).
# -----------------------------------------------------------------------------


@dataclass
class LoadedState:
    label: str
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    model: Any
    device: Any
    config: Any


def _import_torch() -> Any:
    import torch
    return torch


def import_gpt_classes(repo_root: Path) -> tuple[Any, Any]:
    third_party = repo_root / "third_party" / "Sophia"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    try:
        from model import GPT, GPTConfig
    except Exception as exc:
        raise ImportError(
            "Could not import third_party/Sophia/model.py. Run scripts/bootstrap_upstreams.sh."
        ) from exc
    return GPT, GPTConfig


def clean_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        while key.startswith("_orig_mod.") or key.startswith("module."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def checkpoint_model_args(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(checkpoint.get("model_args") or {})
    cfg = checkpoint.get("config") or checkpoint.get("cfg") or {}
    if hasattr(cfg, "__dict__"):
        cfg = vars(cfg)
    for key in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "dropout", "scale_attn_by_inverse_layer_idx"]:
        if key not in source and key in cfg:
            source[key] = cfg[key]
    defaults = {
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "block_size": 256,
        "bias": False,
        "vocab_size": 50304,
        "dropout": 0.0,
        "scale_attn_by_inverse_layer_idx": False,
    }
    for key, value in defaults.items():
        source.setdefault(key, value)
    return source


def load_state(entry: Mapping[str, Any], repo_root: Path, device: str | None = None) -> LoadedState:
    torch = _import_torch()
    device_name = device or os.environ.get("GAUGE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)
    checkpoint_path = Path(entry["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    GPT, GPTConfig = import_gpt_classes(repo_root)
    args = checkpoint_model_args(checkpoint)
    signature = inspect.signature(GPTConfig)
    accepted = {key: value for key, value in args.items() if key in signature.parameters}
    config = GPTConfig(**accepted)
    model = GPT(config)
    state = checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise KeyError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.to(dev)
    model.eval()
    return LoadedState(str(entry["label"]), checkpoint_path, checkpoint, model, dev, config)


def memmap_tokens(data_dir: Path, split: str = "train") -> np.memmap:
    return np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")


def batch_from_starts(data: np.memmap, starts: Sequence[int], block_size: int, device: Any) -> tuple[Any, Any]:
    torch = _import_torch()
    x = np.stack([np.asarray(data[int(start): int(start) + block_size], dtype=np.int64) for start in starts])
    y = np.stack([np.asarray(data[int(start) + 1: int(start) + 1 + block_size], dtype=np.int64) for start in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def get_module(model: Any, path: str) -> Any:
    node = model
    for part in path.split("."):
        if part.isdigit():
            node = node[int(part)]
        else:
            node = getattr(node, part)
    return node


def named_parameter(model: Any, name: str) -> Any:
    params = dict(model.named_parameters())
    if name not in params:
        raise KeyError(f"Parameter not found: {name}")
    return params[name]


def _head_slice(head: int, head_dim: int) -> slice:
    return slice(head * head_dim, (head + 1) * head_dim)


def attention_scale_map(model: Any, block_index: int, intervention: str, head_scales: Sequence[float]) -> dict[str, Any]:
    torch = _import_torch()
    block = get_module(model, f"transformer.h.{block_index}")
    attn = block.attn
    n_head = int(getattr(model.config, "n_head"))
    n_embd = int(getattr(model.config, "n_embd"))
    if len(head_scales) != n_head:
        raise ValueError(f"Expected {n_head} head scales, got {len(head_scales)}")
    head_dim = n_embd // n_head
    scales = torch.as_tensor(head_scales, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    result: dict[str, Any] = {}

    def ones_like_parameter(name: str) -> Any:
        return torch.ones_like(named_parameter(model, name))

    # Separate q/k/v modules, when a patched model exposes them.
    separate = all(hasattr(attn, name) for name in ("q", "k", "v"))
    if separate:
        if intervention == "qk":
            for module_name, reciprocal in [("q", False), ("k", True)]:
                pname = f"transformer.h.{block_index}.attn.{module_name}.weight"
                param = named_parameter(model, pname)
                factor = ones_like_parameter(pname)
                module = getattr(attn, module_name)
                for h in range(n_head):
                    value = (1.0 / scales[h]) if reciprocal else scales[h]
                    if isinstance(module, torch.nn.Linear):
                        factor[_head_slice(h, head_dim), :] = value
                    else:
                        factor[:, _head_slice(h, head_dim)] = value
                result[pname] = factor
        elif intervention == "v_wo":
            vname = f"transformer.h.{block_index}.attn.v.weight"
            vfactor = ones_like_parameter(vname)
            vmodule = attn.v
            for h in range(n_head):
                if isinstance(vmodule, torch.nn.Linear):
                    vfactor[_head_slice(h, head_dim), :] = scales[h]
                else:
                    vfactor[:, _head_slice(h, head_dim)] = scales[h]
            result[vname] = vfactor
        else:
            raise ValueError(intervention)
    else:
        pname = f"transformer.h.{block_index}.attn.c_attn.weight"
        param = named_parameter(model, pname)
        factor = ones_like_parameter(pname)
        module = attn.c_attn
        output_first = isinstance(module, torch.nn.Linear) or param.shape[0] == 3 * n_embd
        offsets = {"q": 0, "k": n_embd, "v": 2 * n_embd}
        selected = [("q", False), ("k", True)] if intervention == "qk" else [("v", False)]
        for component, reciprocal in selected:
            offset = offsets[component]
            for h in range(n_head):
                value = (1.0 / scales[h]) if reciprocal else scales[h]
                hs = _head_slice(h, head_dim)
                if output_first:
                    factor[slice(offset + hs.start, offset + hs.stop), :] = value
                else:
                    factor[:, slice(offset + hs.start, offset + hs.stop)] = value
        result[pname] = factor

    if intervention == "v_wo":
        pname = f"transformer.h.{block_index}.attn.c_proj.weight"
        factor = ones_like_parameter(pname)
        module = attn.c_proj
        for h in range(n_head):
            value = 1.0 / scales[h]
            hs = _head_slice(h, head_dim)
            if isinstance(module, torch.nn.Linear):
                factor[:, hs] = value
            else:
                factor[hs, :] = value
        result[pname] = factor
    return result


def apply_scale_map(model: Any, scale_map: Mapping[str, Any]) -> None:
    torch = _import_torch()
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, factor in scale_map.items():
            params[name].mul_(factor)


def model_logits(model: Any, x: Any) -> Any:
    result = model(x, None)
    logits = result[0] if isinstance(result, tuple) else result
    return logits


def batch_gradient(model: Any, x: Any, y: Any, parameter_names: Sequence[str]) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    result = model(x, y)
    loss = result[1] if isinstance(result, tuple) else result
    if loss is None:
        raise RuntimeError("Model returned no loss")
    loss.backward()
    params = dict(model.named_parameters())
    gradients: dict[str, Any] = {}
    for name in parameter_names:
        grad = params[name].grad
        if grad is None:
            raise RuntimeError(f"No gradient for {name}")
        gradients[name] = grad.detach().clone()
    return gradients


def pair_specifications(bank: Mapping[str, np.ndarray], n_pairs: int, batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.asarray(bank["pair_order"], dtype=np.int64)
    needed = 2 * n_pairs * batch_size
    if needed > order.size:
        raise ValueError(f"Probe bank has {order.size} sequences, need {needed}")
    specs = []
    cursor = 0
    for _ in range(n_pairs):
        a = order[cursor: cursor + batch_size]
        cursor += batch_size
        b = order[cursor: cursor + batch_size]
        cursor += batch_size
        specs.append((a, b))
    return specs


def pairwise_noise(model: Any, data: np.memmap, starts: np.ndarray, specs: Sequence[tuple[np.ndarray, np.ndarray]], block_size: int, names: Sequence[str]) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for first_ids, second_ids in specs:
        x1, y1 = batch_from_starts(data, starts[first_ids], block_size, next(model.parameters()).device)
        x2, y2 = batch_from_starts(data, starts[second_ids], block_size, next(model.parameters()).device)
        g1 = batch_gradient(model, x1, y1, names)
        g2 = batch_gradient(model, x2, y2, names)
        for name in names:
            result[name].append(((g1[name] - g2[name]) / math.sqrt(2.0)).detach().cpu().numpy())
    return result


def geometric_head_scales(n_head: int, span: float = 4.0) -> np.ndarray:
    log_values = np.linspace(-math.log(span), math.log(span), n_head)
    log_values -= log_values.mean()
    return np.exp(log_values)


# -----------------------------------------------------------------------------
# Optimizer adapter
# -----------------------------------------------------------------------------


def checkpoint_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    cfg = checkpoint.get("config") or checkpoint.get("cfg") or {}
    if hasattr(cfg, "__dict__"):
        cfg = vars(cfg)
    return dict(cfg)


def optimizer_name_from_entry(entry: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> str:
    cfg = checkpoint_config(checkpoint)
    value = cfg.get("optimizer") or cfg.get("optimizer_name") or checkpoint.get("optimizer_name")
    if value:
        return str(value).lower()
    label = str(entry["label"]).lower()
    if label.startswith("sophia"):
        return "sophia_official"
    return label.split("_")[0]


def _parameter_groups(model: Any, optimizer_name: str, weight_decay: float) -> list[dict[str, Any]]:
    matrix = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    vector = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    if optimizer_name == "muon":
        return [
            {"params": matrix, "weight_decay": weight_decay},
            {"params": vector, "weight_decay": weight_decay},
        ]
    return [
        {"params": matrix, "weight_decay": weight_decay},
        {"params": vector, "weight_decay": 0.0},
    ]


def instantiate_optimizer(model: Any, entry: Mapping[str, Any], checkpoint: Mapping[str, Any], *, override_name: str | None = None, fresh: bool = False) -> Any:
    torch = _import_torch()
    cfg = checkpoint_config(checkpoint)
    name = (override_name or optimizer_name_from_entry(entry, checkpoint)).lower()
    lr = float(cfg.get("lr", cfg.get("learning_rate", 6e-4)))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    beta1 = float(cfg.get("sophia_beta1", cfg.get("beta1", 0.9)))
    beta2 = float(cfg.get("sophia_beta2", cfg.get("beta2", 0.99)))
    rho = float(cfg.get("sophia_rho", cfg.get("rho", 0.05)))
    groups = _parameter_groups(model, name, weight_decay)

    if name in {"adam", "adamw"}:
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name in {"sgd", "sgdm", "momentum"}:
        optimizer = torch.optim.SGD(groups, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=weight_decay)
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(groups, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        except Exception as exc:
            raise ImportError("Lion requires lion-pytorch or the repository optimizer factory") from exc
    elif name.startswith("sophia"):
        repo_root = find_repo_root()
        third_party = repo_root / "third_party" / "Sophia"
        if str(third_party) not in sys.path:
            sys.path.insert(0, str(third_party))
        from sophia import SophiaG
        optimizer = SophiaG(groups, lr=lr, betas=(beta1, beta2), rho=rho, weight_decay=weight_decay)
    elif name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is unavailable in this PyTorch build")
        optimizer = torch.optim.Muon(groups, lr=lr, weight_decay=weight_decay, momentum=float(cfg.get("momentum", 0.95)))
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    if not fresh:
        state = checkpoint.get("optimizer") or checkpoint.get("optimizer_state_dict")
        if state is None:
            raise KeyError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(state)
    return optimizer


def transform_optimizer_state(optimizer: Any, model: Any, scale_map: Mapping[str, Any]) -> None:
    params_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    first_keys = {"exp_avg", "momentum_buffer", "momentum", "grad_avg"}
    second_keys = {"exp_avg_sq", "hessian", "square_avg", "variance"}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = params_by_id.get(id(parameter))
            if name is None or name not in scale_map:
                continue
            factor = scale_map[name]
            state = optimizer.state.get(parameter, {})
            for key, value in list(state.items()):
                if not hasattr(value, "shape") or tuple(value.shape) != tuple(parameter.shape):
                    continue
                if key in first_keys:
                    value.div_(factor)
                elif key in second_keys:
                    value.div_(factor * factor)


def optimizer_step(optimizer: Any, checkpoint: Mapping[str, Any], model: Any) -> None:
    name = optimizer.__class__.__name__.lower()
    cfg = checkpoint_config(checkpoint)
    if "sophia" in name:
        batch_size = int(cfg.get("batch_size", 8))
        grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
        block_size = int(cfg.get("block_size", getattr(model.config, "block_size", 256)))
        tokens = int(cfg.get("tokens_per_update", batch_size * grad_accum * block_size))
        optimizer.step(bs=tokens)
    else:
        optimizer.step()


def maybe_global_clip(model: Any, checkpoint: Mapping[str, Any]) -> tuple[float, float]:
    torch = _import_torch()
    cfg = checkpoint_config(checkpoint)
    threshold = float(cfg.get("grad_clip", 0.0) or 0.0)
    total = torch.linalg.vector_norm(torch.stack([parameter.grad.detach().norm() for parameter in model.parameters() if parameter.grad is not None])).item()
    if threshold > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)
    return float(total), threshold


def virtual_step_defect(entry: Mapping[str, Any], repo_root: Path, bank: Mapping[str, np.ndarray], intervention: str, block_index: int, *, rho_override: float | None = None, fresh_optimizer: str | None = None) -> dict[str, float]:
    torch = _import_torch()
    state = load_state(entry, repo_root)
    original = state.model
    transformed = copy.deepcopy(original)
    scales = geometric_head_scales(int(state.config.n_head), env_float("GAUGE_HEAD_SCALE_SPAN", 4.0))
    scale_map = attention_scale_map(transformed, block_index, intervention, scales)
    apply_scale_map(transformed, scale_map)

    opt_original = instantiate_optimizer(original, entry, state.checkpoint, override_name=fresh_optimizer, fresh=fresh_optimizer is not None)
    opt_transformed = instantiate_optimizer(transformed, entry, state.checkpoint, override_name=fresh_optimizer, fresh=fresh_optimizer is not None)
    transform_optimizer_state(opt_transformed, transformed, scale_map)
    if rho_override is not None:
        for optimizer in (opt_original, opt_transformed):
            for group in optimizer.param_groups:
                if "rho" in group:
                    group["rho"] = float(rho_override)

    starts = np.asarray(bank["sequence_starts"], dtype=np.int64)
    block_size = int(bank["block_size"][0])
    data_dir = Path(json.loads((repo_root / "analysis_inputs" / "gauge_covariant_theory" / "resolved_manifest.json").read_text())["data"]["data_dir"])
    data = memmap_tokens(data_dir)
    batch_n = env_int("GAUGE_VIRTUAL_BATCH", 4 if smoke_mode() else 8)
    train_ids = np.asarray(bank["pair_order"][:batch_n], dtype=np.int64)
    probe_ids = np.asarray(bank["pair_order"][batch_n: 2 * batch_n], dtype=np.int64)
    x, y = batch_from_starts(data, starts[train_ids], block_size, state.device)
    xp, _ = batch_from_starts(data, starts[probe_ids], block_size, state.device)

    with torch.no_grad():
        pre_o = model_logits(original, xp).detach()
        pre_t = model_logits(transformed, xp).detach()
    if torch.max(torch.abs(pre_o - pre_t)).item() > env_float("GAUGE_LOGIT_TOL", 5e-5):
        raise RuntimeError("Function-preserving transformation failed before optimizer step")

    for model, optimizer in [(original, opt_original), (transformed, opt_transformed)]:
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        maybe_global_clip(model, state.checkpoint)
        optimizer_step(optimizer, state.checkpoint, model)
        optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        post_o = model_logits(original, xp).detach()
        post_t = model_logits(transformed, xp).detach()
    delta_o = post_o - pre_o
    delta_t = post_t - pre_t
    denom = float(torch.mean(delta_o * delta_o).item()) + 1e-30
    defect = float(torch.mean((delta_t - delta_o) ** 2).item() / denom)
    return {
        "defect": defect,
        "functional_rms": float(torch.sqrt(torch.mean(delta_o * delta_o)).item()),
        "pre_logit_max_diff": float(torch.max(torch.abs(pre_o - pre_t)).item()),
    }


# -----------------------------------------------------------------------------
# Forward/backward factorization helpers
# -----------------------------------------------------------------------------


def canonical_spectrum(h: np.ndarray, e: np.ndarray, rcond: float = 1e-7) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    e = np.asarray(e, dtype=float)
    n = max(1, h.shape[0])
    sh = h.T @ h / n
    se = e.T @ e / n
    g = e.T @ h / n
    operator = np.linalg.pinv(se, rcond=rcond) @ g @ np.linalg.pinv(sh, rcond=rcond) @ g.T
    eig = np.linalg.eigvals(operator)
    eig = np.real(eig[np.isfinite(eig)])
    return np.sort(np.clip(eig, 0.0, None))[::-1]


def hook_activation_error(model: Any, module_path: str, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    module = get_module(model, module_path)
    payload: dict[str, Any] = {}

    def forward_hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        payload["h"] = inputs[0].detach()
        if hasattr(output, "register_hook"):
            output.register_hook(lambda grad: payload.__setitem__("e", grad.detach()))

    handle = module.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    handle.remove()
    h = payload["h"].reshape(-1, payload["h"].shape[-1]).cpu().numpy()
    e = payload["e"].reshape(-1, payload["e"].shape[-1]).cpu().numpy()
    return h, e


# -----------------------------------------------------------------------------
# Smoke-mode experiments. These are exact, deterministic integration tests of
# the notebooks and mathematical plumbing. They do not substitute for real GPT
# experiments.
# -----------------------------------------------------------------------------


def _smoke_rng() -> np.random.Generator:
    return np.random.default_rng(20260720)


def smoke_covector() -> pd.DataFrame:
    rng = _smoke_rng()
    rows = []
    for intervention in ["v_wo", "qk"]:
        scale = np.exp(rng.normal(0, 0.7, size=512))
        grad = rng.normal(size=512)
        grad_prime = grad / scale
        pullback = scale * grad_prime
        rows.append({"intervention": intervention, "relative_l2_error": np.linalg.norm(pullback-grad)/np.linalg.norm(grad)})
    return pd.DataFrame(rows)


def smoke_mixture() -> pd.DataFrame:
    rng = _smoke_rng()
    heads = [rng.standard_t(df=4+i, size=2000) for i in range(4)]
    scales = np.array([0.5, 0.8, 1.4, 2.2])
    predicted = np.concatenate([head / scale for head, scale in zip(heads, scales)])
    measured = predicted.copy()
    null = np.concatenate(heads)
    return pd.DataFrame([{
        "predicted_measured_distance": log_shape_distance(predicted, measured),
        "original_transformed_distance": log_shape_distance(null, measured),
    }])


def smoke_optimizer_defect() -> pd.DataFrame:
    rng = _smoke_rng()
    scale = np.exp(rng.normal(0, 0.6, size=512))
    grad = rng.normal(size=512)
    hess = np.exp(rng.normal(0, 0.4, size=512))
    sgd_u = grad
    sgd_up = grad / scale
    sgd_pull = sgd_up / scale  # tangent pullback from theta'=scale*theta
    sophia_u = grad / hess
    sophia_up = (grad/scale) / (hess/(scale*scale))
    sophia_pull = sophia_up / scale
    return pd.DataFrame([
        {"optimizer": "sgd", "defect": np.mean((sgd_pull-sgd_u)**2)/np.mean(sgd_u**2)},
        {"optimizer": "sophia_unsaturated", "defect": np.mean((sophia_pull-sophia_u)**2)/np.mean(sophia_u**2)},
    ])


def smoke_sophia_saturation() -> pd.DataFrame:
    rng = _smoke_rng()
    grad = rng.normal(size=10000)
    hess = np.exp(rng.normal(0, 0.8, size=10000))
    scale = np.exp(rng.normal(0, 0.5, size=10000))
    rows = []
    for rho in [0.02, 0.05, 0.1, 0.2, 0.8, 1.5, 3.0]:
        raw = np.abs(grad) / (rho * hess + 1e-12)
        update = np.sign(grad) * np.minimum(raw, 1.0)
        grad_p = grad / scale
        hess_p = hess / (scale*scale)
        raw_p = np.abs(grad_p) / (rho*hess_p + 1e-12)
        update_p = np.sign(grad_p) * np.minimum(raw_p, 1.0)
        pulled = update_p / scale
        defect = np.mean((pulled-update)**2)/(np.mean(update**2)+1e-30)
        rows.append({"rho":rho,"saturation_rate":float(np.mean(raw>=1)),"defect":float(defect)})
    return pd.DataFrame(rows)


def smoke_continuation() -> pd.DataFrame:
    rng = _smoke_rng()
    defect = 1e-4
    rows=[]
    value=0.0
    for step in range(21):
        value += defect*(1+0.1*rng.normal())
        rows.append({"step":step,"functional_distance":abs(value)})
    return pd.DataFrame(rows)


def smoke_spectrum() -> pd.DataFrame:
    rng = _smoke_rng()
    h = rng.normal(size=(2000, 8))
    e = rng.normal(size=(2000, 6))
    a = rng.normal(size=(8,8)); a += 3*np.eye(8)
    b = rng.normal(size=(6,6)); b += 3*np.eye(6)
    s0 = canonical_spectrum(h,e)
    s1 = canonical_spectrum(h@a, e@b)
    n=min(len(s0),len(s1))
    return pd.DataFrame({"index":np.arange(n),"original":s0[:n],"transformed":s1[:n],"abs_diff":np.abs(s0[:n]-s1[:n])})


def smoke_token_sequence() -> pd.DataFrame:
    rng = _smoke_rng()
    rows=[]
    for batch in range(32):
        token = rng.lognormal(0,0.5,size=(4,32))
        rows.append({
            "batch":batch,
            "token_neff_median":float(np.median([participation_ratio(row) for row in token])),
            "sequence_neff":participation_ratio(token.sum(axis=1)),
            "max_token_share":float(np.max(token/token.sum(axis=1,keepdims=True))),
        })
    return pd.DataFrame(rows)


def smoke_batch_flow() -> pd.DataFrame:
    rng = _smoke_rng()
    base = rng.normal(size=8192)
    rows=[]
    for b in [1,2,4,8,16,32,64]:
        values=base[:(len(base)//b)*b].reshape(-1,b).mean(axis=1)
        rows.append({"batch_size":b,"robust_scale":robust_scale(values)})
    frame=pd.DataFrame(rows)
    slope=np.polyfit(np.log(frame.batch_size),np.log(frame.robust_scale),1)[0]
    frame["fitted_gamma"]=-slope
    return frame


def smoke_algorithm_recipe() -> pd.DataFrame:
    return pd.DataFrame([
        {"comparison":"algorithm_only","candidate":"sgd","functional_gain":1.0,"gauge_defect":0.4},
        {"comparison":"algorithm_only","candidate":"sophia","functional_gain":1.1,"gauge_defect":0.02},
        {"comparison":"full_recipe","candidate":"sgd","functional_gain":0.8,"gauge_defect":0.35},
        {"comparison":"full_recipe","candidate":"sophia","functional_gain":1.2,"gauge_defect":0.08},
    ])


# -----------------------------------------------------------------------------
# Public experiment runners used by notebooks
# -----------------------------------------------------------------------------


def run_00() -> dict[str, Any]:
    root = find_repo_root()
    out = root / "analysis_inputs" / "gauge_covariant_theory"
    out.mkdir(parents=True, exist_ok=True)
    if smoke_mode():
        manifest = {"schema_version":1,"repo_root":str(root),"entries":[{"label":"smoke","checkpoint":"synthetic"}],"data":{"train_bin":"synthetic","val_bin":"synthetic"}}
        save_json(out/"resolved_manifest.json",manifest)
        np.savez_compressed(out/"probe_bank.npz",sequence_starts=np.arange(128),pair_order=np.arange(128),random_seeds=np.arange(64),block_size=np.array([16]))
        probe={"seed":20260720,"n_sequences":128,"smoke":True}
        save_json(out/"probe_bank.json",probe)
        return {"manifest":manifest,"probe":probe}
    manifest = build_resolved_manifest(root,out)
    probe = make_probe_bank(manifest,out)
    return {"manifest":manifest,"probe":probe}


def run_01() -> pd.DataFrame:
    out=output_dir("01_covector_transport")
    if smoke_mode():
        frame=smoke_covector(); frame.to_csv(out/"covector_audit.csv",index=False); return frame
    root=find_repo_root(); manifest,bank=load_manifest_and_bank(root)
    block=env_int("GAUGE_ATTN_BLOCK",3); n_pairs=env_int("GAUGE_COVECTOR_PAIRS",64); batch=env_int("GAUGE_PAIR_BATCH",8)
    data=memmap_tokens(Path(manifest["data"]["data_dir"])); starts=np.asarray(bank["sequence_starts"]); specs=pair_specifications(bank,n_pairs,batch); bs=int(bank["block_size"][0])
    rows=[]
    for entry in manifest["entries"]:
        state=load_state(entry,root); scales=geometric_head_scales(int(state.config.n_head),env_float("GAUGE_HEAD_SCALE_SPAN",4.0))
        for intervention in ["v_wo","qk"]:
            transformed=copy.deepcopy(state.model); smap=attention_scale_map(transformed,block,intervention,scales); apply_scale_map(transformed,smap); names=list(smap)
            for pair_index,(a_ids,b_ids) in enumerate(specs):
                for side,ids in [("a",a_ids),("b",b_ids)]:
                    x,y=batch_from_starts(data,starts[ids],bs,state.device)
                    g=batch_gradient(state.model,x,y,names); gp=batch_gradient(transformed,x,y,names)
                    for name in names:
                        pull=smap[name]*gp[name]; rel=float((pull-g[name]).norm().item()/(g[name].norm().item()+1e-30))
                        rows.append({"state_key":state.label,"intervention":intervention,"pair":pair_index,"side":side,"parameter":name,"relative_l2_error":rel})
            del transformed,state.model
    frame=pd.DataFrame(rows); frame.to_csv(out/"covector_audit.csv",index=False); save_json(out/"decision.json",{"median":float(frame.relative_l2_error.median()),"q99":float(frame.relative_l2_error.quantile(.99))}); return frame


def run_02() -> pd.DataFrame:
    out=output_dir("02_pooled_mixture_prediction")
    if smoke_mode():
        frame=smoke_mixture(); frame.to_csv(out/"mixture_prediction.csv",index=False); return frame
    root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); block=env_int("GAUGE_ATTN_BLOCK",3); n_pairs=env_int("GAUGE_MIXTURE_PAIRS",64); batch=env_int("GAUGE_PAIR_BATCH",8)
    data=memmap_tokens(Path(manifest["data"]["data_dir"])); starts=np.asarray(bank["sequence_starts"]); specs=pair_specifications(bank,n_pairs,batch); bs=int(bank["block_size"][0]); rows=[]
    for entry in manifest["entries"]:
        state=load_state(entry,root); scales=geometric_head_scales(int(state.config.n_head),env_float("GAUGE_HEAD_SCALE_SPAN",4.0))
        for intervention in ["v_wo","qk"]:
            transformed=copy.deepcopy(state.model); smap=attention_scale_map(transformed,block,intervention,scales); apply_scale_map(transformed,smap); names=list(smap)
            orig=pairwise_noise(state.model,data,starts,specs,bs,names); meas=pairwise_noise(transformed,data,starts,specs,bs,names)
            for name in names:
                original=np.concatenate([x.reshape(-1) for x in orig[name]]); measured=np.concatenate([x.reshape(-1) for x in meas[name]])
                factor=smap[name].detach().cpu().numpy(); predicted=np.concatenate([(x/factor).reshape(-1) for x in orig[name]])
                half=max(1,len(orig[name])//2); null_a=np.concatenate([x.reshape(-1) for x in orig[name][:half]]); null_b=np.concatenate([x.reshape(-1) for x in orig[name][half:]])
                rows.append({"state_key":state.label,"intervention":intervention,"parameter":name,"predicted_measured_distance":log_shape_distance(predicted,measured),"null_distance":log_shape_distance(null_a,null_b),"original_measured_distance":log_shape_distance(original,measured)})
            del transformed,state.model
    frame=pd.DataFrame(rows); frame.to_csv(out/"mixture_prediction.csv",index=False); return frame


def run_03() -> pd.DataFrame:
    out=output_dir("03_optimizer_gauge_equivariance")
    if smoke_mode():
        frame=smoke_optimizer_defect(); frame.to_csv(out/"optimizer_defect.csv",index=False); return frame
    root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); rows=[]
    for entry in manifest["entries"]:
        for intervention in ["v_wo","qk"]:
            try:
                rec=virtual_step_defect(entry,root,bank,intervention,env_int("GAUGE_ATTN_BLOCK",3)); rec.update(state_key=entry["label"],intervention=intervention,status="ok")
            except Exception as exc:
                rec={"state_key":entry["label"],"intervention":intervention,"defect":math.nan,"functional_rms":math.nan,"status":repr(exc)}
            rows.append(rec)
    frame=pd.DataFrame(rows); frame.to_csv(out/"optimizer_defect.csv",index=False); return frame


def run_04() -> pd.DataFrame:
    out=output_dir("04_sophia_saturation_mediation")
    if smoke_mode():
        frame=smoke_sophia_saturation(); frame.to_csv(out/"sophia_saturation.csv",index=False); return frame
    root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); rhos=[float(x) for x in env_csv("GAUGE_RHO_SWEEP",["0.02","0.05","0.1","0.2","0.4","0.8","1.5","3.0"])]; rows=[]
    entries=[entry for entry in manifest["entries"] if str(entry["label"]).startswith("sophia")]
    if not entries: raise RuntimeError("No Sophia entries in resolved manifest")
    for entry in entries:
        for rho in rhos:
            try:
                rec=virtual_step_defect(entry,root,bank,"v_wo",env_int("GAUGE_ATTN_BLOCK",3),rho_override=rho)
                state=load_state(entry,root); optimizer=instantiate_optimizer(state.model,entry,state.checkpoint)
                ratios=[]
                cfg=checkpoint_config(state.checkpoint); bs_tokens=float(cfg.get("tokens_per_update",cfg.get("batch_size",8)*cfg.get("gradient_accumulation_steps",1)*cfg.get("block_size",state.config.block_size)))
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        st=optimizer.state.get(p,{})
                        if "exp_avg" in st and "hessian" in st:
                            ratio=(st["exp_avg"].abs()/(rho*bs_tokens*st["hessian"]+1e-15)).detach().cpu().numpy().reshape(-1); ratios.append(ratio)
                all_ratio=np.concatenate(ratios) if ratios else np.array([])
                rec.update(state_key=entry["label"],rho=rho,saturation_rate=float(np.mean(all_ratio>=1)) if all_ratio.size else math.nan,win_rate=float(np.mean(all_ratio<1)) if all_ratio.size else math.nan,status="ok")
            except Exception as exc:
                rec={"state_key":entry["label"],"rho":rho,"defect":math.nan,"saturation_rate":math.nan,"win_rate":math.nan,"status":repr(exc)}
            rows.append(rec)
    frame=pd.DataFrame(rows); frame.to_csv(out/"sophia_saturation.csv",index=False); return frame


def run_05() -> pd.DataFrame:
    out=output_dir("05_gauge_continuation")
    if smoke_mode():
        frame=smoke_continuation(); frame.to_csv(out/"continuation.csv",index=False); return frame
    # Full continuation is intentionally separated from one-step experiments. It
    # requires checkpoint optimizer state, a transformed state, and a shared
    # future-batch stream. The current implementation uses repeated one-step
    # defects as a safe pilot and records the exact requested horizon.
    root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); steps=env_int("GAUGE_CONTINUATION_STEPS",50); rows=[]
    for entry in manifest["entries"]:
        rec=virtual_step_defect(entry,root,bank,"v_wo",env_int("GAUGE_ATTN_BLOCK",3));
        for step in range(steps+1): rows.append({"state_key":entry["label"],"step":step,"predicted_accumulated_defect":step*rec["defect"],"one_step_defect":rec["defect"],"status":"pilot_linearized"})
    frame=pd.DataFrame(rows); frame.to_csv(out/"continuation.csv",index=False); return frame


def run_06() -> pd.DataFrame:
    out=output_dir("06_invariant_forward_backward_spectrum")
    if smoke_mode():
        frame=smoke_spectrum(); frame.to_csv(out/"canonical_spectrum.csv",index=False); return frame
    root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); modules=env_csv("GAUGE_SPECTRUM_MODULES",["transformer.h.3.attn.c_proj","transformer.h.3.mlp.c_fc"]); n_seq=env_int("GAUGE_SPECTRUM_SEQUENCES",64); starts=np.asarray(bank["sequence_starts"]); bs=int(bank["block_size"][0]); data=memmap_tokens(Path(manifest["data"]["data_dir"])); rows=[]
    for entry in manifest["entries"]:
        state=load_state(entry,root); ids=np.asarray(bank["pair_order"][:n_seq]); x,y=batch_from_starts(data,starts[ids],bs,state.device)
        for module_path in modules:
            h,e=hook_activation_error(state.model,module_path,x,y); spectrum=canonical_spectrum(h,e)
            for index,value in enumerate(spectrum[:env_int("GAUGE_SPECTRUM_TOPK",32)]): rows.append({"state_key":state.label,"module":module_path,"index":index,"eigenvalue":float(value)})
    frame=pd.DataFrame(rows); frame.to_csv(out/"canonical_spectrum.csv",index=False); return frame


def run_07() -> pd.DataFrame:
    out=output_dir("07_token_sequence_batch_decomposition")
    if smoke_mode():
        frame=smoke_token_sequence(); frame.to_csv(out/"participation.csv",index=False); return frame
    # Token-level autograd is expensive; use explicit limits and a direction
    # estimated on an independent sequence split.
    torch=_import_torch(); root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); layer=os.environ.get("GAUGE_TOKEN_LAYER","transformer.h.3.attn.c_proj.weight"); n_seq=env_int("GAUGE_TOKEN_SEQUENCES",16); n_tokens=env_int("GAUGE_TOKENS_PER_SEQUENCE",32); starts=np.asarray(bank["sequence_starts"]); bs=int(bank["block_size"][0]); data=memmap_tokens(Path(manifest["data"]["data_dir"])); rows=[]
    for entry in manifest["entries"]:
        state=load_state(entry,root); param=named_parameter(state.model,layer); direction_ids=np.asarray(bank["pair_order"][:4]); xd,yd=batch_from_starts(data,starts[direction_ids],bs,state.device); direction=batch_gradient(state.model,xd,yd,[layer])[layer].detach(); direction=direction/(direction.norm()+1e-30)
        eval_ids=np.asarray(bank["pair_order"][4:4+n_seq])
        for seq_id in eval_ids:
            x,y=batch_from_starts(data,[starts[seq_id]],bs,state.device); state.model.zero_grad(set_to_none=True); logits,_=state.model(x,y); losses=torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1),reduction="none").view(1,-1)
            positions=np.linspace(0,losses.shape[1]-1,min(n_tokens,losses.shape[1]),dtype=int); token_proj=[]
            for pos in positions:
                grad=torch.autograd.grad(losses[0,pos],param,retain_graph=True)[0]; token_proj.append(float(torch.sum(grad*direction).item()))
            weights=np.abs(np.asarray(token_proj)); rows.append({"state_key":state.label,"sequence_start":int(starts[seq_id]),"token_neff":participation_ratio(weights),"max_token_share":float(weights.max()/(weights.sum()+1e-30)),"n_tokens":len(weights)})
    frame=pd.DataFrame(rows); frame.to_csv(out/"participation.csv",index=False); return frame


def run_08() -> pd.DataFrame:
    out=output_dir("08_batch_aggregation_flow")
    if smoke_mode():
        frame=smoke_batch_flow(); frame.to_csv(out/"batch_flow.csv",index=False); return frame
    torch=_import_torch(); root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); layer=os.environ.get("GAUGE_BATCH_LAYER","transformer.h.3.attn.c_proj.weight"); sizes=[int(x) for x in env_csv("GAUGE_BATCH_SIZES",["1","2","4","8","16","32","64"])]; starts=np.asarray(bank["sequence_starts"]); block=int(bank["block_size"][0]); data=memmap_tokens(Path(manifest["data"]["data_dir"])); rows=[]
    for entry in manifest["entries"]:
        state=load_state(entry,root); direction_ids=np.asarray(bank["pair_order"][:8]); x,y=batch_from_starts(data,starts[direction_ids],block,state.device); direction=batch_gradient(state.model,x,y,[layer])[layer].detach(); direction=direction/(direction.norm()+1e-30)
        pool_ids=np.asarray(bank["pair_order"][8:]); projections=[]
        for seq_id in pool_ids:
            xs,ys=batch_from_starts(data,[starts[seq_id]],block,state.device); grad=batch_gradient(state.model,xs,ys,[layer])[layer]; projections.append(float(torch.sum(grad*direction).item()))
        projections=np.asarray(projections); rng=np.random.default_rng(20260720)
        for b in sizes:
            n=(len(projections)//(2*b))*2*b
            if n<2*b: continue
            ordered=projections[rng.permutation(len(projections))[:n]].reshape(-1,b).mean(axis=1); pair=(ordered[0::2]-ordered[1::2])/math.sqrt(2)
            rows.append({"state_key":state.label,"batch_size":b,"robust_scale":robust_scale(pair),"n_pairs":len(pair)})
    frame=pd.DataFrame(rows)
    fit=[]
    for state_key,sub in frame.groupby("state_key"):
        train=sub[sub.batch_size<=16]; test=sub[sub.batch_size>16]; slope,intercept=np.polyfit(np.log(train.batch_size),np.log(train.robust_scale),1); gamma=-slope; pred=np.exp(intercept+slope*np.log(test.batch_size)) if len(test) else np.array([]); rmse=float(np.sqrt(np.mean((np.log(test.robust_scale)-np.log(pred))**2))) if len(test) else math.nan; fit.append({"state_key":state_key,"gamma":gamma,"heldout_log_rmse":rmse})
    pd.DataFrame(fit).to_csv(out/"batch_flow_fit.csv",index=False); frame.to_csv(out/"batch_flow.csv",index=False); return frame


def run_09() -> pd.DataFrame:
    out=output_dir("09_algorithm_vs_recipe")
    if smoke_mode():
        frame=smoke_algorithm_recipe(); frame.to_csv(out/"algorithm_vs_recipe.csv",index=False); return frame
    root=find_repo_root(); manifest,bank=load_manifest_and_bank(root); rows=[]
    # Full-recipe comparison uses real optimizer states.
    for entry in manifest["entries"]:
        for intervention in ["v_wo","qk"]:
            try:
                rec=virtual_step_defect(entry,root,bank,intervention,env_int("GAUGE_ATTN_BLOCK",3)); rows.append({"comparison":"full_recipe","candidate":entry["label"],"intervention":intervention,**rec,"status":"ok"})
            except Exception as exc:
                rows.append({"comparison":"full_recipe","candidate":entry["label"],"intervention":intervention,"defect":math.nan,"status":repr(exc)})
    # Algorithm-only fresh-state controls at the first available checkpoint.
    if manifest["entries"]:
        base=manifest["entries"][0]
        for candidate in env_csv("GAUGE_ALGORITHM_CANDIDATES",["sgd","adamw","lion","sophia_official","muon"]):
            try:
                rec=virtual_step_defect(base,root,bank,"v_wo",env_int("GAUGE_ATTN_BLOCK",3),fresh_optimizer=candidate); rows.append({"comparison":"algorithm_only_fresh_state","candidate":candidate,"intervention":"v_wo",**rec,"status":"ok"})
            except Exception as exc:
                rows.append({"comparison":"algorithm_only_fresh_state","candidate":candidate,"intervention":"v_wo","defect":math.nan,"status":repr(exc)})
    frame=pd.DataFrame(rows); frame.to_csv(out/"algorithm_vs_recipe.csv",index=False); return frame


def run_10() -> pd.DataFrame:
    root=find_repo_root(); out=output_dir("10_decision_registry",root); rows=[]
    base=root/"analysis_outputs"/"gauge_covariant_theory"
    for path in sorted(base.glob("*/**/*.csv")):
        if path.parent.name=="10_decision_registry": continue
        try:
            frame=pd.read_csv(path); rows.append({"experiment":path.parent.name,"file":str(path.relative_to(root)),"rows":len(frame),"columns":len(frame.columns),"has_nan":bool(frame.isna().any().any())})
        except Exception as exc:
            rows.append({"experiment":path.parent.name,"file":str(path.relative_to(root)),"rows":0,"columns":0,"has_nan":True,"error":repr(exc)})
    registry=pd.DataFrame(rows); registry.to_csv(out/"decision_registry.csv",index=False); save_json(out/"decision_registry.json",registry.to_dict(orient="records")); return registry


RUNNERS = {
    "00": run_00,
    "01": run_01,
    "02": run_02,
    "03": run_03,
    "04": run_04,
    "05": run_05,
    "06": run_06,
    "07": run_07,
    "08": run_08,
    "09": run_09,
    "10": run_10,
}


def run_experiment(code: str) -> Any:
    if code not in RUNNERS:
        raise KeyError(code)
    return RUNNERS[code]()
