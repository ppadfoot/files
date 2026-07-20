# Gauge-covariant theory experiments

Этот каталог добавляется **поверх чистого v11 + минимального исправления Sophia**. Существующие файлы и notebooks не удаляются.

## Порядок запуска

```bash
cd ~/project/gpt2_nano_optimizer_pipeline2
conda activate gpt2-nano-optim
export GPT2_NANO_REPO="$PWD"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

jupyter lab notebooks/gauge_covariant_theory/
```

Выполнять строго по порядку `00` → `10`.

`00_manifest_and_probe_bank.ipynb` является обязательным gate. Он должен увидеть все requested runs, dataset bins и создать:

```text
analysis_inputs/gauge_covariant_theory/resolved_manifest.json
analysis_inputs/gauge_covariant_theory/probe_bank.npz
analysis_inputs/gauge_covariant_theory/probe_bank.json
```

## Быстрый программный тест

```bash
GAUGE_SMOKE_TEST=1 bash scripts/gauge_theory/run_all_notebooks.sh
```

Smoke-mode использует синтетические данные и проверяет imports, JSON notebooks, сохранение таблиц/графиков и точные математические identities. Он не является научным экспериментом.

## Научный режим

```bash
unset GAUGE_SMOKE_TEST
export GAUGE_CHECKPOINT_POLICY=early_mid_late
export GAUGE_RUN_FILTER=adamw,lion,muon,sgd,sophia_rho_0p05,sophia_rho_0p2,sophia_rho_0p8,sophia_rho_1p5
```

Параметры вычислительной мощности задаются через `GAUGE_*` environment variables, перечисленные в ноутбуках и `docs/GAUGE_THEORY_EXPERIMENT_SPEC_RU.md`.

## Строгая интерпретация

- Coordinate-tail CCDF остаётся полезной феноменологией, но не считается параметризационно-инвариантным законом.
- Положительные claims принимаются только при shared probe bank, exact provenance и заранее заданном критерии.
- Notebook не имеет права молча пропускать requested run.
- `status != ok`, NaN и missing entries должны сохраняться в final registry.
