from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "patch" / "notebooks" / "gauge_covariant_theory"
NB_DIR.mkdir(parents=True, exist_ok=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(title: str, purpose: str, code_id: str, acceptance: str) -> dict:
    setup = r'''from __future__ import annotations

import os
import sys
from pathlib import Path

from IPython.display import display

# The notebook can be launched from the repository root or from this folder.
_here = Path.cwd().resolve()
_candidates = [_here, *_here.parents]
_helper_dir = None
for _candidate in _candidates:
    _path = _candidate / "notebooks" / "gauge_covariant_theory"
    if (_path / "gauge_theory_utils.py").exists():
        _helper_dir = _path
        break
if _helper_dir is None and (Path.cwd() / "gauge_theory_utils.py").exists():
    _helper_dir = Path.cwd()
if _helper_dir is None:
    raise FileNotFoundError("gauge_theory_utils.py was not found")
if str(_helper_dir) not in sys.path:
    sys.path.insert(0, str(_helper_dir))

import gauge_theory_utils as gtu

print("GAUGE_SMOKE_TEST =", gtu.smoke_mode())
print("repository root =", gtu.find_repo_root())
'''
    run = f'''result = gtu.run_experiment("{code_id}")
if hasattr(result, "head"):
    display(result.head(20))
else:
    display(result)
'''
    boundary = (
        "## Интерпретационная граница\n\n"
        "Зелёный статус допустим только после заранее заданного критерия приёмки, "
        "одинакового probe bank, полного manifest-а и проверки на независимых состояниях. "
        "Smoke-mode проверяет программную целостность, но не является научным результатом."
    )
    return {
        "cells": [
            markdown(f"# {title}\n\n{purpose}"),
            markdown(f"## Заранее заданный критерий приёмки\n\n{acceptance}"),
            code(setup),
            code(run),
            markdown(boundary),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SPECS = [
    (
        "00_manifest_and_probe_bank.ipynb",
        "00. Manifest, provenance и единый probe bank",
        "Создаёт строгий список checkpoints, SHA-256, dataset provenance и один замороженный банк непересекающихся token windows. Все следующие эксперименты обязаны использовать именно его.",
        "00",
        "missing runs = 0; metadata mismatches = 0; train.bin и val.bin существуют; один и тот же hash probe bank используется во всех последующих экспериментах.",
    ),
    (
        "01_covector_transport_audit.ipynb",
        "01. Точный covector-transport audit",
        "Проверяет chain-rule transformation градиента при exact Q/K и V/W_O symmetries. Transformed gradient аналитически pull-back-ится в исходную chart.",
        "01",
        "median relative L2 error < 1e-5 и q99 < 1e-4 для каждого physical tensor, блока и состояния.",
    ),
    (
        "02_pooled_mixture_prediction.ipynb",
        "02. Предсказание pooled хвоста без fit-а",
        "Использует исходные head-conditioned noise samples и известные gauge scale factors, чтобы заранее предсказать transformed pooled CCDF. Это количественная проверка mixture identity.",
        "02",
        "predicted–measured distance лежит внутри 95% null interval independent-bank сравнения; original–measured distance существенно больше null.",
    ),
    (
        "03_optimizer_gauge_equivariance.ipynb",
        "03. Gauge-equivariance defect оптимизаторов",
        "Сравнивает фактический logit increment одного virtual optimizer step в двух function-equivalent charts. Optimizer state, clipping, weight decay и moments должны переноситься вместе с параметрами.",
        "03",
        "Covariant reference имеет defect, совместимый с numerical null; остальные методы сравниваются без post-hoc перенормировки на evaluation batches.",
    ),
    (
        "04_sophia_saturation_mediation.ipynb",
        "04. Sophia: saturation как механизм нарушения covariance",
        "Разделяет same-state virtual rho sweep и trained-state rho comparison. Измеряет win rate, saturation rate и gauge defect.",
        "04",
        "В same-state sweep defect монотонно связан с saturation rate; unsaturated subset даёт defect, близкий к numerical null; результат воспроизводится на независимых checkpoints.",
    ),
    (
        "05_gauge_continuation.ipynb",
        "05. Продолжение траекторий из двух gauge-equivalent состояний",
        "Проверяет, предсказывает ли one-step defect реальное расхождение функций при одинаковой будущей последовательности batches и RNG.",
        "05",
        "Начальные logits совпадают; sequence batches идентичны; ранний one-step defect предсказывает последующее functional divergence на held-out probe bank.",
    ),
    (
        "06_invariant_forward_backward_spectrum.ipynb",
        "06. Инвариантный forward–backward spectrum",
        "Вместо gauge-dependent scalar norms строит generalized canonical-correlation spectrum между forward activations и backward errors.",
        "06",
        "Spectrum original/transformed совпадает внутри independent-bank null, тогда как raw activation/error norms могут меняться.",
    ),
    (
        "07_token_sequence_batch_decomposition.ipynb",
        "07. Token–sequence–batch decomposition",
        "Разлагает редкие события на token contributions внутри sequence и sequence contributions внутри batch. Направление оценивается на независимом split-е.",
        "07",
        "Не менее 1000 tail batches; confidence intervals; результат устойчив к tail threshold, слоям и checkpoints; one-token и one-sequence hypotheses проверяются отдельно.",
    ),
    (
        "08_batch_aggregation_flow.ipynb",
        "08. Pairwise batch-aggregation flow",
        "Использует pairwise batch differences и nested non-overlapping groups, чтобы не оценивать population mean маленькой reference sample.",
        "08",
        "Scaling fit выполняется на b<=16 и проверяется на withheld b>=32; один exponent принимается только при mode-wise и checkpoint-wise воспроизводимости.",
    ),
    (
        "09_algorithm_vs_recipe.ipynb",
        "09. Algorithm effect против full-recipe effect",
        "Разделяет same-state algorithm intervention с matched state warmup и сравнение реально обученных recipes. Clipping controls считаются отдельно.",
        "09",
        "Нельзя смешивать две таблицы. Algorithm claim принимается только при common state/probe bank; recipe claim — только как сравнение целых training systems.",
    ),
    (
        "10_decision_registry.ipynb",
        "10. Финальный registry решений",
        "Собирает таблицы всех экспериментов, не меняя критерии после просмотра результатов. Missing/NaN/status ошибки остаются видимыми.",
        "10",
        "Каждый положительный claim имеет preregistered criterion, competing hypothesis, null control, список прошедших состояний и явный falsification status.",
    ),
]

for filename, title, purpose, code_id, acceptance in SPECS:
    payload = notebook(title, purpose, code_id, acceptance)
    (NB_DIR / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"Generated {len(SPECS)} notebooks in {NB_DIR}")
