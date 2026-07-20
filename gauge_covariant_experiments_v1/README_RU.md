# Эксперименты калибровочно-ковариантной теории

Этот пакет **только добавляет** новые notebooks, helper-модуль, проверки и скрипты миграции. Он не заменяет `gpt2nano/`, не меняет training loop и не удаляет старые notebooks.

## Научная цель

Проверяется одна узкая теория: физически содержательным объектом стохастического обучения является функциональный ток

\[
j_B=Df_\theta[-\eta\,\mathsf M_{\theta,s}g_B],
\]

а не распределение координат gradient covector. Для exact symmetry `theta' = psi(theta)` mobility должна преобразовываться ковариантно:

\[
\mathsf M_{\psi\theta,\psi_*s}=D\psi\,\mathsf M_{\theta,s}D\psi^T.
\]

Пакет не предполагает, что теория верна. Каждый notebook имеет заранее указанное условие фальсификации.

## Notebooks

1. `00_preflight_and_probe_bank.ipynb` — строгий manifest, hashes и общий probe bank.
2. `01_covector_pullback_audit.ipynb` — точное преобразование gradient covector.
3. `02_head_mixture_prediction.ipynb` — no-fit prediction transformed pooled CCDF из head-conditioned laws.
4. `03_optimizer_equivariance_defect.ipynb` — functional update defect для optimizer maps.
5. `04_sophia_saturation_mediation.ipynb` — same-state rho sweep и проверка роли coordinate saturation.
6. `05_gauge_continuation.ipynb` — продолжение двух gauge-equivalent trajectories на общей последовательности batches.
7. `06_invariant_forward_backward_spectrum.ipynb` — generalized canonical-correlation spectrum.
8. `07_token_sequence_batch_decomposition.ipynb` — token/sequence/batch participation.
9. `08_batch_aggregation_flow.ipynb` — pairwise nested batch aggregation без оценки population mean.
10. `09_algorithm_vs_recipe.ipynb` — разделение same-state algorithm effect и full-recipe trajectory effect.

## Быстрый запуск

```bash
cd /path/to/gpt2_nano_optimizer_pipeline_clean
export GPT2_NANO_REPO="$PWD"
python scripts/gauge_covariant/verify_required_files.py
python scripts/gauge_covariant/audit_notebooks.py
bash scripts/gauge_covariant/run_all_notebooks.sh --quick
```

Полный режим:

```bash
bash scripts/gauge_covariant/run_all_notebooks.sh --full
```

## Строгие свойства

- все states используют один сохранённый probe bank;
- отсутствующий requested run является ошибкой в strict mode, а не warning;
- original/transformed models получают идентичные batches;
- Q/K/V обрабатываются как slices физического `c_attn.weight`;
- checkpoint provenance сохраняется в каждом output;
- optimizer-family и recipe-level сравнения разделены;
- результаты пишутся только в `analysis_outputs/gauge_covariant_experiments/`;
- исходные checkpoints никогда не изменяются.

## Синтетический integration test

CI выполняет все notebooks сверху вниз в `GAUGE_SYNTHETIC=1`. Он проверяет JSON, imports, plotting, table/JSON exports, exact gauge identities и failure gates. Это не заменяет научный прогон на реальных checkpoints.
