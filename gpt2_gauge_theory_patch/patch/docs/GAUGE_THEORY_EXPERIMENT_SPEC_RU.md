# Техническое задание: калибровочно-ковариантная теория стохастического обучения

## Центральный объект

Пусть `f(theta)` — функция сети, `g_B=d_theta L_B` — minibatch-gradient covector, а optimizer state задаёт mobility map `M_{theta,s}`. Функциональный стохастический ток:

```text
j_B = Df_theta[-eta M_{theta,s} g_B].
```

Для exact symmetry `theta' = psi(theta)` фундаментальная динамика должна удовлетворять covariance condition:

```text
M_{psi(theta),psi_*s} = Dpsi M_{theta,s} Dpsi^T,
```

что влечёт инвариантность функционального тока `j'_B=j_B`.

Текущие coordinate-tail графики рассматриваются как chart-dependent phenomenology, а не как конечный закон.

---

## P0. Provenance и shared probe bank

**Цель:** исключить скрытые различия dataset samples, coordinates, directions и checkpoints.

**Обязательные артефакты:**

- explicit run/checkpoint manifest;
- SHA-256 checkpoints и dataset bins;
- один общий банк непересекающихся 256-token windows;
- nested batch partitions;
- единый seed bank.

**Запрет:** requested run не может быть silently skipped.

---

## E1. Covector transport

**Интервенции:** exact `Q_h -> c_h Q_h`, `K_h -> K_h/c_h`; exact `V_h -> c_h V_h`, `W_{O,h} -> W_{O,h}/c_h`.

**Предсказание:** `g = J_psi^T g'`.

**Метрики:** relative L2/Linf error, per-head error, pulled-back CCDF distance.

**Приёмка:** median L2 < 1e-5, q99 < 1e-4.

**Что опровергает:** неверную orientation fused Q/K/V, неправильный tensor slicing, non-exact intervention.

---

## E2. Pooled-mixture prediction

**Предсказание без fit-а:** transformed coordinate samples равны original covector samples, разделённым на известные parameter scale factors. Predicted pooled CCDF строится до измерения transformed model.

**Null:** original model на двух independent probe-bank halves.

**Приёмка:** predicted–measured distance внутри 95% null interval; original–measured выше null.

**Что доказывает:** изменение pooled хвоста объясняется gauge-induced mixing, а не изменением функции.

---

## E3. Optimizer gauge-equivariance defect

**Объект:** actual virtual optimizer step со всеми details — optimizer state, bias correction, global clipping, weight decay, epsilon, Sophia saturation и реальный `tokens_per_update`.

**Метрика:**

```text
E_psi = E ||Delta logits' - Delta logits||^2 / (E ||Delta logits||^2 + eps).
```

**Обязательное условие:** optimizer state трансформируется вместе с параметрами.

**Что доказывает:** какой optimizer задаёт transport на quotient/function space, а какой зависит от chart.

---

## E4. Sophia saturation mediation

### E4a — same-state rho intervention

Один checkpoint, одни `exp_avg` и `hessian`, меняется только rho.

### E4b — trained-state rho comparison

Используются четыре реально обученных состояния.

**Измерения:** raw ratio, win rate, saturation rate, external clip activation, gauge defect.

**Предсказание:** unsaturated diagonal Hessian-ratio map почти covariant; saturation увеличивает defect.

**Фальсификация:** большой defect на несатурированных coordinates или отсутствие связи defect–saturation в same-state sweep.

---

## E5. Gauge continuation

Две function-equivalent copies получают одинаковые будущие batches, dropout/RNG и sampled Hessian labels. Сравниваются logits, loss, hidden representations и optimizer state 500–2000 steps.

**Предсказание:** one-step defect предсказывает скорость дальнейшего functional divergence.

**Важно:** текущий notebook сначала выполняет безопасный linearized pilot. Полный continuation следует запускать после успешных E1–E4.

---

## E6. Invariant forward–backward spectrum

Для affine module собираются token matrices `H` и `E`, затем generalized canonical-correlation spectrum:

```text
eig(S_E^+ G S_H^+ G^T),  G=E^T H/T.
```

**Предсказание:** spectrum сохраняется при invertible basis changes, хотя raw activation/error norms меняются.

**Приёмка:** spectrum distance внутри independent-probe null.

---

## E7. Token–sequence–batch decomposition

Два разных уровня participation:

1. token contributions внутри 256-token sequence;
2. sequence contributions внутри optimizer batch.

Direction оценивается на независимом split-е; signed contributions не обрезаются молча.

**Приёмка:** минимум 1000 tail batches на состояние, bootstrap CI, threshold sensitivity.

**Что различает:** one-token burst, one-sequence burst и collective mode.

---

## E8. Pairwise batch aggregation

Используется:

```text
delta_b = (g_bar(B1)-g_bar(B2))/sqrt(2),
```

по nested non-overlapping groups. Population mean отдельно не оценивается.

**Batch sizes:** 1,2,4,8,16,32,64,128.

**Cross-validation:** fit на `b<=16`, test на `b>=32`.

**Что различает:** Gaussian, stable-like, truncated crossover и mode-dependent anisotropic flow.

---

## E9. Algorithm против recipe

Две отдельные таблицы:

- same-state algorithm intervention с matched warmup и common probe bank;
- full-recipe trajectories на собственных checkpoints.

Clipping controls считаются отдельно. Смешивать таблицы запрещено.

---

## E10. Decision registry

Каждый claim должен содержать:

- preregistered criterion;
- competing hypothesis;
- null control;
- список checkpoints/optimizers;
- passed/failed/undetermined;
- ссылку на machine-readable CSV/JSON.

## Минимальные scientific gates

- минимум early/mid/late checkpoints;
- не менее трёх training seeds для финального claim;
- один и тот же probe bank;
- no silent skips;
- uncertainty intervals;
- held-out test по batch size/checkpoint/model;
- exact function-equivalence check перед любой gauge интерпретацией.
