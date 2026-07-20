# Миграция из повреждённого рабочего каталога в чистый репозиторий

## Принцип

Исходный код из повреждённой директории **не копируется**. Новый source tree строится из:

1. чистого `gpt2_nano_optimizer_pipeline_v11_pcsgd_clean(2).zip`;
2. минимального Sophia replacement v24, который добавляет только проверенный full-sequence Hessian protocol, rho runner и dataset helper scripts;
3. additive gauge-theory patch из этого пакета.

Из повреждённого каталога переносятся только вычисленные данные и артефакты.

---

## Обязательно перенести

### 1. Все runs и checkpoints

Копировать каталог целиком:

```text
runs/
```

Нужны не только `model` weights, но и:

- `ckpt_last.pt`;
- intermediate checkpoints;
- optimizer state;
- training/evaluation logs;
- run-level JSON/CSV provenance;
- paper exports внутри run directories.

Особенно сохранить:

```text
runs/final_stage2/adamw_best/
runs/final_stage2/lion_best/
runs/final_stage2/muon_best/
runs/final_stage2/sgd_best/
runs/research1_sophia_rho/sophia_official_rho0p05_lr6e-4_wd0p2_clip1/
runs/research1_sophia_rho/sophia_official_rho0p2_lr6e-4_wd0p2_clip1/
runs/research1_sophia_rho/sophia_official_rho0p8_lr6e-4_wd0p2_clip1/
runs/research1_sophia_rho/sophia_official_rho1p5_lr6e-4_wd0p2_clip1/
```

Если существуют `runs/sweeps_stage2/`, `runs/final_*`, pilot runs или другие checkpoints, переносить весь `runs/`, чтобы ничего не пересчитывать.

### 2. OpenWebText binaries

Копировать:

```text
third_party/Sophia/data/openwebtext/train.bin
third_party/Sophia/data/openwebtext/val.bin
```

Также перенести, если существует:

```text
third_party/Sophia/data/openwebtext/meta.pkl
```

Код `third_party/Sophia` из повреждённого каталога не копировать; новый repo сам bootstrap-ит pinned upstream. Переносятся только dataset files.

### 3. Конфигурации, созданные вычислениями

Перенести, если существуют:

```text
configs/best_stage2/
configs/research1_sophia/
```

Если clean source уже содержит файл с тем же именем, сначала сравнить SHA-256; не перезаписывать source configuration автоматически.

### 4. Analysis outputs

Копировать целиком:

```text
analysis_outputs/
analysis_inputs/
```

`analysis_inputs/gauge_covariant_theory` можно пересоздать notebook-ом 00, но перенос сохраняет exact probe bank и облегчает воспроизводимость.

### 5. Выполненные notebooks и figures

Перенести только как артефакты, не заменяя source notebooks новой версии:

```text
notebooks/section1_sophia_rho_grid/*.ipynb
notebooks/fundamental_mechanism_experiments/*.ipynb
fig/
paper_figure_exports*/
```

Рекомендуемое место для старых выполненных копий:

```text
artifacts/executed_notebooks_from_old_repo/
```

### 6. Логи и experiment tracking

Опционально, но желательно:

```text
logs/
wandb/
```

---

## Не переносить

```text
gpt2nano/
scripts/
research1_audit/
tests/
third_party/Sophia/*.py
third_party/google_automl/
__pycache__/
*.pyc
.venv/
env/
.git/
```

Причина: именно source tree повреждён посторонним patch-ем.

HuggingFace cache `.cache/huggingface/` не требуется, если `train.bin` и `val.bin` сохранены. Его можно перенести только для экономии повторных downloads.

---

## Безопасная команда миграции

Пусть:

```bash
OLD=~/project/gpt2_nano_optimizer_pipeline2_broken
NEW=~/project/gpt2_nano_optimizer_pipeline2_clean
```

Создать snapshot inventory старого каталога:

```bash
cd "$OLD"
find runs analysis_outputs analysis_inputs configs -type f -print0 2>/dev/null \
  | sort -z | xargs -0 sha256sum > ~/old_repo_scientific_files.sha256
```

Перенести runs и analysis:

```bash
rsync -a --info=progress2 "$OLD/runs/" "$NEW/runs/"
[ ! -d "$OLD/analysis_outputs" ] || rsync -a "$OLD/analysis_outputs/" "$NEW/analysis_outputs/"
[ ! -d "$OLD/analysis_inputs" ] || rsync -a "$OLD/analysis_inputs/" "$NEW/analysis_inputs/"
```

Перенести dataset с dereference symlink:

```bash
mkdir -p "$NEW/third_party/Sophia/data/openwebtext"
cp -L "$OLD/third_party/Sophia/data/openwebtext/train.bin" \
      "$NEW/third_party/Sophia/data/openwebtext/train.bin"
cp -L "$OLD/third_party/Sophia/data/openwebtext/val.bin" \
      "$NEW/third_party/Sophia/data/openwebtext/val.bin"
[ ! -f "$OLD/third_party/Sophia/data/openwebtext/meta.pkl" ] || \
  cp -L "$OLD/third_party/Sophia/data/openwebtext/meta.pkl" \
        "$NEW/third_party/Sophia/data/openwebtext/meta.pkl"
```

Перенести generated configs без перезаписи существующих source files:

```bash
[ ! -d "$OLD/configs/best_stage2" ] || rsync -a --ignore-existing "$OLD/configs/best_stage2/" "$NEW/configs/best_stage2/"
[ ! -d "$OLD/configs/research1_sophia" ] || rsync -a --ignore-existing "$OLD/configs/research1_sophia/" "$NEW/configs/research1_sophia/"
```

Сохранить выполненные notebooks отдельно:

```bash
mkdir -p "$NEW/artifacts/executed_notebooks_from_old_repo"
find "$OLD/notebooks" -name '*.ipynb' -exec cp --parents '{}' "$NEW/artifacts/executed_notebooks_from_old_repo/" ';'
```

---

## Финальная проверка

```bash
cd "$NEW"
bash scripts/bootstrap_upstreams.sh
bash scripts/check_openwebtext_bins.sh
python scripts/audit_sophia_official.py
python scripts/research1_sophia/check_inventory.py
python scripts/gauge_theory/verify_environment.py
GAUGE_SMOKE_TEST=1 bash scripts/gauge_theory/run_all_notebooks.sh
```

Проверить checkpoints вручную:

```bash
find runs -name 'ckpt*.pt' -type f -printf '%s %p\n' | sort -n
```

Нулевой размер файла запрещён.

## Чек-лист полноты

- [ ] весь `runs/` перенесён;
- [ ] optimizer state присутствует в checkpoints;
- [ ] четыре corrected Sophia rho runs присутствуют;
- [ ] AdamW/Lion/Muon/SGD final runs присутствуют;
- [ ] intermediate checkpoints сохранены;
- [ ] `train.bin`, `val.bin`, при наличии `meta.pkl` перенесены;
- [ ] generated configs перенесены;
- [ ] `analysis_outputs/` сохранён;
- [ ] старые executed notebooks сохранены как artifacts;
- [ ] source code из повреждённого repo не копировался;
- [ ] upstream Sophia заново bootstrap-нут;
- [ ] smoke test новых notebooks прошёл;
- [ ] manifest notebook 00 видит все requested runs;
- [ ] SHA-256 inventory создан до удаления старого каталога.
