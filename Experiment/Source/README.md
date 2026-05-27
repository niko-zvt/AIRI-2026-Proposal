# MOF Latent Demo

Воспроизводимый эксперимент к
[Research Proposal](../../Research%20Proposal/Zhivotenko-RP.md).
Моя цель проверить, что **латентное пространство MOF/PMTransformer** можно
использовать как поисковый слой для обратного дизайна MOF.

## Идея

Foundation-модель MOF/PMTransformer уже умеет связывать структуру MOF с её
свойствами. Если латент этой модели осмысленный, то близкие в нём структуры
должны быть близки и по целевым свойствам - и тогда обратный дизайн сводится
к k-NN-поиску в латенте.

> **Гипотеза.** Близость в латенте PMTransformer коррелирует с близостью по
> proxy-свойствам MOF, и поиск в латенте даёт измеримое преимущество над
> случайным baseline'ом.

## Что делает эксперимент

1. Берёт 4500 MOF - по 1500 из QMOF, CoREMOF и hMOF (`seed=42`).
2. Прогоняет их через pretrained PMTransformer **без fine-tuning** и снимает
   3 варианта эмбеддинга (`cls`, `raw_cls`, `concat`).
3. Считает 6 базовых proxy-свойств (`n_atoms`, `n_metal_atoms`,
   `metal_fraction`, `formula_weight`, `cell_volume`, `density`).
4. Прогоняет k-NN-поиск в латенте и сравнивает с двумя baseline'ами:
   `proxy_kNN` (верхняя граница) и `random_kNN` (нижняя граница).
5. Собирает отчёт с графиками, метриками и явным вердиктом.

## Quick start

Запустить скрипт из `./Experiment/Source/`
```bash
./run.sh
```

`run.sh` сам поставит окружение, скачает чекпойнт PMTransformer и базы
QMOF/CoREMOF/hMOF, прогонит пайплайн и положит результат в
`../Results/report.md`.

## Требования

- Linux, Python 3.10.
- ~8–10 GB на диске для первого запуска (чекпойнт + три базы MOF).
- GPU желателен, но не обязателен.

## Результат

После прогона в `Results/`:

- `report.md` - отчёт с анализом и вердиктом по гипотезе;
- `metrics.json` - метрики для трёх латентов и обоих baseline'ов;
- `plots/*.png` и `interactive/*.html` - PCA/UMAP/t-SNE и кривые `Recall@k`, `MAE@k`.

## Что за рамками демо

- Fine-tuning под конкретные KPI.
- Физические proxy через Zeo++ (PLD, LCD, ASA, void fraction).
- Декодирование "латент >> CIF" и DFT-верификация кандидатов.
