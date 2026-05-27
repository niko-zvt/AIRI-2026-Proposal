"""Шаг 00 - формирование пула из 1500 QMOF + 1500 CoREMOF + 1500 hMOF."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Final

import numpy as np

from _common import (
    POOL_DIR,
    POOL_JSON_PATH,
    SEED,
    ensure_results_layout,
    get_coremof_raw_dir,
    get_hmof_split_dirs,
    get_logger,
    get_qmof_raw_dir,
    set_global_seed,
    validate_runtime_paths,
)

LOGGER = get_logger("00_select_pool")

QMOF_SAMPLE_SIZE: Final[int] = 1500
COREMOF_SAMPLE_SIZE: Final[int] = 1500
HMOF_SAMPLE_SIZE: Final[int] = 1500
QMOF_STRATA: Final[int] = 5
COREMOF_STRATA: Final[int] = 5
HMOF_STRATA: Final[int] = 5
REQUIRED_SUFFIXES: Final[tuple[str, ...]] = (".graphdata", ".grid", ".griddata16")
SPLIT_NAME: Final[str] = "test"


def list_complete_cif_ids(directory: Path) -> list[str]:
    """Возвращает cif_id, у которых есть все три файла: .graphdata, .grid, .griddata16."""
    if not directory.exists():
        raise FileNotFoundError(f"Каталог не существует: {directory}")

    by_id: dict[str, set[str]] = {}
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        suffix = entry.suffix
        if suffix not in REQUIRED_SUFFIXES:
            continue
        cif_id = entry.name[: -len(suffix)]
        by_id.setdefault(cif_id, set()).add(suffix)

    complete = sorted(
        cif_id for cif_id, suffixes in by_id.items() if set(REQUIRED_SUFFIXES) <= suffixes
    )
    LOGGER.info(
        "Каталог %s: %d полных структур из %d уникальных id",
        directory,
        len(complete),
        len(by_id),
    )
    return complete


def read_n_atoms(graphdata_path: Path) -> int:
    """Извлекает число атомов из .graphdata (поле atom_num)."""
    if not graphdata_path.is_file():
        raise FileNotFoundError(f"Файл не существует: {graphdata_path}")

    with graphdata_path.open("rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(
            f"Неожиданный формат graphdata в {graphdata_path}: "
            f"тип={type(data).__name__}"
        )

    atom_num = np.asarray(data[1])
    return int(atom_num.shape[0])


def stratified_sample(
    candidates: list[str],
    sizes: dict[str, int],
    sample_size: int,
    n_strata: int,
    rng: np.random.Generator,
    label: str,
) -> list[str]:
    """Стратифицированная случайная выборка по числу атомов."""
    if not candidates:
        raise ValueError(f"{label}: список кандидатов пуст")
    if sample_size > len(candidates):
        raise ValueError(
            f"{label}: запрошено {sample_size} структур, доступно {len(candidates)}"
        )

    n_atoms_array = np.array([sizes[cif_id] for cif_id in candidates], dtype=np.int64)
    quantiles = np.linspace(0, 1, n_strata + 1)
    edges = np.quantile(n_atoms_array, quantiles)
    edges[0] -= 1
    edges[-1] += 1

    per_stratum = max(1, sample_size // n_strata)
    remainder = sample_size - per_stratum * n_strata

    selected: list[str] = []
    for stratum_idx in range(n_strata):
        low, high = edges[stratum_idx], edges[stratum_idx + 1]
        in_stratum = [
            cif_id
            for cif_id, n in zip(candidates, n_atoms_array)
            if low < n <= high
        ]
        take = per_stratum + (1 if stratum_idx < remainder else 0)
        take = min(take, len(in_stratum))
        if take == 0:
            continue
        chosen = rng.choice(in_stratum, size=take, replace=False)
        selected.extend(chosen.tolist())
        LOGGER.info(
            "%s страта %d: n_atoms in (%.0f, %.0f], доступно %d, выбрано %d",
            label,
            stratum_idx + 1,
            low,
            high,
            len(in_stratum),
            take,
        )

    if len(selected) < sample_size:
        leftover = [cif for cif in candidates if cif not in selected]
        extras = rng.choice(
            leftover, size=sample_size - len(selected), replace=False
        ).tolist()
        selected.extend(extras)
        LOGGER.info("%s добор из общего пула: +%d", label, len(extras))

    return sorted(selected[:sample_size])


def list_complete_cif_ids_split(split_dirs: tuple[Path, ...]) -> dict[str, Path]:
    """{cif_id → split_dir} для базы с train/val/test layout (hMOF)."""
    found: dict[str, Path] = {}
    seen_any = False
    for sub in split_dirs:
        if not sub.is_dir():
            LOGGER.warning("Split-каталог отсутствует: %s", sub)
            continue
        seen_any = True
        for cif_id in list_complete_cif_ids(sub):
            found.setdefault(cif_id, sub)
    if not seen_any:
        raise FileNotFoundError(
            f"Ни один из split-каталогов не существует: {list(split_dirs)}"
        )
    LOGGER.info(
        "Split-объединение по %d каталогам: %d уникальных id",
        len(split_dirs),
        len(found),
    )
    return found


def make_pool_id(source: str, cif_id: str) -> str:
    """Префикс источником защищает от коллизий cif_id между QMOF / CoREMOF / hMOF."""
    return f"{source}__{cif_id}"


def link_pool_files(pool: list[dict[str, str | int]]) -> Path:
    """Готовит Results/_pool/test/ через symlinks для moftransformer.Dataset."""
    split_dir = POOL_DIR / SPLIT_NAME
    split_dir.mkdir(parents=True, exist_ok=True)

    for entry in split_dir.iterdir():
        if entry.is_symlink() or entry.is_file():
            entry.unlink()

    targets: dict[str, float] = {}
    for record in pool:
        pool_id = str(record["pool_id"])
        cif_id = str(record["cif_id"])
        source_dir = Path(str(record["source_dir"]))
        for suffix in REQUIRED_SUFFIXES:
            src = source_dir / f"{cif_id}{suffix}"
            dst = split_dir / f"{pool_id}{suffix}"
            if not src.is_file():
                raise FileNotFoundError(f"Отсутствует исходный файл: {src}")
            os.symlink(src, dst)
        targets[pool_id] = 1.0

    json_path = POOL_DIR / f"{SPLIT_NAME}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(targets, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Symlink-пул: %d структур в %s, манифест %s", len(pool), split_dir, json_path
    )
    return POOL_DIR


def collect_source_from_index(
    source: str,
    cif_to_dir: dict[str, Path],
    sample_size: int,
    n_strata: int,
    rng: np.random.Generator,
) -> list[dict[str, str | int]]:
    """Выборка из источника, где cif_id может лежать в разных каталогах (hMOF)."""
    LOGGER.info("Считаю n_atoms для %s: %d файлов", source, len(cif_to_dir))
    sizes: dict[str, int] = {}
    for cif_id, source_dir in cif_to_dir.items():
        try:
            sizes[cif_id] = read_n_atoms(source_dir / f"{cif_id}.graphdata")
        except (ValueError, OSError) as err:
            LOGGER.warning("Пропускаю %s/%s: %s", source, cif_id, err)
    candidates = sorted(sizes.keys())
    LOGGER.info("%s-кандидатов с валидным n_atoms: %d", source, len(candidates))

    selected = stratified_sample(
        candidates=candidates,
        sizes=sizes,
        sample_size=sample_size,
        n_strata=n_strata,
        rng=rng,
        label=source,
    )

    return [
        {
            "pool_id": make_pool_id(source, cif_id),
            "cif_id": cif_id,
            "source": source,
            "source_dir": str(cif_to_dir[cif_id]),
            "n_atoms": sizes[cif_id],
        }
        for cif_id in selected
    ]


def collect_source(
    source: str,
    raw_dir: Path,
    sample_size: int,
    n_strata: int,
    rng: np.random.Generator,
) -> list[dict[str, str | int]]:
    """Выборка из источника с плоским raw/-каталогом (QMOF, CoREMOF)."""
    cif_ids = list_complete_cif_ids(raw_dir)
    cif_to_dir = {cif_id: raw_dir for cif_id in cif_ids}
    return collect_source_from_index(
        source=source,
        cif_to_dir=cif_to_dir,
        sample_size=sample_size,
        n_strata=n_strata,
        rng=rng,
    )


def build_pool() -> list[dict[str, str | int]]:
    """Собирает итоговый пул: 1500 QMOF + 1500 CoREMOF + 1500 hMOF."""
    rng = np.random.default_rng(SEED)
    qmof_pool = collect_source(
        source="qmof",
        raw_dir=get_qmof_raw_dir(),
        sample_size=QMOF_SAMPLE_SIZE,
        n_strata=QMOF_STRATA,
        rng=rng,
    )
    coremof_pool = collect_source(
        source="coremof",
        raw_dir=get_coremof_raw_dir(),
        sample_size=COREMOF_SAMPLE_SIZE,
        n_strata=COREMOF_STRATA,
        rng=rng,
    )
    hmof_index = list_complete_cif_ids_split(get_hmof_split_dirs())
    hmof_pool = collect_source_from_index(
        source="hmof",
        cif_to_dir=hmof_index,
        sample_size=HMOF_SAMPLE_SIZE,
        n_strata=HMOF_STRATA,
        rng=rng,
    )
    pool = qmof_pool + coremof_pool + hmof_pool

    pool_ids = {entry["pool_id"] for entry in pool}
    if len(pool_ids) != len(pool):
        raise RuntimeError("Коллизия pool_id - не должно случаться при префиксе source__")

    LOGGER.info(
        "Итоговый пул: %d (qmof=%d, coremof=%d, hmof=%d)",
        len(pool),
        len(qmof_pool),
        len(coremof_pool),
        len(hmof_pool),
    )
    return pool


def main() -> None:
    ensure_results_layout()
    set_global_seed(SEED)
    validate_runtime_paths()

    pool = build_pool()

    with POOL_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(pool, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Записан %s (%d записей)", POOL_JSON_PATH, len(pool))

    link_pool_files(pool)


if __name__ == "__main__":
    main()
