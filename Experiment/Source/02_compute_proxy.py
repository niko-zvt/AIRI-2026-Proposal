"""Шаг 02 - расчёт 6 proxy-дескрипторов прямо из .graphdata и .grid.

Density считается как formula_weight / cell_volume. Атомы берутся из 8 Å
супер-ячейки (.graphdata), а параметры ячейки - из 30 Å (.grid), поэтому
абсолютные значения занижены, но относительный порядок структур сохраняется
и пригоден как proxy для retrieval.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from pymatgen.core import Element

from _common import (
    EMBEDDINGS_INDEX_PATH,
    POOL_JSON_PATH,
    PROXY_CSV_PATH,
    SEED,
    ensure_results_layout,
    get_logger,
    set_global_seed,
    validate_runtime_paths,
)

LOGGER = get_logger("02_compute_proxy")

AMU_PER_AA3_TO_G_PER_CM3: Final[float] = 1.66053906660


def parse_grid_cell(grid_path: Path) -> tuple[float, float, float, float, float, float]:
    """Возвращает (a, b, c, alpha, beta, gamma) из .grid."""
    if not grid_path.is_file():
        raise FileNotFoundError(f"Файл .grid не найден: {grid_path}")

    with grid_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if len(lines) < 2:
        raise ValueError(f"Файл .grid слишком короткий: {grid_path}")

    try:
        a, b, c = (float(x) for x in lines[0].split()[1:4])
        alpha, beta, gamma = (float(x) for x in lines[1].split()[1:4])
    except (IndexError, ValueError) as err:
        raise ValueError(f"Не удалось разобрать {grid_path}: {err}") from err

    return a, b, c, alpha, beta, gamma


def cell_volume_aa3(
    a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float
) -> float:
    """Объём ячейки в Å³ из длин рёбер (Å) и углов (градусы)."""
    cos_a = math.cos(math.radians(alpha_deg))
    cos_b = math.cos(math.radians(beta_deg))
    cos_g = math.cos(math.radians(gamma_deg))
    discriminant = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    if discriminant <= 0.0:
        raise ValueError(
            "Некорректная ячейка: дискриминант объёма ≤ 0 "
            f"(a={a}, b={b}, c={c}, α={alpha_deg}, β={beta_deg}, γ={gamma_deg})"
        )
    return float(a * b * c * math.sqrt(discriminant))


def parse_atom_numbers(graphdata_path: Path) -> np.ndarray:
    """Атомные номера из .graphdata (поле atom_num)."""
    if not graphdata_path.is_file():
        raise FileNotFoundError(f"Файл .graphdata не найден: {graphdata_path}")

    with graphdata_path.open("rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Неожиданный формат graphdata: {graphdata_path}")
    return np.asarray(data[1], dtype=np.int64)


def compute_proxy_row(
    pool_id: str, cif_id: str, source: str, source_dir: Path
) -> dict[str, float | int | str]:
    """Считает 6 proxy-дескрипторов для одной структуры."""
    atom_num = parse_atom_numbers(source_dir / f"{cif_id}.graphdata")
    n_atoms = int(atom_num.shape[0])
    if n_atoms == 0:
        raise ValueError(f"{pool_id}: ноль атомов в graphdata")

    n_metal_atoms = 0
    formula_weight = 0.0
    for z in atom_num.tolist():
        element = Element.from_Z(int(z))
        if element.is_metal:
            n_metal_atoms += 1
        formula_weight += float(element.atomic_mass)

    a, b, c, alpha, beta, gamma = parse_grid_cell(source_dir / f"{cif_id}.grid")
    cell_volume = cell_volume_aa3(a, b, c, alpha, beta, gamma)
    density = formula_weight * AMU_PER_AA3_TO_G_PER_CM3 / cell_volume

    return {
        "pool_id": pool_id,
        "cif_id": cif_id,
        "source": source,
        "n_atoms": n_atoms,
        "n_metal_atoms": int(n_metal_atoms),
        "metal_fraction": float(n_metal_atoms) / float(n_atoms),
        "formula_weight": float(formula_weight),
        "cell_volume": float(cell_volume),
        "density": float(density),
    }


def main() -> None:
    ensure_results_layout()
    set_global_seed(SEED)
    validate_runtime_paths()

    if not POOL_JSON_PATH.is_file():
        raise FileNotFoundError(f"{POOL_JSON_PATH} не найден; запустите 00_select_pool.py")
    if not EMBEDDINGS_INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"{EMBEDDINGS_INDEX_PATH} не найден; запустите 01_extract_embeddings.py"
        )

    with POOL_JSON_PATH.open("r", encoding="utf-8") as fh:
        pool: list[dict[str, str | int]] = json.load(fh)
    with EMBEDDINGS_INDEX_PATH.open("r", encoding="utf-8") as fh:
        pool_id_order: list[str] = json.load(fh)["pool_ids"]

    pool_by_id: dict[str, dict[str, str | int]] = {
        str(entry["pool_id"]): entry for entry in pool
    }
    missing = set(pool_id_order) - set(pool_by_id.keys())
    if missing:
        raise ValueError(
            f"В embeddings_index.json есть pool_id вне pool.json: {missing}"
        )

    rows: list[dict[str, float | int | str]] = []
    for pool_id in pool_id_order:
        entry = pool_by_id[pool_id]
        try:
            row = compute_proxy_row(
                pool_id=pool_id,
                cif_id=str(entry["cif_id"]),
                source=str(entry["source"]),
                source_dir=Path(str(entry["source_dir"])),
            )
        except (FileNotFoundError, ValueError) as err:
            LOGGER.error("Пропускаю %s из-за ошибки: %s", pool_id, err)
            raise
        rows.append(row)
        LOGGER.debug(
            "%s: n_atoms=%d, density=%.3f g/cm³, V=%.1f Å³",
            pool_id,
            row["n_atoms"],
            row["density"],
            row["cell_volume"],
        )

    df = pd.DataFrame(rows, columns=[
        "pool_id", "cif_id", "source", "n_atoms", "n_metal_atoms",
        "metal_fraction", "formula_weight", "cell_volume", "density",
    ])
    df.to_csv(PROXY_CSV_PATH, index=False, encoding="utf-8")
    LOGGER.info(
        "Записан %s: %d строк; density: min=%.3f, mean=%.3f, max=%.3f",
        PROXY_CSV_PATH,
        len(df),
        df["density"].min(),
        df["density"].mean(),
        df["density"].max(),
    )


if __name__ == "__main__":
    main()
