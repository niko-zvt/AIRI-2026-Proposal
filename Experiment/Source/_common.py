"""Общие константы, пути и утилиты для скриптов 00..05."""
from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

SEED: Final[int] = 42

THIS_FILE: Final[Path] = Path(__file__).resolve()
SOURCE_DIR: Final[Path] = THIS_FILE.parent
EXPERIMENT_DIR: Final[Path] = SOURCE_DIR.parent
RESULTS_DIR: Final[Path] = EXPERIMENT_DIR / "Results"
PLOTS_DIR: Final[Path] = RESULTS_DIR / "plots"
INTERACTIVE_DIR: Final[Path] = RESULTS_DIR / "interactive"
POOL_DIR: Final[Path] = RESULTS_DIR / "_pool"

POOL_JSON_PATH: Final[Path] = RESULTS_DIR / "pool.json"
EMBEDDINGS_CLS_PATH: Final[Path] = RESULTS_DIR / "embeddings_cls.npy"
EMBEDDINGS_RAW_CLS_PATH: Final[Path] = RESULTS_DIR / "embeddings_raw_cls.npy"
EMBEDDINGS_CONCAT_PATH: Final[Path] = RESULTS_DIR / "embeddings_concat.npy"
EMBEDDINGS_INDEX_PATH: Final[Path] = RESULTS_DIR / "embeddings_index.json"
PROXY_CSV_PATH: Final[Path] = RESULTS_DIR / "proxy.csv"
METRICS_JSON_PATH: Final[Path] = RESULTS_DIR / "metrics.json"
RETRIEVAL_EXAMPLES_CSV: Final[Path] = RESULTS_DIR / "retrieval_examples.csv"
REPORT_MD_PATH: Final[Path] = RESULTS_DIR / "report.md"
RUN_LOG_PATH: Final[Path] = RESULTS_DIR / "run.log"

EMBEDDING_KINDS: Final[tuple[str, ...]] = ("cls", "raw_cls", "concat")
EMBEDDING_PATHS: Final[dict[str, Path]] = {
    "cls": EMBEDDINGS_CLS_PATH,
    "raw_cls": EMBEDDINGS_RAW_CLS_PATH,
    "concat": EMBEDDINGS_CONCAT_PATH,
}

# Файлы-маркеры: их наличие означает, что соответствующая база распакована.
_QMOF_PROBE: Final[str] = "ABACUF01_FSR.grid"
_COREMOF_PROBE: Final[str] = "ABAVIJ_clean.grid"
_HMOF_PROBE_REL: Final[Path] = Path("downstream_release") / "train_raspa_100bar.json"
_HMOF_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


@dataclass(frozen=True)
class RuntimePaths:
    """Валидированные пути к чекпойнту и базам для одного запуска."""

    source_dir: Path
    results_dir: Path
    moftransformer_root: Path
    pmtransformer_ckpt: Path
    qmof_raw_dir: Path
    coremof_raw_dir: Path
    hmof_root_dir: Path


def ensure_results_layout() -> None:
    """Создаёт каталоги Results/*, если их ещё нет."""
    for directory in (RESULTS_DIR, PLOTS_DIR, INTERACTIVE_DIR, POOL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    """Фиксирует seed во всех доступных RNG (random, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    """Возвращает logger с выводом в stdout и в Results/run.log."""
    ensure_results_layout()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(RUN_LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_device() -> str:
    """Возвращает 'cuda' если доступна, иначе 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _moftransformer_root() -> Path:
    """Корневой каталог установленного пакета moftransformer."""
    try:
        import moftransformer
    except ImportError as err:
        raise ImportError(
            "Пакет moftransformer не установлен. Запустите Source/run.sh - он "
            "сам поставит moftransformer==2.2.0 и скачает базы внутрь Source/.venv."
        ) from err
    return Path(moftransformer.__root_dir__)


def get_pmtransformer_ckpt() -> Path:
    """Путь до pmtransformer.ckpt."""
    return _moftransformer_root() / "database" / "pmtransformer.ckpt"


def get_qmof_raw_dir() -> Path:
    """Путь до распакованной базы QMOF."""
    return _moftransformer_root() / "database" / "qmof" / "raw"


def get_coremof_raw_dir() -> Path:
    """Путь до распакованной базы CoREMOF."""
    return _moftransformer_root() / "database" / "coremof" / "raw"


def get_hmof_root_dir() -> Path:
    """Корневой каталог распакованного hMOF."""
    return _moftransformer_root() / "database" / "hmof"


def get_hmof_split_dirs() -> tuple[Path, ...]:
    """Каталоги split'ов hMOF (train, val, test)."""
    base = get_hmof_root_dir() / "downstream_release"
    return tuple(base / split for split in _HMOF_SPLITS)


def validate_runtime_paths() -> RuntimePaths:
    """Проверяет, что moftransformer установлен и все базы скачаны."""
    moftransformer_root = _moftransformer_root()
    pmt_ckpt = get_pmtransformer_ckpt()
    qmof_raw = get_qmof_raw_dir()
    coremof_raw = get_coremof_raw_dir()
    hmof_root = get_hmof_root_dir()

    issues: list[str] = []
    if not pmt_ckpt.is_file():
        issues.append(
            f"  pmtransformer.ckpt не найден: {pmt_ckpt}\n"
            f"    Запустите: moftransformer download pretrain_model"
        )
    if not (qmof_raw / _QMOF_PROBE).is_file():
        issues.append(
            f"  QMOF не распакован: {qmof_raw}\n"
            f"    Запустите: moftransformer download qmof"
        )
    if not (coremof_raw / _COREMOF_PROBE).is_file():
        issues.append(
            f"  CoREMOF не распакован: {coremof_raw}\n"
            f"    Запустите: moftransformer download coremof"
        )
    if not (hmof_root / _HMOF_PROBE_REL).is_file():
        issues.append(
            f"  hMOF не распакован: {hmof_root / 'downstream_release'}\n"
            f"    Запустите: moftransformer download hmof"
        )

    if issues:
        details = "\n".join(issues)
        raise FileNotFoundError(
            "Не найдены обязательные ресурсы moftransformer:\n" + details
        )

    return RuntimePaths(
        source_dir=SOURCE_DIR,
        results_dir=RESULTS_DIR,
        moftransformer_root=moftransformer_root,
        pmtransformer_ckpt=pmt_ckpt,
        qmof_raw_dir=qmof_raw,
        coremof_raw_dir=coremof_raw,
        hmof_root_dir=hmof_root,
    )
