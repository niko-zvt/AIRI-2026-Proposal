"""Шаг 01 - извлечение трёх эмбеддингов из pretrained PMTransformer."""
from __future__ import annotations

import copy
import json
from typing import Any, Final

import numpy as np

from _common import (
    EMBEDDINGS_CLS_PATH,
    EMBEDDINGS_CONCAT_PATH,
    EMBEDDINGS_INDEX_PATH,
    EMBEDDINGS_RAW_CLS_PATH,
    POOL_DIR,
    POOL_JSON_PATH,
    SEED,
    ensure_results_layout,
    get_device,
    get_logger,
    get_pmtransformer_ckpt,
    set_global_seed,
    validate_runtime_paths,
)

LOGGER = get_logger("01_extract_embeddings")

BATCH_SIZE: Final[int] = 4
NBR_FEA_LEN: Final[int] = 64
IMG_SIZE: Final[int] = 30
NUM_WORKERS: Final[int] = 0


def build_module() -> Any:
    """Собирает moftransformer Module с pmtransformer.ckpt без downstream-голов."""
    ckpt_path = get_pmtransformer_ckpt()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"PMTransformer checkpoint не найден: {ckpt_path}")

    import torch  # noqa: F401

    from moftransformer.config import config as default_config_factory
    from moftransformer.modules.module import Module

    cfg = copy.deepcopy(default_config_factory())
    cfg["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0,
        "moc": 0, "bbc": 0, "classification": 0, "regression": 0,
    }
    cfg["test_only"] = True
    cfg["visualize"] = False
    cfg["load_path"] = str(ckpt_path)
    cfg["nbr_fea_len"] = NBR_FEA_LEN
    cfg["img_size"] = IMG_SIZE
    cfg["max_grid_len"] = -1

    LOGGER.info("Создаю Module с чекпойнтом %s", ckpt_path)
    module = Module(cfg).eval()

    device = get_device()
    LOGGER.info("Перемещаю модель на %s", device)
    module = module.to(device)
    return module


def build_dataloader() -> Any:
    """DataLoader поверх Results/_pool (split='test')."""
    from torch.utils.data import DataLoader

    from moftransformer.datamodules.dataset import Dataset

    test_dir = POOL_DIR / "test"
    test_json = POOL_DIR / "test.json"
    if not test_dir.is_dir() or not test_json.is_file():
        raise FileNotFoundError(
            f"Не найден symlink-пул в {POOL_DIR}. Сначала запустите 00_select_pool.py"
        )

    dataset = Dataset(
        data_dir=str(POOL_DIR),
        split="test",
        nbr_fea_len=NBR_FEA_LEN,
        draw_false_grid=False,
        downstream="",
        tasks=[],
    )
    LOGGER.info("Dataset собран: %d структур", len(dataset))

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return Dataset.collate(batch, img_size=IMG_SIZE)

    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=_collate,
        drop_last=False,
    )


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Переносит тензоры батча на device, оставляя строки/списки как есть."""
    import torch

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            moved[key] = [t.to(device, non_blocking=True) for t in value]
        else:
            moved[key] = value
    return moved


def masked_mean(features: Any, mask: Any) -> Any:
    """Среднее по последовательности с учётом маски паддинга."""
    mask_f = mask.to(features.dtype).unsqueeze(-1)
    summed = (features * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts


def extract_all_embeddings() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Прогоняет пул через модель и возвращает (pool_ids, cls, raw_cls, concat)."""
    import torch

    module = build_module()
    loader = build_dataloader()
    device = get_device()

    pool_ids_total: list[str] = []
    cls_chunks: list[np.ndarray] = []
    raw_cls_chunks: list[np.ndarray] = []
    concat_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pool_ids = list(batch["cif_id"])
            batch = move_batch_to_device(batch, device)
            ret = module.infer(batch, mask_grid=False)

            cls_feats = ret["cls_feats"].detach().cpu().numpy()
            raw_cls_feats = ret["raw_cls_feats"].detach().cpu().numpy()

            graph_mean = masked_mean(ret["graph_feats"], ret["graph_masks"])
            grid_mean = masked_mean(ret["grid_feats"], ret["grid_masks"])
            concat = torch.cat([graph_mean, grid_mean], dim=-1).detach().cpu().numpy()

            pool_ids_total.extend(pool_ids)
            cls_chunks.append(cls_feats)
            raw_cls_chunks.append(raw_cls_feats)
            concat_chunks.append(concat)

            if batch_idx % 25 == 0 or batch_idx == 0:
                LOGGER.info(
                    "Батч %d: %d структур, cls.shape=%s, concat.shape=%s",
                    batch_idx,
                    len(pool_ids),
                    cls_feats.shape,
                    concat.shape,
                )

    emb_cls = np.concatenate(cls_chunks, axis=0)
    emb_raw_cls = np.concatenate(raw_cls_chunks, axis=0)
    emb_concat = np.concatenate(concat_chunks, axis=0)

    LOGGER.info(
        "Готово: cls=%s, raw_cls=%s, concat=%s",
        emb_cls.shape,
        emb_raw_cls.shape,
        emb_concat.shape,
    )
    return pool_ids_total, emb_cls, emb_raw_cls, emb_concat


def main() -> None:
    ensure_results_layout()
    set_global_seed(SEED)
    validate_runtime_paths()

    if not POOL_JSON_PATH.is_file():
        raise FileNotFoundError(
            f"{POOL_JSON_PATH} не найден. Сначала запустите 00_select_pool.py"
        )

    pool_ids, emb_cls, emb_raw_cls, emb_concat = extract_all_embeddings()

    np.save(EMBEDDINGS_CLS_PATH, emb_cls)
    np.save(EMBEDDINGS_RAW_CLS_PATH, emb_raw_cls)
    np.save(EMBEDDINGS_CONCAT_PATH, emb_concat)
    LOGGER.info(
        "Сохранил: %s, %s, %s",
        EMBEDDINGS_CLS_PATH,
        EMBEDDINGS_RAW_CLS_PATH,
        EMBEDDINGS_CONCAT_PATH,
    )

    with EMBEDDINGS_INDEX_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"pool_ids": pool_ids}, fh, indent=2, ensure_ascii=False)
    LOGGER.info(
        "Записан индекс: %s (%d pool_id)", EMBEDDINGS_INDEX_PATH, len(pool_ids)
    )


if __name__ == "__main__":
    main()
