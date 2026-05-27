"""Шаг 04 - k-NN retrieval в латенте против proxy-kNN (потолок) и random_kNN (пол).

Каждая структура поочерёдно становится target. Ground truth - top-k в proxy.
Считаем Recall@k, MAE@k, Diversity@k и корреляцию матриц расстояний
(Spearman/Pearson) между латентом и proxy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from _common import (
    EMBEDDING_KINDS,
    EMBEDDING_PATHS,
    EMBEDDINGS_INDEX_PATH,
    METRICS_JSON_PATH,
    PLOTS_DIR,
    PROXY_CSV_PATH,
    RETRIEVAL_EXAMPLES_CSV,
    SEED,
    ensure_results_layout,
    get_logger,
    set_global_seed,
)

LOGGER = get_logger("04_retrieval_eval")

K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 10, 20)
PROXY_FEATURES: Final[tuple[str, ...]] = (
    "n_atoms",
    "n_metal_atoms",
    "metal_fraction",
    "formula_weight",
    "cell_volume",
    "density",
)
NUM_RETRIEVAL_EXAMPLES: Final[int] = 5
DPI: Final[int] = 150
PROXY_BASELINE_KEY: Final[str] = "proxy_kNN (baseline)"
RANDOM_BASELINE_KEY: Final[str] = "random_kNN (baseline)"


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Z-score стандартизация по колонкам."""
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def pairwise_distance(features: np.ndarray) -> np.ndarray:
    """Полная матрица евклидовых расстояний [N, N]."""
    return squareform(pdist(features, metric="euclidean"))


def topk_neighbours(distances: np.ndarray, k: int) -> np.ndarray:
    """Индексы top-k ближайших соседей для каждой строки (без самой себя)."""
    n = distances.shape[0]
    if k >= n:
        raise ValueError(f"k={k} >= N={n}")
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argpartition(masked, kth=k, axis=1)[:, :k]


def sort_topk(neighbours: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Сортирует top-k индексы строки по возрастанию расстояния."""
    n = neighbours.shape[0]
    sorted_neighbours = np.empty_like(neighbours)
    for i in range(n):
        order = np.argsort(distances[i, neighbours[i]])
        sorted_neighbours[i] = neighbours[i, order]
    return sorted_neighbours


def random_topk(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Для каждого i возвращает k случайных индексов != i, без повторов."""
    if k >= n:
        raise ValueError(f"random_topk: k={k} >= n={n}")
    out = np.empty((n, k), dtype=np.int64)
    pool = np.arange(n)
    for i in range(n):
        candidates = np.delete(pool, i)
        out[i] = rng.choice(candidates, size=k, replace=False)
    return out


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
    """Средний Recall@k: |retrieved ∩ ground_truth| / k."""
    n, k = retrieved.shape
    overlaps = np.zeros(n, dtype=np.float64)
    for i in range(n):
        overlaps[i] = len(np.intersect1d(retrieved[i], ground_truth[i]))
    return float(overlaps.mean() / k)


def mae_at_k(
    retrieved: np.ndarray,
    target_proxy: np.ndarray,
    candidate_proxy: np.ndarray,
) -> float:
    """|mean(proxy_retrieved) − proxy_target|, усреднённо по target и каналам."""
    means = candidate_proxy[retrieved].mean(axis=1)
    return float(np.abs(means - target_proxy).mean())


def diversity_at_k(retrieved: np.ndarray, embeddings: np.ndarray) -> float:
    """Средний (1 − cos) среди пар эмбеддингов внутри top-k."""
    n, k = retrieved.shape
    if k < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    normalized = embeddings / safe
    diffs: list[float] = []
    for i in range(n):
        idxs = retrieved[i]
        sub = normalized[idxs]
        sims = sub @ sub.T
        off = sims[np.triu_indices(k, k=1)]
        diffs.append(float(1.0 - off.mean()))
    return float(np.mean(diffs))


def correlation_of_distances(
    distances_a: np.ndarray, distances_b: np.ndarray
) -> tuple[float, float]:
    """Spearman и Pearson по off-diagonal элементам двух матриц расстояний."""
    iu = np.triu_indices_from(distances_a, k=1)
    a = distances_a[iu]
    b = distances_b[iu]
    spearman_value = spearmanr(a, b).correlation
    pearson_value = pearsonr(a, b).statistic
    spearman = 0.0 if spearman_value is None or np.isnan(spearman_value) else float(spearman_value)
    pearson = 0.0 if pearson_value is None or np.isnan(pearson_value) else float(pearson_value)
    return spearman, pearson


def render_metric_curve(
    ks: tuple[int, ...],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Кривая метрика(k) для нескольких рядов на одном графике."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, values in series.items():
        ax.plot(ks, values, marker="o", label=label)
    ax.set_xlabel("k (top-k retrieved)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    LOGGER.info("Сохранил %s", out_path)


def collect_retrieval_examples(
    pool_ids: list[str],
    proxy_df: pd.DataFrame,
    neighbours_by_space: dict[str, np.ndarray],
    n_examples: int,
) -> pd.DataFrame:
    """Таблица top-k соседей для первых n_examples target'ов по всем пространствам."""
    rows: list[dict[str, object]] = []
    for i in range(min(n_examples, len(pool_ids))):
        target_id = pool_ids[i]
        target_row = proxy_df.loc[target_id]
        for space, neighbours in neighbours_by_space.items():
            for rank, j in enumerate(neighbours[i]):
                neighbour_id = pool_ids[j]
                neighbour_row = proxy_df.loc[neighbour_id]
                rows.append(
                    {
                        "target_pool_id": target_id,
                        "target_cif_id": str(target_row["cif_id"]),
                        "target_source": str(target_row["source"]),
                        "target_density": float(target_row["density"]),
                        "target_metal_fraction": float(target_row["metal_fraction"]),
                        "space": space,
                        "rank": int(rank + 1),
                        "neighbour_pool_id": neighbour_id,
                        "neighbour_cif_id": str(neighbour_row["cif_id"]),
                        "neighbour_source": str(neighbour_row["source"]),
                        "neighbour_density": float(neighbour_row["density"]),
                        "neighbour_metal_fraction": float(
                            neighbour_row["metal_fraction"]
                        ),
                        "delta_density": float(
                            neighbour_row["density"] - target_row["density"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_results_layout()
    set_global_seed(SEED)

    if not EMBEDDINGS_INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"{EMBEDDINGS_INDEX_PATH} не найден; запустите 01_extract_embeddings.py"
        )
    if not PROXY_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"{PROXY_CSV_PATH} не найден; запустите 02_compute_proxy.py"
        )

    with EMBEDDINGS_INDEX_PATH.open("r", encoding="utf-8") as fh:
        pool_ids: list[str] = json.load(fh)["pool_ids"]
    proxy_df = pd.read_csv(PROXY_CSV_PATH).set_index("pool_id").loc[pool_ids]

    proxy_matrix = proxy_df[list(PROXY_FEATURES)].to_numpy(dtype=np.float64)
    proxy_std = standardize(proxy_matrix)
    proxy_distances = pairwise_distance(proxy_std)
    LOGGER.info(
        "Proxy: shape=%s, средняя дистанция=%.3f",
        proxy_matrix.shape,
        proxy_distances.mean(),
    )

    embedding_distances: dict[str, np.ndarray] = {}
    embedding_raw: dict[str, np.ndarray] = {}
    for name in EMBEDDING_KINDS:
        emb = np.load(EMBEDDING_PATHS[name])
        embedding_raw[name] = emb
        embedding_distances[name] = pairwise_distance(standardize(emb))
        LOGGER.info("Эмбеддинг %s: shape=%s", name, emb.shape)

    metrics: dict[str, object] = {
        "n_structures": len(pool_ids),
        "k_values": list(K_VALUES),
        "embedding_kinds": list(EMBEDDING_KINDS),
        "proxy_features": list(PROXY_FEATURES),
        "seed": SEED,
        "distance_correlations": {},
        "retrieval": {},
    }

    for name, dist_matrix in embedding_distances.items():
        spearman, pearson = correlation_of_distances(dist_matrix, proxy_distances)
        metrics["distance_correlations"][name] = {
            "spearman": spearman,
            "pearson": pearson,
        }
        LOGGER.info(
            "Корреляция расстояний (%s vs proxy): Spearman=%.3f, Pearson=%.3f",
            name,
            spearman,
            pearson,
        )

    k_max = max(K_VALUES)
    n = len(pool_ids)

    proxy_topk_full = topk_neighbours(proxy_distances, k=k_max)
    proxy_topk_full = sort_topk(proxy_topk_full, proxy_distances)

    rng = np.random.default_rng(SEED)
    random_topk_full = random_topk(n=n, k=k_max, rng=rng)

    series_recall: dict[str, list[float]] = {}
    series_mae: dict[str, list[float]] = {}

    # proxy baseline тождественен ground truth - записываем явно для прозрачности.
    series_recall[PROXY_BASELINE_KEY] = []
    series_mae[PROXY_BASELINE_KEY] = []
    proxy_diversity: list[float] = []
    for k in K_VALUES:
        retrieved = proxy_topk_full[:, :k]
        ground_truth = proxy_topk_full[:, :k]
        series_recall[PROXY_BASELINE_KEY].append(recall_at_k(retrieved, ground_truth))
        series_mae[PROXY_BASELINE_KEY].append(
            mae_at_k(retrieved, proxy_std, proxy_std)
        )
        proxy_diversity.append(diversity_at_k(retrieved, embedding_raw["cls"]))
    metrics["retrieval"][PROXY_BASELINE_KEY] = {
        "recall@k": series_recall[PROXY_BASELINE_KEY],
        "mae@k": series_mae[PROXY_BASELINE_KEY],
        "diversity@k": proxy_diversity,
    }

    series_recall[RANDOM_BASELINE_KEY] = []
    series_mae[RANDOM_BASELINE_KEY] = []
    random_diversity: list[float] = []
    for k in K_VALUES:
        retrieved = random_topk_full[:, :k]
        ground_truth = proxy_topk_full[:, :k]
        series_recall[RANDOM_BASELINE_KEY].append(recall_at_k(retrieved, ground_truth))
        series_mae[RANDOM_BASELINE_KEY].append(
            mae_at_k(retrieved, proxy_std, proxy_std)
        )
        random_diversity.append(diversity_at_k(retrieved, embedding_raw["cls"]))
    metrics["retrieval"][RANDOM_BASELINE_KEY] = {
        "recall@k": series_recall[RANDOM_BASELINE_KEY],
        "mae@k": series_mae[RANDOM_BASELINE_KEY],
        "diversity@k": random_diversity,
    }
    LOGGER.info(
        "Random baseline: Recall@5=%.4f, MAE@5=%.4f, Diversity@5=%.4f",
        series_recall[RANDOM_BASELINE_KEY][2],
        series_mae[RANDOM_BASELINE_KEY][2],
        random_diversity[2],
    )

    neighbours_for_examples: dict[str, np.ndarray] = {
        PROXY_BASELINE_KEY: proxy_topk_full[:, : K_VALUES[2]],
        RANDOM_BASELINE_KEY: random_topk_full[:, : K_VALUES[2]],
    }

    for name, dist_matrix in embedding_distances.items():
        topk_full = topk_neighbours(dist_matrix, k=k_max)
        topk_full = sort_topk(topk_full, dist_matrix)
        recalls: list[float] = []
        maes: list[float] = []
        diversities: list[float] = []
        for k in K_VALUES:
            retrieved = topk_full[:, :k]
            ground_truth = proxy_topk_full[:, :k]
            recalls.append(recall_at_k(retrieved, ground_truth))
            maes.append(mae_at_k(retrieved, proxy_std, proxy_std))
            diversities.append(diversity_at_k(retrieved, embedding_raw[name]))
        series_recall[f"latent ({name})"] = recalls
        series_mae[f"latent ({name})"] = maes
        metrics["retrieval"][f"latent ({name})"] = {
            "recall@k": recalls,
            "mae@k": maes,
            "diversity@k": diversities,
        }
        neighbours_for_examples[f"latent ({name})"] = topk_full[:, : K_VALUES[2]]
        LOGGER.info(
            "Латент %s: Recall@5=%.3f, MAE@5=%.3f, Diversity@5=%.3f",
            name,
            recalls[2],
            maes[2],
            diversities[2],
        )

    render_metric_curve(
        ks=K_VALUES,
        series=series_recall,
        title="Recall@k vs proxy-kNN ground truth",
        ylabel="Recall@k",
        out_path=PLOTS_DIR / "recall_at_k.png",
    )
    render_metric_curve(
        ks=K_VALUES,
        series=series_mae,
        title="MAE@k (standardized proxy units)",
        ylabel="MAE@k",
        out_path=PLOTS_DIR / "mae_at_k.png",
    )

    examples_df = collect_retrieval_examples(
        pool_ids=pool_ids,
        proxy_df=proxy_df,
        neighbours_by_space=neighbours_for_examples,
        n_examples=NUM_RETRIEVAL_EXAMPLES,
    )
    examples_df.to_csv(RETRIEVAL_EXAMPLES_CSV, index=False, encoding="utf-8")
    LOGGER.info("Записан %s (%d строк)", RETRIEVAL_EXAMPLES_CSV, len(examples_df))

    with METRICS_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Записан %s", METRICS_JSON_PATH)


if __name__ == "__main__":
    main()
