"""Шаг 03 - PCA / UMAP / t-SNE проекции латента + интерактивные plotly HTML."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    EMBEDDING_KINDS,
    EMBEDDING_PATHS,
    EMBEDDINGS_INDEX_PATH,
    INTERACTIVE_DIR,
    PLOTS_DIR,
    PROXY_CSV_PATH,
    SEED,
    ensure_results_layout,
    get_logger,
    set_global_seed,
)

LOGGER = get_logger("03_visualize_latent")

COLOR_COLUMNS: Final[tuple[str, ...]] = ("density", "metal_fraction")
PCA_VARIANCE_DIM: Final[int] = 50
UMAP_N_NEIGHBORS: Final[int] = 15
UMAP_MIN_DIST: Final[float] = 0.1
DPI: Final[int] = 150


def standardize(features: np.ndarray) -> np.ndarray:
    """Z-score стандартизация по колонкам, нулевые std заменяются на 1."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (features - mean) / std


def compute_pca(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA с фиксированным random_state."""
    from sklearn.decomposition import PCA

    n = features.shape[0]
    n_components = min(n_components, n - 1, features.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    proj = pca.fit_transform(features)
    return proj, pca.explained_variance_ratio_


def compute_umap(features: np.ndarray) -> np.ndarray:
    """UMAP в 2D с фиксированным seed."""
    import umap

    n_neighbors = min(UMAP_N_NEIGHBORS, max(2, features.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        random_state=SEED,
    )
    return reducer.fit_transform(features)


def compute_tsne(features: np.ndarray) -> np.ndarray:
    """t-SNE в 2D с автоматической perplexity."""
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, features.shape[0] // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def render_scatter_2d(
    coords: np.ndarray,
    color_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    color_label: str,
    out_path: Path,
) -> None:
    """2D scatter с цветовой шкалой → PNG."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_values,
        cmap="viridis",
        s=30,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    LOGGER.info("Сохранил %s", out_path)


def render_scatter_3d(
    coords: np.ndarray,
    color_values: np.ndarray,
    title: str,
    color_label: str,
    out_path: Path,
) -> None:
    """3D scatter с цветовой шкалой → PNG."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=color_values,
        cmap="viridis",
        s=30,
        alpha=0.85,
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, label=color_label)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    LOGGER.info("Сохранил %s", out_path)


def render_scree(explained: np.ndarray, title: str, out_path: Path) -> None:
    """Scree-plot PCA: explained variance + кумулятивная сумма."""
    fig, ax = plt.subplots(figsize=(7, 5))
    components = np.arange(1, len(explained) + 1)
    ax.bar(components, explained, color="#4C72B0", alpha=0.8, label="component")
    ax.plot(
        components,
        np.cumsum(explained),
        "o-",
        color="#C44E52",
        label="cumulative",
    )
    ax.set_title(title)
    ax.set_xlabel("PCA component")
    ax.set_ylabel("explained variance ratio")
    ax.set_ylim(0, max(1.0, np.cumsum(explained)[-1] * 1.05))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    LOGGER.info("Сохранил %s", out_path)


def render_interactive_2d(
    coords: np.ndarray,
    pool_ids: list[str],
    proxy_df: pd.DataFrame,
    color_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Интерактивный 2D scatter (plotly HTML) с tooltip по proxy."""
    import plotly.express as px

    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "pool_id": pool_ids,
            "cif_id": [proxy_df.loc[c, "cif_id"] for c in pool_ids],
            "source": [proxy_df.loc[c, "source"] for c in pool_ids],
            "n_atoms": [proxy_df.loc[c, "n_atoms"] for c in pool_ids],
            "metal_fraction": [proxy_df.loc[c, "metal_fraction"] for c in pool_ids],
            "density": [proxy_df.loc[c, "density"] for c in pool_ids],
        }
    )
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_col,
        symbol="source",
        hover_data=["pool_id", "cif_id", "n_atoms", "metal_fraction", "density"],
        title=title,
        color_continuous_scale="Viridis",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.4, color="white")))
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    LOGGER.info("Сохранил %s", out_path)


def visualize_one_embedding(
    name: str,
    embeddings: np.ndarray,
    pool_ids: list[str],
    proxy_df: pd.DataFrame,
) -> None:
    """Все проекции и графики для одного варианта эмбеддинга."""
    LOGGER.info("=== Визуализация для %s (shape=%s) ===", name, embeddings.shape)
    standardized = standardize(embeddings)

    pca_2d, exp_2d = compute_pca(standardized, n_components=2)
    pca_3d, exp_3d = compute_pca(standardized, n_components=3)
    pca_scree_proj, exp_scree = compute_pca(
        standardized, n_components=PCA_VARIANCE_DIM
    )

    umap_2d = compute_umap(standardized)
    tsne_2d = compute_tsne(standardized)

    LOGGER.info(
        "%s: PCA(2D) explained=%.3f / %.3f, UMAP/tSNE готовы",
        name,
        float(exp_2d[0]),
        float(exp_2d[1]),
    )

    for color_col in COLOR_COLUMNS:
        color_values = proxy_df.loc[pool_ids, color_col].to_numpy(dtype=float)

        render_scatter_2d(
            pca_2d,
            color_values,
            title=f"PCA-2D | {name} | colored by {color_col}",
            xlabel=f"PC1 ({exp_2d[0]:.1%})",
            ylabel=f"PC2 ({exp_2d[1]:.1%})",
            color_label=color_col,
            out_path=PLOTS_DIR / f"pca_2d_{name}_{color_col}.png",
        )
        render_scatter_3d(
            pca_3d,
            color_values,
            title=f"PCA-3D | {name} | colored by {color_col}",
            color_label=color_col,
            out_path=PLOTS_DIR / f"pca_3d_{name}_{color_col}.png",
        )
        render_scatter_2d(
            umap_2d,
            color_values,
            title=f"UMAP | {name} | colored by {color_col}",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            color_label=color_col,
            out_path=PLOTS_DIR / f"umap_{name}_{color_col}.png",
        )
        render_scatter_2d(
            tsne_2d,
            color_values,
            title=f"t-SNE | {name} | colored by {color_col}",
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
            color_label=color_col,
            out_path=PLOTS_DIR / f"tsne_{name}_{color_col}.png",
        )

    render_scree(
        exp_scree,
        title=f"PCA scree plot | {name}",
        out_path=PLOTS_DIR / f"pca_scree_{name}.png",
    )

    render_interactive_2d(
        pca_2d,
        pool_ids,
        proxy_df,
        color_col=COLOR_COLUMNS[0],
        title=f"PCA-2D | {name} | colored by {COLOR_COLUMNS[0]}",
        out_path=INTERACTIVE_DIR / f"pca_2d_{name}.html",
    )
    render_interactive_2d(
        umap_2d,
        pool_ids,
        proxy_df,
        color_col=COLOR_COLUMNS[0],
        title=f"UMAP | {name} | colored by {COLOR_COLUMNS[0]}",
        out_path=INTERACTIVE_DIR / f"umap_{name}.html",
    )


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
    proxy_df = pd.read_csv(PROXY_CSV_PATH).set_index("pool_id")

    if set(pool_ids) - set(proxy_df.index):
        raise ValueError("В embeddings_index.json есть pool_id вне proxy.csv")

    for name in EMBEDDING_KINDS:
        path = EMBEDDING_PATHS[name]
        if not path.is_file():
            raise FileNotFoundError(f"Эмбеддинг не найден: {path}")
        embeddings = np.load(path)
        if embeddings.shape[0] != len(pool_ids):
            raise ValueError(
                f"{path}: {embeddings.shape[0]} строк против {len(pool_ids)} pool_ids"
            )
        visualize_one_embedding(name, embeddings, pool_ids, proxy_df)


if __name__ == "__main__":
    main()
