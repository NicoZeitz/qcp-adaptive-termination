import colorsys
from collections import defaultdict
from typing import cast

import evoc
import numpy as np
from evoc.clustering import (
    knn_graph,
    label_propagation_init,
    neighbor_graph_matrix,
)
from evoc.clustering import node_embedding as evoc_node_embedding

EVOC_BASE_HUES = [0.00, 0.33, 0.66, 0.08, 0.50, 0.83, 0.16, 0.75, 0.41, 0.58, 0.25, 0.91]

CATEGORICAL_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#b55c1a",
    "#888888",
]

NOISE_COLOR = "#e0522d"


def build_evoc_layout(X: np.ndarray) -> tuple[np.ndarray, evoc.EVoC]:
    """Compute 2D EVoC node embedding and return (coords, clusterer)."""
    rng = np.random.RandomState(42)
    clusterer = evoc.EVoC(
        min_samples=1,
        base_min_cluster_size=max(5, int(len(X) * 0.05)),
    )
    clusterer.fit_predict(X)
    n_neighbors = clusterer.n_neighbors
    nn_inds, nn_dists = knn_graph(X.astype(np.float32), n_neighbors=n_neighbors, random_state=rng)
    graph = neighbor_graph_matrix(n_neighbors, nn_inds, nn_dists, symmetrize=True)
    init_emb = label_propagation_init(
        graph,
        n_components=2,
        approx_n_parts=int(np.clip(8 * np.sqrt(X.shape[0]), 256, 16384)),
        random_scale=0.1,
        scaling=0.5,
        noise_level=0.5,
        random_state=rng,
        data=X.astype(np.float32),
        n_label_prop_iter=20,
    )
    coords = cast(
        np.ndarray,
        evoc_node_embedding(
            graph,
            n_components=2,
            n_epochs=50,
            initial_embedding=init_emb,
            negative_sample_rate=1.0,
            noise_level=0.5,
            random_state=rng,
            verbose=False,
            reproducible_flag=True,
            initial_alpha=0.1,
        ),
    )
    return coords, clusterer


def build_color_map(cluster_layers: list[np.ndarray], top_labels: np.ndarray) -> dict[int, str]:
    """Build fine-cluster-id to hex color map encoding hierarchy via hue/lightness."""
    fine_labels = cluster_layers[0] if len(cluster_layers) > 1 else top_labels
    fine_to_top: dict[int, int] = {}
    for fine, top in zip(fine_labels.tolist(), top_labels.tolist(), strict=True):
        if fine >= 0:
            fine_to_top[int(fine)] = int(top)
    top_to_fines: dict[int, list[int]] = defaultdict(list)
    for fc, tc in sorted(fine_to_top.items()):
        top_to_fines[tc].append(fc)
    top_clusters = sorted({int(x) for x in top_labels if x >= 0})
    color_map: dict[int, str] = {}
    for tc in top_clusters:
        hue = EVOC_BASE_HUES[tc % len(EVOC_BASE_HUES)]
        fines = top_to_fines.get(tc, [])
        n = len(fines)
        for j, fc in enumerate(fines):
            lightness = 0.35 + (j / max(n - 1, 1)) * 0.27 if n > 1 else 0.48
            r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.70)
            color_map[fc] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return color_map
