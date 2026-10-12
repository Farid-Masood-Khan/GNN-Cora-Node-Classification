"""
visualize.py
2D t-SNE visualization of Cora GNN learned node embeddings.

Author: Farid Masood Khan
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_node_embeddings():
    base_dir = Path(__file__).resolve().parent.parent
    emb_file = base_dir / "models" / "cora_embeddings.npy"
    lbl_file = base_dir / "models" / "cora_labels.npy"

    if not emb_file.exists():
        print(f"[visualize] Running train.py first to produce embeddings...")
        from train import run_gnn_pipeline
        run_gnn_pipeline()

    embeddings = np.load(emb_file)
    labels = np.load(lbl_file)

    print(f"[visualize] Computing 2D t-SNE projection on {embeddings.shape[0]} node embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter_without_progress=300)
    emb_2d = tsne.fit_transform(embeddings[:1000])  # project first 1000 nodes for clarity

    class_names = [
        "Case Based",
        "Genetic Algorithms",
        "Neural Networks",
        "Probabilistic Methods",
        "Reinforcement Learning",
        "Rule Learning",
        "Theory",
    ]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels[:1000],
        cmap="tab10",
        alpha=0.8,
        s=30,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter, ticks=range(7), ax=ax)
    cbar.ax.set_yticklabels(class_names, fontsize=9)
    ax.set_title("2D t-SNE of GCN Learned Node Embeddings (Cora Citation Graph)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontweight="bold")
    plt.tight_layout()

    out_file = base_dir / "models" / "cora_tsne_embeddings.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[visualize] Saved t-SNE visualization to: {out_file}")


if __name__ == "__main__":
    plot_node_embeddings()
