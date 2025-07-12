import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_tsne(embeddings, labels):
    """
    Projette les embeddings en 2D avec t-SNE et retourne une figure matplotlib.
    
    Args:
        embeddings (pd.DataFrame or np.ndarray): Données haute dimension.
        labels (list, pd.Series, np.ndarray): Labels associés à chaque point.

    Returns:
        fig (matplotlib.figure.Figure): Scatterplot t-SNE.
    """
    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1)  # Évite l'erreur si peu d'exemples

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)

    df_tsne = pd.DataFrame(tsne_result, columns=["x", "y"])
    df_tsne["label"] = labels

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df_tsne,
        x="x", y="y",
        hue="label",
        palette="tab10",
        s=70,
        alpha=0.8,
        ax=ax
    )
    ax.set_title("Projection t-SNE des embeddings", fontsize=14)
    ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.close(fig)
    return fig
