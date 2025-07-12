# tests/test_tsne_plot.py

import pytest
import pandas as pd
import numpy as np
from vis.tsne_plot import plot_tsne

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'dim1': np.random.rand(10),
        'dim2': np.random.rand(10),
        'dim3': np.random.rand(10),
        'label': ['A'] * 5 + ['B'] * 5
    })

def test_plot_tsne_returns_figure(sample_df):
    embeddings = sample_df.drop(columns=['label'])
    labels = sample_df['label']
    
    fig = plot_tsne(embeddings, labels)
    assert fig is not None
    assert hasattr(fig, 'axes')

