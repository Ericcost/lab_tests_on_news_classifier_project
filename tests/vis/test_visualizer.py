# tests/test_visualizer.py

import pytest
import pandas as pd
from vis.visualizer import Visualizer

@pytest.fixture
def sample_texts():
    original = "Ceci est un test simple pour vérifier les mots communs et la longueur"
    cleaned = "test simple vérifier mots communs longueur"
    return original, cleaned

@pytest.fixture
def sample_df(sample_texts):
    original, cleaned = sample_texts
    return pd.DataFrame({
        'text': [original],
        'cleaned_text': [cleaned],
        'section': ['test']
    })

def test_check_cleaned_text_column(sample_df):
    vis = Visualizer()
    assert vis.check_cleaned_text_column(sample_df) is True

def test_plot_lengths_returns_figure(sample_texts):
    vis = Visualizer()
    fig = vis.plot_lengths(*sample_texts)
    assert fig is not None
    assert hasattr(fig, 'axes')

def test_show_wordclouds_returns_figure(sample_texts):
    vis = Visualizer()
    fig = vis.show_wordclouds(*sample_texts)
    assert fig is not None
    assert len(fig.axes) == 2

def test_show_common_words_heatmap(sample_texts):
    vis = Visualizer()
    fig = vis.show_common_words_heatmap(*sample_texts)
    assert fig is not None
    assert hasattr(fig, 'axes')

def test_show_common_words_list(sample_texts):
    vis = Visualizer()
    fig = vis.show_common_words_list(*sample_texts)
    assert fig is not None
    assert len(fig.axes) == 2

def test_display_analysis(sample_texts):
    vis = Visualizer()
    figs = vis.display_analysis(*sample_texts)
    assert isinstance(figs, list)
    assert all(hasattr(fig, 'axes') for fig in figs)
