import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from data_extraction.manager import load_or_fetch_articles

# 1. Si le fichier CSV existe, on le charge sans appel à l'API
@patch("data_extraction.manager.os.path.exists")
@patch("data_extraction.manager.pd.read_csv")
def test_load_existing_file(mock_read_csv, mock_exists):
    mock_exists.return_value = True
    mock_df = pd.DataFrame({"section": ["technology"], "title": ["Test title"], "text": ["Test body"]})
    mock_read_csv.return_value = mock_df

    result = load_or_fetch_articles("data/articles.csv")

    mock_read_csv.assert_called_once_with("data/articles.csv")
    assert isinstance(result, pd.DataFrame)
    assert "section" in result.columns
    assert len(result) == 1

# 2. Si le fichier CSV n’existe pas, on appelle fetch_articles_for_section
@patch("data_extraction.manager.fetch_articles_for_section")
@patch("data_extraction.manager.os.path.exists")
@patch("data_extraction.manager.pd.DataFrame.to_csv")
def test_fetch_if_file_does_not_exist(mock_to_csv, mock_exists, mock_fetch):
    mock_exists.return_value = False

    # Simuler un article pour chaque section
    mock_fetch.side_effect = lambda section: [{
        "section": section,
        "title": f"Article in {section}",
        "text": "Body"
    }]

    result = load_or_fetch_articles("data/test_articles.csv")

    # On attend autant d'appels que de sections
    assert mock_fetch.call_count > 0
    assert isinstance(result, pd.DataFrame)
    assert "section" in result.columns
    assert "title" in result.columns
    assert "text" in result.columns
    assert len(result) == len(result["section"].unique())

# 3. Si le fichier n'existe pas mais aucun article n'est retourné
@patch("data_extraction.manager.fetch_articles_for_section")
@patch("data_extraction.manager.os.path.exists")
def test_fetch_returns_empty_list(mock_exists, mock_fetch):
    mock_exists.return_value = False
    mock_fetch.return_value = []

    result = load_or_fetch_articles("data/test_empty.csv")

    # Le DataFrame retourné doit être vide
    assert isinstance(result, pd.DataFrame)
    assert result.empty

# 4. Vérifie que le dossier de sortie est bien créé si nécessaire
@patch("data_extraction.manager.fetch_articles_for_section")
@patch("data_extraction.manager.os.makedirs")
@patch("data_extraction.manager.os.path.exists")
@patch("data_extraction.manager.pd.DataFrame.to_csv")
def test_output_directory_creation(mock_to_csv, mock_exists, mock_makedirs, mock_fetch):
    mock_exists.return_value = False
    mock_fetch.side_effect = lambda section: [{
        "section": section,
        "title": f"Title {section}",
        "text": "Text"
    }]

    load_or_fetch_articles("data/test_dir_creation.csv")

    mock_makedirs.assert_called_once_with("data", exist_ok=True)
    mock_to_csv.assert_called_once()
