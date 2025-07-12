import pytest
from unittest.mock import patch
from data_extraction.fetcher import fetch_articles_for_section

@patch("data_extraction.fetcher.requests.get")
def test_fetch_articles_invalid_section_raises_error(mock_get):
    fake_response = {
        "response": {
            "results": [
                {
                    "sectionName": "unknown",
                    "webTitle": "Fake Title",
                    "fields": {
                        "bodyText": "Fake text"
                    }
                }
            ]
        }
    }

    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = fake_response

    with pytest.raises(ValueError, match="Section invalide : unknown"):
        fetch_articles_for_section("technology", max_pages=1)


@patch("data_extraction.fetcher.requests.get")
def test_fetch_articles_valid_section_passes(mock_get):
    fake_response = {
        "response": {
            "results": [
                {
                    "sectionName": "technology",
                    "webTitle": "Title",
                    "fields": {"bodyText": "Some text"}
                }
            ]
        }
    }

    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = fake_response

    articles = fetch_articles_for_section("technology", max_pages=1)
    assert len(articles) == 1
    assert articles[0]["section"] == "technology"
