import pytest
from nlp.preprocessing import Preprocessing

import nltk

# Télécharger les ressources nécessaires une fois pour les tests
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


@pytest.fixture(scope="module")
def preprocessor():
    return Preprocessing()

def test_clean_text(preprocessor):
    dirty_text = "Hello, this is a TEST! Visit https://example.com or email me@example.com."
    cleaned = preprocessor.clean_text(dirty_text)
    assert isinstance(cleaned, str)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "!" not in cleaned
    assert "test" in cleaned.lower()

def test_clean_text_non_string(preprocessor):
    assert preprocessor.clean_text(None) == ""
    assert preprocessor.clean_text(12345) == ""

def test_tokenize(preprocessor):
    text = "this is a test"
    tokens = preprocessor.tokenize(text)
    assert isinstance(tokens, list)
    assert tokens == ["this", "is", "a", "test"]

def test_remove_stopwords(preprocessor):
    tokens = ["this", "is", "a", "test", "document"]
    filtered = preprocessor.remove_stopwords(tokens)
    assert "this" not in filtered  # "this" est un stopword
    assert "test" in filtered
    assert "document" in filtered

def test_lemmatize(preprocessor):
    tokens = ["cars", "running", "mice"]
    lemmas = preprocessor.lemmatize(tokens)
    assert "car" in lemmas
    assert "run" in lemmas
    assert "mouse" in lemmas

def test_preprocess_pipeline(preprocessor):
    raw = "Cats are RUNNING around the garden. Visit www.test.com"
    processed = preprocessor.preprocess(raw)
    assert isinstance(processed, str)
    assert "cat" in processed
    assert "run" in processed
    assert "visit" in processed
    assert "www" not in processed
