import pytest
import pandas as pd
from nlp.embeddings import Embeddings

def test_fit_transform_basic():
    corpus = [
        "This is a test.",
        "This test is simple.",
        "Test the embeddings module."
    ]
    index = ["doc1", "doc2", "doc3"]

    emb = Embeddings()
    tfidf_df = emb.fit_transform(corpus, index=index)

    assert isinstance(tfidf_df, pd.DataFrame)
    assert list(tfidf_df.index) == index
    assert "test" in tfidf_df.columns
    assert tfidf_df.shape[0] == 3
    assert tfidf_df.shape[1] > 0

def test_get_vocab():
    corpus = ["one two three", "two three four"]
    emb = Embeddings()
    emb.fit_transform(corpus)
    vocab = emb.get_vocab()
    assert "two" in vocab
    assert "four" in vocab
