# embeddings.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class Embeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_df = None
    
    def fit_transform(self, corpus, index=None):
        """
        corpus : list[str] -- liste de textes
        index : list[str] -- index du DataFrame (ex: sections)
        
        Retourne un DataFrame pandas TF-IDF.
        """
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        vocab = self.vectorizer.get_feature_names_out()
        self.tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vocab, index=index)
        return self.tfidf_df
    
    def get_vocab(self):
        return self.vectorizer.get_feature_names_out()
