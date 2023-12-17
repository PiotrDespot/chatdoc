from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class KeywordExtractor:

    def __init__(self, model_name: str, ngram_range):
        self._model = SentenceTransformer(model_name)
        self._ngram_range = ngram_range

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @model.deleter
    def model(self):
        del self._model

    def extract(self, document):
        vectorizer = CountVectorizer(ngram_range=self._ngram_range)
        document_term_matrix = vectorizer.fit_transform([document])
        extracted_words = vectorizer.get_feature_names_out(document)

        document_embeddings = self._model.encode(document)
        word_embeddings = self._model.encode(extracted_words)

        frequent_words_indices = document_term_matrix[0].nonzero()[1]
        frequent_word_embeddings = word_embeddings[frequent_words_indices]
        document_embeddings = document_embeddings.reshape(1, -1)

        distances = cosine_similarity(document_embeddings, frequent_word_embeddings)

        return extracted_words[frequent_words_indices[distances.argsort()[0][-5:]]]
