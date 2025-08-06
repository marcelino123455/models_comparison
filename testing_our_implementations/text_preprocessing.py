import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from typing import List, Tuple, Dict, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
import os
# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """
    Text preprocessing class for lyrics data
    """

    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessor

        Args:
            language: Language for stopwords (default: english)
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''

        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_process(self, text: str, remove_stopwords: bool = True,
                           apply_stemming: bool = True) -> List[str]:
        """
        Tokenize and process text

        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming

        Returns:
            List of processed tokens
        """
        if not text:
            return []

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Apply stemming if requested
        if apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]

        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]

        return tokens

    def preprocess(self, text: str, remove_stopwords: bool = True,
                   apply_stemming: bool = True) -> str:
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming

        Returns:
            Preprocessed text as string
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_process(cleaned_text, remove_stopwords, apply_stemming)
        return ' '.join(tokens)


class TFIDFVectorizer:
    """
    Custom TF-IDF Vectorizer implementation from scratch
    """

    def __init__(self, max_features: Optional[int] = None, min_df: int = 1,
                 max_df: float = 1.0, ngram_range: Tuple[int, int] = (1, 1)):
        """
        Initialize TF-IDF Vectorizer

        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
            ngram_range: Range of n-grams to extract
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = None
        self.feature_names_ = []

    def _build_ngrams(self, tokens: List[str]) -> List[str]:
        """Build n-grams from tokens"""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        return ngrams

    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """Build vocabulary from documents"""
        doc_freq = Counter()
        n_docs = len(documents)

        for doc in documents:
            tokens = doc.split()
            ngrams = self._build_ngrams(tokens)
            unique_ngrams = set(ngrams)
            for ngram in unique_ngrams:
                doc_freq[ngram] += 1

        # Filter by document frequency
        filtered_vocab = {}
        for term, freq in doc_freq.items():
            if (freq >= self.min_df and
                freq <= self.max_df * n_docs):
                filtered_vocab[term] = freq

        # Sort by frequency and limit features
        sorted_terms = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
        if self.max_features:
            sorted_terms = sorted_terms[:self.max_features]

        # Build vocabulary mapping
        vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}

        return vocabulary, doc_freq

    def _compute_tf(self, document: str) -> Dict[str, float]:
        """Compute term frequency for a document"""
        tokens = document.split()
        ngrams = self._build_ngrams(tokens)

        tf = Counter(ngrams)
        total_terms = len(ngrams)

        if total_terms == 0:
            return {}

        # Normalize by document length
        for term in tf:
            tf[term] = tf[term] / total_terms

        return tf

    def _compute_idf(self, documents: List[str], doc_freq: Dict[str, int]) -> Dict[str, float]:
        """Compute inverse document frequency"""
        n_docs = len(documents)
        idf = {}

        for term in self.vocabulary_:
            # Add 1 to avoid log(0)
            idf[term] = np.log(n_docs / (doc_freq.get(term, 0) + 1))

        return idf

    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        Fit the vectorizer to documents

        Args:
            documents: List of documents to fit on

        Returns:
            Self for chaining
        """
        self.vocabulary_, doc_freq = self._build_vocabulary(documents)
        self.idf_ = self._compute_idf(documents, doc_freq)
        self.feature_names_ = list(self.vocabulary_.keys())

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF matrix

        Args:
            documents: List of documents to transform

        Returns:
            TF-IDF matrix
        """
        if not self.vocabulary_:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        n_docs = len(documents)
        n_features = len(self.vocabulary_)

        # Initialize matrix
        tfidf_matrix = np.zeros((n_docs, n_features))

        for doc_idx, doc in enumerate(documents):
            tf = self._compute_tf(doc)

            for term, tf_score in tf.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    idf_score = self.idf_[term]
                    tfidf_matrix[doc_idx, term_idx] = tf_score * idf_score

        return tfidf_matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit and transform documents

        Args:
            documents: List of documents

        Returns:
            TF-IDF matrix
        """
        return self.fit(documents).transform(documents)

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names_
