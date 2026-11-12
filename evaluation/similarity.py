from typing import Tuple, List, Dict
import math
from utils.text_parser import clean_text, calculate_jaccard_similarity

def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """Generates n-grams from list of tokens."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def pure_python_cosine_similarity(text1: str, text2: str) -> float:
    """Pure python TF-IDF / N-gram cosine similarity calculation with zero external network dependencies."""
    tokens1 = clean_text(text1).split()
    tokens2 = clean_text(text2).split()

    if not tokens1 or not tokens2:
        return 0.0

    items1 = tokens1 + extract_ngrams(tokens1, 2)
    items2 = tokens2 + extract_ngrams(tokens2, 2)

    vocab = list(set(items1 + items2))
    
    vec1 = [items1.count(w) for w in vocab]
    vec2 = [items2.count(w) for w in vocab]

    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    mag1 = math.sqrt(sum(v1 ** 2 for v1 in vec1))
    mag2 = math.sqrt(sum(v2 ** 2 for v2 in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)

def compute_similarity(text1: str, text2: str) -> float:
    """
    Fast, deterministic TF-IDF + N-gram Cosine Similarity Engine (0.0 to 1.0).
    Runs instantly (< 0.01 seconds) without network calls or PyTorch model downloads.
    """
    c1 = clean_text(text1)
    c2 = clean_text(text2)
    
    if not c1 or not c2:
        return 0.0
    if c1 == c2:
        return 1.0

    # Scikit-learn TF-IDF Cosine Similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit([c1, c2])
        vectors = vectorizer.transform([c1, c2])
        tfidf_sim = float(cosine_similarity(vectors[0], vectors[1])[0][0])
        jaccard_sim = calculate_jaccard_similarity(c1, c2)
        combined_sim = 0.70 * tfidf_sim + 0.30 * jaccard_sim
        return max(0.0, min(1.0, round(combined_sim, 4)))
    except Exception:
        # Pure Python Fallback
        cos_sim = pure_python_cosine_similarity(c1, c2)
        jaccard_sim = calculate_jaccard_similarity(c1, c2)
        combined = 0.65 * cos_sim + 0.35 * jaccard_sim
        return max(0.0, min(1.0, round(combined, 4)))
