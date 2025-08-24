"""
Custom Lightweight Embeddings
No pretrained models - simple hash-based and word-based embeddings
"""

print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\rag\\custom_embeddings.py | Starting file execution")

import hashlib
import re
from typing import List, Dict, Set
from collections import Counter
import numpy as np
from langchain_core.embeddings import Embeddings

class SimpleMedicalEmbeddings(Embeddings):
    """
    Lightweight custom embeddings without pretrained models
    Uses a combination of:
    1. Medical term frequency
    2. Simple hash-based features
    3. Text statistics
    """
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize custom embeddings"""
        print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\rag\\custom_embeddings.py | Function = SimpleMedicalEmbeddings.__init__")
        self.embedding_dim = embedding_dim
        
        # Medical keywords for enhanced representation
        self.medical_terms = {
            'disease', 'treatment', 'patient', 'symptoms', 'diagnosis', 'therapy',
            'medicine', 'medical', 'health', 'clinical', 'hospital', 'doctor',
            'nurse', 'medication', 'drug', 'surgery', 'cancer', 'diabetes',
            'heart', 'blood', 'pressure', 'pain', 'infection', 'virus',
            'bacteria', 'fever', 'chronic', 'acute', 'syndrome', 'condition'
        }
        
        print(f"✅ Custom Medical Embeddings initialized (dim={embedding_dim})")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract features from text"""
        text = self._preprocess_text(text)
        words = text.split()
        
        if not words:
            return [0.0] * self.embedding_dim
        
        features = []
        
        # 1. Text statistics (20 features)
        features.extend([
            len(text),                           # Text length
            len(words),                          # Word count
            len(set(words)),                     # Unique words
            np.mean([len(w) for w in words]),    # Average word length
            len([w for w in words if len(w) > 6]), # Long words
        ])
        
        # 2. Medical term frequency (10 features)
        medical_count = sum(1 for word in words if word in self.medical_terms)
        features.extend([
            medical_count,                       # Medical terms count
            medical_count / len(words),          # Medical density
        ])
        
        # 3. Hash-based features (remaining dimensions)
        remaining_dims = self.embedding_dim - len(features)
        
        # Create hash features from word combinations
        word_hashes = []
        for i, word in enumerate(words[:20]):  # Use first 20 words
            # Simple hash feature
            hash_val = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            word_hashes.append(hash_val % 1000 / 1000.0)  # Normalize to 0-1
        
        # Pad or truncate to fit remaining dimensions
        if len(word_hashes) < remaining_dims:
            word_hashes.extend([0.0] * (remaining_dims - len(word_hashes)))
        else:
            word_hashes = word_hashes[:remaining_dims]
        
        features.extend(word_hashes)
        
        # Ensure exactly the right dimension
        if len(features) != self.embedding_dim:
            features = features[:self.embedding_dim]
            features.extend([0.0] * (self.embedding_dim - len(features)))
        
        # Normalize features to unit vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = [f / norm for f in features]
        
        return features
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self._extract_features(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        return [self._extract_features(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


class FastHashEmbeddings(Embeddings):
    """
    Ultra-lightweight hash-based embeddings
    Fastest option with minimal resource usage
    """
    
    def __init__(self, embedding_dim: int = 64):
        """Initialize fast hash embeddings"""
        self.embedding_dim = embedding_dim
        print(f"✅ Fast Hash Embeddings initialized (dim={embedding_dim})")
    
    def _hash_text(self, text: str) -> List[float]:
        """Generate hash-based embedding"""
        # Clean text
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        words = text.split()
        
        if not words:
            return [0.0] * self.embedding_dim
        
        # Create multiple hash features
        features = []
        
        for i in range(self.embedding_dim):
            # Create different hash seeds for each dimension
            combined_text = f"{text}_{i}"
            hash_val = int(hashlib.md5(combined_text.encode()).hexdigest()[:8], 16)
            # Normalize to [-1, 1]
            features.append((hash_val % 2000 - 1000) / 1000.0)
        
        # Normalize to unit vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = [f / norm for f in features]
        
        return features
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self._hash_text(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        return [self._hash_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


class WordCountEmbeddings(Embeddings):
    """
    Word frequency based embeddings
    Uses vocabulary and word counts for representation
    """
    
    def __init__(self, embedding_dim: int = 100, vocab_size: int = 1000):
        """Initialize word count embeddings"""
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.vocabulary = {}  # Will be built from data
        self.is_fitted = False
        print(f"✅ Word Count Embeddings initialized (dim={embedding_dim}, vocab={vocab_size})")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into words"""
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        return text.split()
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            words = self._preprocess_text(text)
            word_counts.update(words)
        
        # Take most common words
        most_common = word_counts.most_common(self.vocab_size)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.is_fitted = True
        print(f"✅ Vocabulary built with {len(self.vocabulary)} words")
    
    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to word count vector"""
        if not self.is_fitted:
            # If not fitted, use a simple hash-based approach
            return FastHashEmbeddings(self.embedding_dim)._hash_text(text)
        
        words = self._preprocess_text(text)
        word_counts = Counter(words)
        
        # Create feature vector
        features = [0.0] * min(self.embedding_dim, len(self.vocabulary))
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                if idx < len(features):
                    features[idx] = count
        
        # Pad to full dimension if needed
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = [f / norm for f in features]
        
        return features
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self._text_to_vector(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        # Build vocabulary if not fitted
        if not self.is_fitted:
            self._build_vocabulary(texts)
        
        return [self._text_to_vector(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
