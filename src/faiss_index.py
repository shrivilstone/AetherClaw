import os
import pickle
import numpy as np

try:
    import faiss
except Exception as e:
    raise RuntimeError("Missing dependency: please install 'faiss-cpu' (pip).\nRun: pip install faiss-cpu") from e


class FaissIndex:
    """Simple FAISS index wrapper that stores metadata separately.

    - Stores metadata (list of dicts with at least 'path') in a pickle next to the index file.
    - Uses HNSW for efficiency by default.
    """
    def __init__(self, dim: int, index_path: str = None, use_hnsw: bool = False):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = (index_path + '.meta.pkl') if index_path else None
        self.use_hnsw = use_hnsw
        self.metadatas = []

        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            if use_hnsw:
                self.index = faiss.IndexHNSWFlat(dim, 32)
            else:
                # inner-product flat index (use normalized vectors for cosine)
                self.index = faiss.IndexFlatIP(dim)

    def add_documents(self, docs_texts, vectors: np.ndarray, metas: list):
        vectors = vectors.astype('float32')
        # index assumes normalized vectors for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadatas.extend(metas)

    def save(self, index_path: str = None):
        if index_path is None:
            index_path = self.index_path
        if index_path is None:
            raise ValueError('index_path not set')
        faiss.write_index(self.index, index_path)
        if self.metadatas is not None:
            with open(index_path + '.meta.pkl', 'wb') as f:
                pickle.dump(self.metadatas, f)

    def load(self, index_path: str):
        self.index = faiss.read_index(index_path)
        self.metadatas = []
        if os.path.exists(index_path + '.meta.pkl'):
            with open(index_path + '.meta.pkl', 'rb') as f:
                self.metadatas = pickle.load(f)

    def search(self, qvecs: np.ndarray, k: int = 5):
        # qvecs: (n, dim)
        q = qvecs.astype('float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return D, I

    def is_empty(self):
        try:
            return self.index.ntotal == 0
        except Exception:
            return True
