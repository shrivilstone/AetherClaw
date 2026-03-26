import os
import math
from collections import Counter
from typing import List, Tuple

class DocumentStore:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.expanduser(root_dir)
        self.docs = []  # list of (path, text, term_freq)
        self._load()

    def _tokenize(self, text: str):
        # very small tokenizer for demo
        return [w.lower() for w in text.split() if w.isalnum() or ''.join(ch.isalnum() for ch in w)]

    def _load(self):
        if not os.path.isdir(self.root_dir):
            return
        for root, _, files in os.walk(self.root_dir):
            for fn in files:
                if not fn.lower().endswith('.md') and not fn.lower().endswith('.txt'):
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except Exception:
                    continue
                tokens = [t for t in (w.strip('\n\r.,!?;:\"\'') for w in text.split()) if t]
                tf = Counter(tokens)
                self.docs.append((path, text, tf))

    def query(self, q: str, k: int = 3) -> List[Tuple[str,str,float]]:
        # naive tf-based similarity: dot product of shared tokens
        qtokens = [w.lower() for w in q.split() if w]
        qtf = Counter(qtokens)
        scores = []
        for path, text, tf in self.docs:
            # dot product
            s = 0
            for t, v in qtf.items():
                s += v * tf.get(t, 0)
            # normalize by lengths
            denom = math.sqrt(sum(qtf.values()) * sum(tf.values()) + 1e-9)
            scores.append((path, text, s / (denom or 1)))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:k]

if __name__ == '__main__':
    ds = DocumentStore('~/ObsidianVaults/ProjectsVault/Resources')
    print(ds.query('project summary', k=3))


class Retriever(DocumentStore):
    """Retriever using dense embeddings (numpy) with an optional BM25 hybrid fallback.

    This implementation avoids depending on FAISS C extensions to reduce platform
    instability; it stores dense vectors as a NumPy array and performs matrix
    dot-product search (suitable for small-to-medium vaults).
    """
    def __init__(self, root_dir: str, mode: str = 'auto', index_path: str = None, embedding_model: str = 'all-MiniLM-L6-v2'):
        super().__init__(root_dir)
        self.root_dir = os.path.expanduser(root_dir)
        self.mode = mode
        self.docs_list = [{'path': p, 'text': t} for (p, t, _) in self.docs]

        # embedding components
        self.emb = None
        self.vectors = None
        self.metas = None
        self.index_path = index_path
        self.embedding_model = embedding_model

        # BM25 fallback
        self.bm25 = None

        try:
            from .embeddings import EmbeddingClient
        except Exception:
            self.mode = 'naive'
            return

        # build or load dense numpy vectors
        try:
            self.emb = EmbeddingClient(model_name=embedding_model)
            docs_texts = [d['text'] for d in self.docs_list]
            if docs_texts:
                vecs = self.emb.embed_texts(docs_texts)
                # persist vectors if index_path provided
                try:
                    if index_path:
                        np_path = os.path.expanduser(index_path + '.npy')
                        import numpy as _np
                        _np.save(np_path, vecs)
                except Exception:
                    pass
                self.vectors = vecs
                self.metas = [{'path': d['path']} for d in self.docs_list]
        except Exception:
            # fall back to naive mode
            self.mode = 'naive'

        if self.mode in ('auto', 'hybrid'):
            try:
                from rank_bm25 import BM25Okapi
                corpus_tokens = [[tok.lower() for tok in d['text'].split()] for d in self.docs_list]
                self.bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
            except Exception:
                self.bm25 = None

    def _tokenize(self, text: str):
        return [w.lower() for w in text.split() if w]

    def query(self, q: str, k: int = 3):
        # dense numpy search
        if self.vectors is not None:
            try:
                qvec = self.emb.embed_texts([q])
                # vectors are L2-normalized; dot product == cosine similarity
                import numpy as _np
                scores = (self.vectors @ qvec.T).reshape(-1)
                order = _np.argsort(scores)[::-1][:k]
                results = []
                for i in order:
                    results.append((self.docs_list[i]['path'], self.docs_list[i]['text'], float(scores[i])))
                # hybrid: if not enough results, use BM25 fallback
                if self.mode == 'hybrid' and len(results) < k and self.bm25 is not None:
                    qtokens = self._tokenize(q)
                    bm_scores = self.bm25.get_scores(qtokens)
                    bm_order = _np.argsort(bm_scores)[::-1][:k]
                    for i in bm_order:
                        path = self.docs_list[i]['path']
                        if path not in [r[0] for r in results]:
                            results.append((path, self.docs_list[i]['text'], float(bm_scores[i])))
                return results[:k]
            except Exception:
                pass

        # fallback to naive token-based similarity
        return super().query(q, k=k)
