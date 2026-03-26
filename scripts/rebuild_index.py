#!/usr/bin/env python3
"""Rebuild FAISS index from vault resources.

Usage:
  python scripts/rebuild_index.py --resources ~/ObsidianVaults/ProjectsVault/Resources --index ~/ObsidianVaults/ProjectsVault/Resources/faiss.index
"""
import os
import sys


def main():
    cwd = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, cwd)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resources', default='~/ObsidianVaults/ProjectsVault/Resources')
    parser.add_argument('--index', default=None)
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()

    resources_dir = os.path.expanduser(args.resources)
    if args.index is None:
        index_path = os.path.join(resources_dir, 'faiss.index')
    else:
        index_path = os.path.expanduser(args.index)

    os.makedirs(resources_dir, exist_ok=True)

    # collect documents
    docs = []
    for root, _, files in os.walk(resources_dir):
        for fn in files:
            if not fn.lower().endswith(('.md', '.txt')):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                continue
            docs.append({'path': path, 'text': text})

    if not docs:
        print('No documents found under', resources_dir)
        return

    # Try to use the SentenceTransformer embeddings; if not available, fall back to TF-IDF vectors.
    texts = [d['text'] for d in docs]
    metas = [{'path': d['path']} for d in docs]

    try:
        from src.embeddings import EmbeddingClient
    except Exception as e:
        print('sentence-transformers unavailable, falling back to TF-IDF embeddings:', e)
        # lightweight TF-IDF fallback (works without torch)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as _np
        except Exception as ee:
            print('TF-IDF fallback requires scikit-learn; install it or sentence-transformers.', ee)
            raise

        print('Computing TF-IDF vectors (fallback, approximate)')
        vect = TfidfVectorizer(max_features=1024)
        X = vect.fit_transform(texts)
        vecs = X.toarray().astype('float32')
        # L2-normalize
        norms = _np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms

        dim = vecs.shape[1]
    else:
        emb = EmbeddingClient(model_name=args.model)
        print(f'Building embeddings for {len(docs)} documents using model {emb.model_name}...')
        vecs = emb.embed_texts(texts)
        dim = vecs.shape[1]

    print('Creating FAISS index...')
    from src.faiss_index import FaissIndex
    # remove existing index if dimension may differ to ensure rebuild
    try:
        if index_path and os.path.exists(index_path):
            try:
                os.remove(index_path)
            except Exception:
                pass
        if index_path and os.path.exists(index_path + '.meta.pkl'):
            try:
                os.remove(index_path + '.meta.pkl')
            except Exception:
                pass
    except Exception:
        pass
    index = FaissIndex(dim, index_path=index_path)
    index.add_documents(texts, vecs, metas)
    index.save(index_path)
    print('Saved FAISS index to', index_path)


if __name__ == '__main__':
    main()
