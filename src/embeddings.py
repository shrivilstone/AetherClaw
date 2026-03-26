import os
import warnings
import numpy as np


class EmbeddingClient:
    """Embedding client with two backends:

    - Prefer downloading model snapshot via `huggingface_hub` and loading local files.
    - Use `sentence-transformers` if importable and can load from the snapshot.
    - Otherwise use `transformers` + mean pooling (requires `torch`).
    """
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.backend = None

        # normalize short model ids like 'all-MiniLM-L6-v2' -> 'sentence-transformers/all-MiniLM-L6-v2'
        if '/' not in model_name:
            model_name = f'sentence-transformers/{model_name}'
        self.model_name = model_name

        # auth token support
        token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_HUB_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

        # Attempt to download a local snapshot via huggingface_hub (best-effort).
        local_dir = None
        try:
            from huggingface_hub import snapshot_download
            try:
                # newer API accepts use_auth_token
                local_dir = snapshot_download(repo_id=self.model_name, repo_type='model', use_auth_token=(token if token else None))
            except TypeError:
                # fallback to older param name
                local_dir = snapshot_download(repo_id=self.model_name, repo_type='model', token=(token if token else None))
        except Exception as e:
            warnings.warn(f"huggingface_hub snapshot_download failed or not available ({e}); will attempt direct transformers load.")

        # If sentence-transformers is importable and compatible, prefer it (load from local snapshot when possible)
        if local_dir is not None:
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self._st_model = SentenceTransformer(local_dir)
                    self.dim = self._st_model.get_sentence_embedding_dimension()
                    self.backend = 'sentence_transformers'
                    return
                except Exception:
                    # fall through to transformers-based loader
                    pass
            except Exception:
                # cannot import sentence_transformers; continue to transformers fallback
                pass

        # transformers fallback (load from local_dir when available to avoid network)
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except Exception as e:
            raise RuntimeError("Missing embedding backend: install 'transformers' + 'torch' or 'sentence-transformers'.") from e

        tok_kwargs = {'use_fast': True}
        if token:
            tok_kwargs['use_auth_token'] = token

        try:
            if local_dir is not None:
                # prefer loading from the downloaded snapshot
                self._tok = AutoTokenizer.from_pretrained(local_dir, **tok_kwargs, local_files_only=True)
                self._model = AutoModel.from_pretrained(local_dir, local_files_only=True)
                self.model_name = local_dir
            else:
                self._tok = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
                self._model = AutoModel.from_pretrained(self.model_name, use_auth_token=(token if token else None))
        except Exception as e:
            # try prefixed repo id as a last resort
            try:
                alt = self.model_name
                if not alt.startswith('sentence-transformers/'):
                    alt = 'sentence-transformers/' + self.model_name
                self._tok = AutoTokenizer.from_pretrained(alt, **tok_kwargs)
                self._model = AutoModel.from_pretrained(alt, use_auth_token=(token if token else None))
                self.model_name = alt
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load tokenizer/model for '{self.model_name}'. "
                    "If this is a private or gated HF repo, authenticate with 'huggingface-cli login' "
                    "or set HUGGINGFACE_HUB_TOKEN. Ensure model id is correct or install matching versions. "
                    f"Errors: {e}; {e2}"
                ) from e2

        # ensure model on CPU
        try:
            self._model.to('cpu')
        except Exception:
            pass

        cfg = getattr(self._model, 'config', None)
        self.dim = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'dim', None) or 768
        self.backend = 'transformers'

    def embed_texts(self, texts, batch_size: int = 32):
        if not texts:
            return np.zeros((0, self.dim), dtype='float32')

        if self.backend == 'sentence_transformers':
            emb = self._st_model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
            emb = emb.astype('float32')
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            emb = emb / norms
            return emb

        # transformers fallback
        import torch
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self._tok(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                out = self._model(**{k: v for k, v in encoded.items()})
                hidden = getattr(out, 'last_hidden_state', None)
                if hidden is None:
                    if isinstance(out, (tuple, list)) and len(out) > 0:
                        hidden = out[0]
                    else:
                        raise RuntimeError('Model output missing last_hidden_state')
                mask = encoded.get('attention_mask')
                if mask is None:
                    emb = hidden.mean(dim=1)
                else:
                    mask = mask.unsqueeze(-1)
                    summed = (hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1e-9)
                    emb = summed / denom
                emb = emb.cpu().numpy()
                all_embs.append(emb.astype('float32'))

        emb = np.vstack(all_embs)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norms
        return emb

