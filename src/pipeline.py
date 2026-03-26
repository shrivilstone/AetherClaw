from .ollama_client import OllamaClient
from .hf_client import HuggingFaceClient
from .retriever import Retriever
from .memory_store import FileMemory
from .memory_tail import MemoryTail
from .assistant_filing import AssistantFiling
import os

DEFAULT_MODEL = 'qwen3.5:9b'


class RAGPipeline:
    def __init__(self, model: str = DEFAULT_MODEL, resources_dir: str = '~/ObsidianVaults/ProjectsVault/Resources', working_memory_path: str = '~/ObsidianVaults/ProjectsVault/WorkingMemory.md'):
        # choose client: HF model ids contain '/' (e.g., 'Tesslate/OmniCoder-9B') or may be prefixed with 'hf:'
        if isinstance(model, str) and ('/' in model or (model.startswith('hf:'))):
            # normalize 'hf:repo/name' -> 'repo/name'
            model_id = model[3:] if model.startswith('hf:') else model
            try:
                self.client = HuggingFaceClient(model_id)
            except Exception:
                # fallback to Ollama if HF client fails to initialize
                self.client = OllamaClient(model=model)
        else:
            self.client = OllamaClient(model=model)
        # use hybrid mode (dense FAISS + BM25 fallback). If dependencies missing, Retriever falls back to naive.
        res_dir_expanded = os.path.expanduser(resources_dir)
        index_path = os.path.join(res_dir_expanded, 'faiss.index')
        self.store = Retriever(resources_dir, mode='hybrid', index_path=index_path)
        self.mem = FileMemory(working_memory_path)
        self.tail = MemoryTail(working_memory_path)
        try:
            self.assistant = AssistantFiling()
        except Exception:
            self.assistant = None

    def build_context(self, query: str, k: int = 3) -> str:
        # recent short-term tail + persistent snapshots
        tail_text = self.tail.read_tail()
        mem_text = self.mem.read()
        retrieved = self.store.query(query, k=k)
        snippets = []
        for path, text, score in retrieved:
            snippets.append(f'From {path} (score={score:.3f}):\n' + text[:2000])
        ctx = 'Working memory (tail):\n' + (tail_text or '(none)') + '\n\nSnapshots:\n' + (mem_text[:4000] or '(none)') + '\n\nRetrieved passages:\n' + '\n\n'.join(snippets)
        return ctx

    def ask(self, query: str, k: int = 3, max_tokens: int = 512) -> str:
        ctx = self.build_context(query, k=k)
        prompt = f"System: You are a helpful assistant. Use the context below when answering.\n\n{ctx}\n\nUser: {query}\n\nAnswer concisely and list next actions."
        resp = self.client.run(prompt, max_tokens=max_tokens)
        # create a short one-line summary to keep tail small
        try:
            import re
            m = re.search(r"(.+?[\.!?])(\s|$)", resp.strip())
            if m:
                summary = m.group(1).strip()
            else:
                summary = resp.strip().splitlines()[0][:200]
        except Exception:
            summary = (resp or '').strip()[:200]

        # append a concise tail entry and short snapshot (avoid writing full response into WorkingMemory)
        try:
            self.tail.append(summary, source='assistant')
        except Exception:
            pass
        try:
            self.mem.append_snapshot('response_summary', f'Q: {query}\n\nA_summary: {summary}')
        except Exception:
            pass

        # record short progress to Obsidian (Progress.md) to avoid overwriting tail
        try:
            if self.assistant:
                self.assistant.append_short_progress(summary, label='response')
        except Exception:
            pass
        return resp

if __name__ == '__main__':
    p = RAGPipeline()
    print(p.ask('Summarize the active projects and suggest next actions.', k=2))
