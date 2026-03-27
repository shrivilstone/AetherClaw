from .ollama_client import OllamaClient
from .hf_client import HuggingFaceClient
from .retriever import Retriever
from .memory_store import FileMemory
from .memory_tail import MemoryTail
from .assistant_filing import AssistantFiling
import os

DEFAULT_MODEL = 'hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M'


class RAGPipeline:
    def __init__(self, model: str = DEFAULT_MODEL, resources_dir: str = None, working_memory_path: str = None):
        # Calculate dynamic absolute paths relative to AetherClaw's new location inside Obsidian Vault
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vault_dir = os.path.abspath(os.path.join(base_dir, '../../'))
        
        resources_dir = resources_dir or os.path.join(vault_dir, 'Resources')
        working_memory_path = working_memory_path or os.path.join(vault_dir, 'WorkingMemory.md')
        # choose client: HF model ids contain '/' (e.g., 'Tesslate/OmniCoder-9B') or may be prefixed with 'hf:'
        if isinstance(model, str) and ('/' in model or (model.startswith('hf:'))):
            if model.startswith('hf.co/'):
                self.client = OllamaClient(model=model)
            else:
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
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        
        # IMPROVED MEMORY SYSTEM: Create a logical "transition summary"
        # Format: [User: <brief goal> -> Assistant: <brief action/result>]
        try:
            # Generate a very short summary of the response using a regex or simple split
            import re
            first_sentence = re.split(r'(?<=[.!?]) +', resp.strip())[0]
            short_resp = first_sentence[:100] + "..." if len(first_sentence) > 100 else first_sentence
            short_query = query[:50] + "..." if len(query) > 50 else query
            
            transition_summary = f"[User: {short_query} -> Assistant: {short_resp}]"
        except Exception:
            transition_summary = f"[Interaction: {query[:30]}...]"

        # Append to tail and snapshots with the refined format
        try:
            self.tail.append(transition_summary, source='assistant')
        except Exception:
            pass
        try:
            self.mem.append_snapshot('transition', transition_summary)
        except Exception:
            pass

        # record short progress to Obsidian (Progress.md)
        try:
            if self.assistant:
                self.assistant.append_short_progress(transition_summary, label='memory_transition')
        except Exception:
            pass
        return resp

if __name__ == '__main__':
    p = RAGPipeline()
    print(p.ask('Summarize the active projects and suggest next actions.', k=2))
