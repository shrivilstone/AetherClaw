import os
from datetime import datetime
from typing import Optional


class MemoryTail:
    """A compact short-term working-memory (the "tail").

    - Keeps a short in-memory buffer of recent items.
    - Rolls up older items into the persistent WorkingMemory.md file.
    - Optionally uses a summarizer (an LLM client) to compress rollups.
    """
    def __init__(self, path: str, short_term_max_items: int = 200, rollup_threshold: int = 150):
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.short_term = []
        self.max_items = short_term_max_items
        self.rollup_threshold = rollup_threshold

    def append(self, text: str, source: Optional[str] = None):
        ts = datetime.utcnow().isoformat() + 'Z'
        entry = {'ts': ts, 'text': text, 'source': source}
        self.short_term.append(entry)
        if len(self.short_term) >= self.rollup_threshold:
            self.rollup_to_file()

    def rollup_to_file(self, summarizer=None):
        # take the oldest half and roll them up (summarize if possible)
        n = max(1, len(self.short_term) // 2)
        to_roll = self.short_term[:n]
        texts = [f"{e['ts']} — {e.get('source','agent')}: {e['text']}" for e in to_roll]
        if summarizer:
            try:
                summary = summarizer.run("Summarize concisely the following snippets:\n\n" + "\n\n".join(texts), max_tokens=200)
            except Exception:
                summary = "\n".join(texts)
        else:
            summary = "\n".join(texts)

        with open(self.path, 'a', encoding='utf-8') as f:
            f.write('\n---\n')
            f.write(f'**Tail rollup** {datetime.utcnow().isoformat()}Z\n\n')
            f.write(summary)
            f.write('\n')

        # drop rolled items
        self.short_term = self.short_term[n:]

    def read_tail(self, limit_chars: int = 4000) -> str:
        entries = self.short_term[-50:]
        texts = [f"{e['ts']} — {e.get('text','')}" for e in entries]
        s = '\n'.join(texts)
        if len(s) > limit_chars:
            return s[-limit_chars:]
        return s
