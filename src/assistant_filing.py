import os
from datetime import datetime


class AssistantFiling:
    """Simple file-backed filing system for the assistant inside the Obsidian vault.

    Files created under: ~/ObsidianVaults/ProjectsVault/Assistant
    - Progress.md (chronological snapshots)
    - Inbox.md (incoming tasks / items)
    - Thoughts.md (brief ephemeral notes)
    """
    def __init__(self, vault_path: str = '~/ObsidianVaults/ProjectsVault/Assistant'):
        self.base = os.path.expanduser(vault_path)
        os.makedirs(self.base, exist_ok=True)
        self.progress_path = os.path.join(self.base, 'Progress.md')
        self.inbox_path = os.path.join(self.base, 'Inbox.md')
        self.thoughts_path = os.path.join(self.base, 'Thoughts.md')

    def _ts(self):
        return datetime.utcnow().isoformat() + 'Z'

    def append_progress(self, text: str, label: str = 'note') -> None:
        ts = self._ts()
        with open(self.progress_path, 'a', encoding='utf-8') as f:
            f.write('\n---\n')
            f.write(f'**{label}** {ts}\n\n')
            f.write(text)
            f.write('\n')

    def append_short_progress(self, text: str, label: str = 'note', max_len: int = 240) -> None:
        """Append a very short single-line progress entry (keeps tail small)."""
        ts = self._ts()
        one_line = ' '.join(text.splitlines())
        if len(one_line) > max_len:
            one_line = one_line[:max_len].rstrip() + '…'
        with open(self.progress_path, 'a', encoding='utf-8') as f:
            f.write('\n---\n')
            f.write(f'**{label}** {ts}\n\n')
            f.write(one_line)
            f.write('\n')

    def add_inbox(self, title: str, text: str) -> None:
        ts = self._ts()
        with open(self.inbox_path, 'a', encoding='utf-8') as f:
            f.write('\n---\n')
            f.write(f'**{title}** {ts}\n\n')
            f.write(text)
            f.write('\n')

    def add_thought(self, text: str) -> None:
        ts = self._ts()
        with open(self.thoughts_path, 'a', encoding='utf-8') as f:
            f.write('\n---\n')
            f.write(f'{ts}\n\n')
            f.write(text)
            f.write('\n')
