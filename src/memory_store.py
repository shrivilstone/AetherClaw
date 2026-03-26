import os
from datetime import datetime

class FileMemory:
    def __init__(self, path: str):
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append_snapshot(self, label: str, content: str):
        ts = datetime.utcnow().isoformat() + 'Z'
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write('\n---\n')
            f.write(f'**Snapshot** {ts} — {label}\n\n')
            f.write(content)
            f.write('\n')

    def read(self):
        if not os.path.exists(self.path):
            return ''
        with open(self.path, 'r', encoding='utf-8') as f:
            return f.read()

if __name__ == '__main__':
    fm = FileMemory('~/ObsidianVaults/ProjectsVault/WorkingMemory.md')
    fm.append_snapshot('test', 'This is a test')
    print('Appended')
