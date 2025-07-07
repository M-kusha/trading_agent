# print_clean_tree.py
import os

# Directories to exclude from the printed tree
EXCLUDE_DIRS = {
    '.git', '.vscode', '__pycache__', 'node_modules', 'dist', 'logs', 'checkpoints',
    'data', 'tests', 'frontend/node_modules', 'env', 'venv', 'site-packages',
    'modules.zip', 'envs.zip', '.idea', '.mypy_cache', '.pytest_cache',
    '.ipynb_checkpoints', '.eggs'
}

# Files to exclude from the printed tree
EXCLUDE_FILES = {
    '.DS_Store', '*.pyc', '*.log', '*.jsonl', '*.pkl', '*.zip', '*.egg-info',
    '*.whl', '*.tar.gz'
}

# Helper to check if a filename should be excluded
def should_exclude(name):
    if name in EXCLUDE_DIRS:
        return True
    for pat in EXCLUDE_FILES:
        if pat.startswith('*'):
            if name.endswith(pat[1:]):
                return True
        elif pat.endswith('*'):
            if name.startswith(pat[:-1]):
                return True
        elif name == pat:
            return True
    return False

def print_tree(startpath, prefix=""):
    entries = [e for e in os.listdir(startpath) if not should_exclude(e)]
    entries.sort()
    for i, entry in enumerate(entries):
        path = os.path.join(startpath, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print(os.path.basename(os.getcwd()) + "/")
    print_tree(".")
