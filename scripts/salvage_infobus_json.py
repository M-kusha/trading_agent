import os
import json
import shutil
import sys

PATH = os.path.abspath(os.path.join(os.getcwd(), 'infobus_data.json'))

def salvage_truncated_json(content: str):
    try:
        return json.loads(content)
    except Exception:
        pass
    depth = 0
    in_str = False
    esc = False
    last_balanced = -1
    for i, ch in enumerate(content):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth = max(0, depth - 1)
                if depth == 0:
                    last_balanced = i
    if last_balanced >= 0:
        snippet = content[: last_balanced + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

def atomic_write_json(path: str, data: dict):
    payload = json.dumps(data, indent=2)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        try:
            shutil.copy2(path, path + '.bak')
        except Exception:
            pass
    os.replace(tmp, path)

def main():
    if not os.path.exists(PATH):
        print('No infobus_data.json found')
        return 0
    try:
        with open(PATH, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        data = salvage_truncated_json(content)
        if data is None:
            print('Unable to salvage JSON')
            return 2
        atomic_write_json(PATH, data)
        print('Repaired infobus_data.json successfully')
        return 0
    except Exception as e:
        print('Repair failed:', e)
        return 3

if __name__ == '__main__':
    sys.exit(main())

