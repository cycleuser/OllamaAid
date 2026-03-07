#!/usr/bin/env bash
# OllamaAid - Build and upload to PyPI
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="C:/Miniconda3/envs/dev/python.exe"
VERSION_FILE="ollama_aid/__version__.py"

echo "=== OllamaAid PyPI Upload ==="

echo "[1/6] Bumping patch version..."
"$PYTHON" -c "
import re, sys
p = '$VERSION_FILE'
t = open(p).read()
m = re.search(r'(__version__\s*=\s*\"(\d+\.\d+\.)(\d+)\")', t)
if not m: print('ERROR: cannot parse version'); sys.exit(1)
old_v = m.group(2) + m.group(3)
new_v = m.group(2) + str(int(m.group(3)) + 1)
open(p, 'w').write(t.replace(m.group(1), '__version__ = \"' + new_v + '\"'))
print(f'  {old_v} -> {new_v}')
"

echo "[2/6] Cleaning old builds..."
rm -rf dist/ build/ *.egg-info ollama_aid.egg-info

echo "[3/6] Installing build tools..."
"$PYTHON" -m pip install --upgrade build twine -q

echo "[4/6] Building package..."
"$PYTHON" -m build
"$PYTHON" -m twine check dist/*

echo "[5/6] Uploading to PyPI..."
"$PYTHON" -m twine upload dist/*

echo "[6/6] Committing and pushing to GitHub..."
NEW_VER=$("$PYTHON" -c "import re; t=open('$VERSION_FILE').read(); m=re.search(r'__version__\s*=\s*\"([\d.]+)\"', t); print(m.group(1))")
git add -A
git commit -m "Release v${NEW_VER}"
git tag "v${NEW_VER}"
git push origin main --tags

echo "=== Done! Released v${NEW_VER} ==="
