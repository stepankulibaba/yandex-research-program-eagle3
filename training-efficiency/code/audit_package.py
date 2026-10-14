"""Audit only staged additions/changes in this contribution; never print secrets."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DIRECTION = ROOT / 'training-efficiency'
MAX_FILE_BYTES = 1024 * 1024
MAX_PACKAGE_BYTES = 4 * 1024 * 1024
SAFE_SUFFIXES = {'.py', '.json', '.md', '.png', '.txt'}
PRIVATE_COMPONENTS = {'data', 'models', 'hf_cache', 'pip_cache', 'env', '.venv', '__pycache__',
                      'Automodel', 'EAGLE', 'references', 'vendor', 'remote', 'snapshots'}
SENSITIVE_TEXT = {
    'private_absolute_path': r'/Users/|/home/|(?<![A-Za-z])[A-Z]:[/\\]',
    'private_mesh_address': r'\b100\.(?:\d{1,3}\.){2}\d{1,3}\b',
    'private_address': r'\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b',
    'ssh_material': r'BEGIN [A-Z ]*PRIVATE KEY|ssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/]{30}',
    'service_token': r'\b(?:hf_|ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9]{20,}\b|github_pat_[A-Za-z0-9_]{30,}',
    'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:com|ru|org|net)\b',
    'personal_infrastructure': r'gpu-playbook|tailscale|id_ed25519',
}

def staged_files():
    output = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'], cwd=ROOT)
    return [name for name in output.decode().split('\0') if name]

def audit():
    errors = []
    files = staged_files()
    assert files, 'No staged package to audit'
    total = 0
    for name in files:
        path = Path(name)
        if name != 'README.md' and path.parts[0] != 'training-efficiency':
            errors.append((name, 'outside contribution'))
        if any(part in PRIVATE_COMPONENTS or part.startswith('_') for part in path.parts):
            errors.append((name, 'excluded directory'))
        if path.name != '.gitignore' and path.suffix not in SAFE_SUFFIXES:
            errors.append((name, 'unapproved file type'))
        blob = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        total += len(blob)
        if len(blob) > MAX_FILE_BYTES:
            errors.append((name, 'oversized file'))
        if path.suffix == '.png':
            assert blob.startswith(b'\x89PNG\r\n\x1a\n'), name
            continue
        text = blob.decode('utf-8')
        if path.name != 'audit_package.py':
            for label, pattern in SENSITIVE_TEXT.items():
                if re.search(pattern, text, re.IGNORECASE):
                    errors.append((name, label))
        if path.suffix == '.json':
            json.loads(text)
    if total > MAX_PACKAGE_BYTES:
        errors.append(('package', 'oversized package'))
    # Root README changes must be exactly the new direction's row.
    before = subprocess.check_output(['git', 'show', 'HEAD:README.md'], cwd=ROOT).decode()
    after = subprocess.check_output(['git', 'show', ':README.md'], cwd=ROOT).decode()
    added = [line for line in after.splitlines(keepends=True) if '](training-efficiency/)' in line]
    if len(added) != 1 or after.replace(added[0], '', 1) != before:
        errors.append(('README.md', 'unrelated root edits'))
    assert not errors, '\n'.join(f'{name}: {kind}' for name, kind in errors)
    print(json.dumps({'staged_files': len(files), 'total_bytes': total, 'forbidden_files': 0,
                      'sensitive_text_matches': 0, 'root_readme_new_row_only': True}, ensure_ascii=False))

if __name__ == '__main__':
    audit()
