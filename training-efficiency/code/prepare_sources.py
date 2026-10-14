"""Fetch only pinned upstream source files; never download model weights or data."""
import argparse
import hashlib
import json
import ssl
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / 'code'
try:
    import certifi
    TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    TLS_CONTEXT = ssl.create_default_context()

def prepare(verify_only=False):
    count = 0
    for name, repo in [('nemo', 'NVIDIA-NeMo/Automodel'), ('eagle', 'SafeAILab/EAGLE')]:
        manifest = json.loads((CODE / f'{name}_hashes.json').read_text())
        for relative, expected in manifest['files'].items():
            if name == 'nemo':
                destination = ROOT / 'Automodel' / relative
                upstream_path = relative
            else:
                destination = ROOT / relative
                upstream_path = relative.removeprefix('EAGLE/')
            if not destination.exists():
                if verify_only:
                    raise FileNotFoundError(f'Missing pinned source: {relative}')
                url = f'https://raw.githubusercontent.com/{repo}/{manifest["commit"]}/{upstream_path}'
                with urlopen(url, timeout=30, context=TLS_CONTEXT) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f'Upstream SHA mismatch: {relative}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Local SHA mismatch: {relative}')
            count += 1
        if name == 'nemo' and not verify_only:
            (ROOT / 'Automodel/source_hashes.json').write_text(json.dumps(manifest['files'], indent=2) + '\n')
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(f'Pinned upstream sources verified: {prepare(args.verify_only)}')
