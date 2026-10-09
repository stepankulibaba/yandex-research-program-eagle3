"""Download a model at its pinned revision (common.MODELS) and check that every file arrived whole.

    python download.py <repo> <local dir> <pattern> [<pattern> ...]      e.g. '*.json' '*.safetensors'

- A model already validated at this revision (snapshot_manifest.json) is only re-checked locally, without the
  Hub: the connection to the Hub drops at night and must not stop the run.
- Otherwise: list the files on the Hub, download, check sizes, shard indexes and safetensors headers;
  up to 5 attempts.
- The open mirror unsloth/Meta-Llama-3.1-8B-Instruct names the tokenizer class "PreTrainedTokenizer"; Meta's
  original says "PreTrainedTokenizerFast" (same tokenizer.json). The authors' scripts load it with use_fast=False
  and get no tokenizer from the mirror's value, so it is set back to Meta's.
"""
import fnmatch
import json
from pathlib import Path
import struct
import sys
import time

from common import MODELS, SCHEMA, atomic_json, file_hash

TOKENIZER_FIX = ('"PreTrainedTokenizer"', '"PreTrainedTokenizerFast"')


def validate_files(directory, files):
    """files: {name: size on the Hub or None}. Raises ValueError on anything missing or truncated."""
    directory = Path(directory)
    for name, size in files.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f'Missing/empty snapshot file: {name}')
        if size is not None and path.stat().st_size != size:
            if name != 'tokenizer_config.json':
                raise ValueError(f'Truncated snapshot file: {name}')
            # Our tokenizer fix changes the size: undo it and compare again.
            restored = path.read_text(encoding='utf-8').replace(TOKENIZER_FIX[1], TOKENIZER_FIX[0])
            if len(restored.encode()) != size:
                raise ValueError('Unexpected tokenizer modification')
        if name.endswith('.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            if name.endswith('.index.json'):
                for shard in set(data['weight_map'].values()):
                    if shard not in files:
                        raise ValueError(f'Unrequested weight shard: {shard}')
        if name.endswith('.safetensors'):     # the header says how long the file must be
            with path.open('rb') as stream:
                header_size = struct.unpack('<Q', stream.read(8))[0]
                if not 0 < header_size < min(path.stat().st_size, 100_000_000):
                    raise ValueError('Invalid safetensors header')
                header = json.loads(stream.read(header_size))
            end = max(t['data_offsets'][1] for k, t in header.items() if k != '__metadata__')
            if end + header_size + 8 != path.stat().st_size:
                raise ValueError('Truncated safetensors payload')
    if 'config.json' not in files:
        raise ValueError('Missing config.json')


def already_valid(repo, revision, directory):
    try:
        manifest = json.loads((Path(directory) / 'snapshot_manifest.json').read_text(encoding='utf-8'))
        if manifest['schema'] == SCHEMA and manifest['repo'] == repo and manifest['revision'] == revision:
            validate_files(directory, manifest['files'])
            return manifest
    except (OSError, ValueError, KeyError):
        pass
    return None


def main(repo, directory, *patterns):
    from huggingface_hub import HfApi, snapshot_download
    revisions = dict(MODELS.values())
    if repo not in revisions:
        raise ValueError('Model revision must be pinned in common.py')
    revision = revisions[repo]

    manifest = already_valid(repo, revision, directory)
    if manifest:
        print(f'Already validated {repo}@{revision}: {len(manifest["files"])} files (local check)', flush=True)
        return

    error = None
    for attempt in range(1, 6):
        try:
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            files = {f.rfilename: f.size for f in info.siblings
                     if any(fnmatch.fnmatch(f.rfilename, p) for p in patterns)}
            if not files or not any(n.endswith(('.bin', '.safetensors')) for n in files):
                raise ValueError('No weights in requested snapshot')
            if repo == MODELS['target'][0] and not {'tokenizer.json', 'tokenizer_config.json'} <= files.keys():
                raise ValueError('Missing tokenizer in snapshot')
            snapshot_download(repo, revision=revision, local_dir=directory, allow_patterns=list(patterns))
            validate_files(directory, files)
            tokenizer_config = Path(directory) / 'tokenizer_config.json'
            if (tokenizer_config.exists() and json.loads(tokenizer_config.read_text(encoding='utf-8'))
                    .get('tokenizer_class') == 'PreTrainedTokenizer'):
                tokenizer_config.write_text(tokenizer_config.read_text(encoding='utf-8').replace(*TOKENIZER_FIX),
                                            encoding='utf-8', newline='\n')
            atomic_json(Path(directory) / 'snapshot_manifest.json', {
                'schema': SCHEMA, 'repo': repo, 'revision': revision, 'files': files,
                'json_hashes': {n: file_hash(Path(directory) / n) for n in files if n.endswith('.json')},
                'tokenizer_class_patch': True})
            print(f'Validated {repo}@{revision}: {len(files)} files', flush=True)
            return
        except Exception as exc:        # network errors of every kind; the next attempt resumes partial files
            error = exc
            print(f'Download attempt {attempt}: {type(exc).__name__}: {exc}', flush=True)
            if attempt < 5:
                time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f'Pinned snapshot download failed: {repo}') from error


if __name__ == '__main__':
    main(*sys.argv[1:])
