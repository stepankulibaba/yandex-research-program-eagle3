"""Pinned snapshots validated against the Hub file manifest; no shard shortcut."""
import fnmatch
import json
from pathlib import Path
import struct
import sys
import time
from runtime import MODELS, SCHEMA, atomic_json, file_hash


def validate_files(directory, files):
    directory = Path(directory)
    for name, size in files.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f'Missing/empty snapshot file: {name}')
        if size is not None and path.stat().st_size != size:
            if name != 'tokenizer_config.json':
                raise ValueError(f'Truncated snapshot file: {name}')
            restored = path.read_text(encoding='utf-8').replace('"PreTrainedTokenizerFast"', '"PreTrainedTokenizer"')
            if len(restored.encode()) != size:
                raise ValueError('Unexpected tokenizer modification')
        if name.endswith('.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            if name.endswith('.index.json'):
                for shard in set(data['weight_map'].values()):
                    if shard not in files:
                        raise ValueError(f'Unrequested weight shard: {shard}')
        if name.endswith('.safetensors'):
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


def main(repo, directory, *patterns):
    from huggingface_hub import HfApi, snapshot_download
    revisions = dict(MODELS.values())
    if repo not in revisions:
        raise ValueError('Model revision must be pinned in runtime.py')
    revision = revisions[repo]
    # Offline fast path: a snapshot validated earlier at this revision needs no Hub call
    # (the connection to the Hub drops at night; a failed check must not stop the run).
    try:
        manifest = json.loads((Path(directory) / 'snapshot_manifest.json').read_text(encoding='utf-8'))
        if manifest['schema'] == SCHEMA and manifest['repo'] == repo and manifest['revision'] == revision:
            validate_files(directory, manifest['files'])
            print(f'Already validated {repo}@{revision}: {len(manifest["files"])} files (local check)', flush=True)
            return
    except (OSError, ValueError, KeyError):
        pass
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
            p = Path(directory) / 'tokenizer_config.json'
            if p.exists() and json.loads(p.read_text(encoding='utf-8')).get('tokenizer_class') == 'PreTrainedTokenizer':
                p.write_text(p.read_text(encoding='utf-8').replace('"PreTrainedTokenizer"', '"PreTrainedTokenizerFast"'),
                             encoding='utf-8', newline='\n')
            atomic_json(Path(directory) / 'snapshot_manifest.json', {
                'schema': SCHEMA, 'repo': repo, 'revision': revision, 'files': files,
                'json_hashes': {n: file_hash(Path(directory) / n) for n in files if n.endswith('.json')},
                'tokenizer_class_patch': True})
            print(f'Validated {repo}@{revision}: {len(files)} files', flush=True)
            return
        except Exception as exc:
            error = exc
            print(f'Download attempt {attempt}: {type(exc).__name__}: {exc}', flush=True)
            if attempt < 5:
                time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f'Pinned snapshot download failed: {repo}') from error


if __name__ == '__main__':
    main(*sys.argv[1:])
