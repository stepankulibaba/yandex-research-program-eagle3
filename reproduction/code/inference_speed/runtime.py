"""Standard-library integrity, process ownership and benchmark contracts."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

SCHEMA = 3
BENCHES = ('mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'sum')
SYSTEM = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
          "  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
          "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
          "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't "
          "know the answer to a question, please don't share false information.")
EAGLE_COMMIT = 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b'
NEMO_COMMIT = '3770b1e199711ff8bef2c8616ef60c1fdef4a82e'
MODELS = {
    'target': ('unsloth/Meta-Llama-3.1-8B-Instruct', 'a2856192dd7c25b842431f39c179a6c2c2f627d1'),
    'draft': ('yuhuili/EAGLE3-LLaMA3.1-Instruct-8B', 'ada412b672e293d682423de84a095447bf38a637'),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def positive(value, label, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label}: not a finite number')
    if value < 0 or (not zero and value == 0):
        raise ValueError(f'{label}: invalid {value}')
    return value


def integer(value, label, zero=False):
    positive(value, label, zero)
    if int(value) != value:
        raise ValueError(f'{label}: not an integer')
    return int(value)


def questions(repo, bench, count=80):
    if bench not in BENCHES:
        raise ValueError('Unknown benchmark')
    rows = [json.loads(line) for line in (Path(repo) / 'eagle/data' / bench / 'question.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not 0 < count <= len(rows):
        raise ValueError('Invalid question count')
    rows = rows[:count]
    if len({r['question_id'] for r in rows}) != count:
        raise ValueError('Duplicate question IDs')
    return rows


def validate_answers(path, expected, kind):
    if kind == 'eagle':
        rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        rows = read_json(path)
    expected_ids = {r['question_id']: len(r['turns']) for r in expected}
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError('Incomplete question coverage')
    seen = set()
    for r in rows:
        qid = r['question_id']
        if qid in seen or qid not in expected_ids:
            raise ValueError('Duplicate or unexpected question ID')
        seen.add(qid)
        if kind == 'eagle':
            if len(r['choices']) != 1:
                raise ValueError('Expected exactly one choice')
            c = r['choices'][0]
            for key in ('turns', 'wall_time', 'new_tokens', 'idxs'):
                if len(c[key]) != expected_ids[qid]:
                    raise ValueError('Incomplete turns')
            for seconds, tokens, idx in zip(c['wall_time'], c['new_tokens'], c['idxs']):
                positive(seconds, 'seconds')
                integer(tokens, 'tokens')
                integer(idx, 'idx', zero=True)
            if not all(isinstance(t, str) for t in c['turns']):
                raise ValueError('Invalid text')
        else:
            if len(r['turns']) != expected_ids[qid]:
                raise ValueError('Incomplete turns')
            for t in r['turns']:
                positive(t['seconds'], 'seconds')
                integer(t['tokens'], 'tokens')
                for key in ('steps', 'accepted_draft_tokens'):
                    if t.get(key) is not None:
                        integer(t[key], key, zero=True)
                if not isinstance(t.get('prompt_ids'), list) or not t['prompt_ids']:
                    raise ValueError('Missing prompt IDs')
                if not isinstance(t.get('text'), str) or not t.get('finish_reason'):
                    raise ValueError('Missing response audit data')
    return rows


def validate_speed(path):
    r = read_json(path)
    if r.get('schema') != SCHEMA or r.get('protocol') != 'matched-bf16-fixed2048-v1':
        raise ValueError('Unknown training protocol')
    for key in ('seconds', 'optimizer_steps', 'nonpad_tokens', 'padded_tokens', 'supervised_tokens'):
        positive(r[key], key)
    if r['padded_tokens'] < r['nonpad_tokens'] or r['nonpad_tokens'] < r['supervised_tokens']:
        raise ValueError('Inconsistent token counts')
    if r.get('finite_loss') is not True or r.get('complete') is not True:
        raise ValueError('Incomplete/nonfinite training')
    return r


def done_path(path):
    return Path(str(path) + '.done.json')


def finished(path, manifest, validator):
    try:
        mark = read_json(done_path(path))
        if mark['schema'] != SCHEMA or mark['fingerprint'] != digest(manifest) or mark['sha256'] != file_hash(path):
            return False
        validator(path)
        return True
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return False


def finish(path, manifest, validator):
    validator(path)
    atomic_json(done_path(path), {'schema': SCHEMA, 'fingerprint': digest(manifest), 'manifest': manifest,
                                'sha256': file_hash(path), 'finished_at': time.time()})


def verified_artifact(path, validator):
    """Readers require a successful-process marker AND an unchanged artifact."""
    m = read_json(done_path(path))
    if m.get('schema') != SCHEMA or m['sha256'] != file_hash(path) or m['fingerprint'] != digest(m['manifest']):
        raise ValueError(f'Stale/uncommitted artifact: {path}')
    validator(path)
    return m['manifest']


def terminate_owned(proc, grace=10):
    if proc is None:
        return
    # Popen(start_new_session=True) establishes this group's ownership, even when
    # its leader exits before workers. Never discover/kill arbitrary port owners.
    def send(sig):
        with contextlib.suppress(ProcessLookupError):
            if os.name == 'posix':
                os.killpg(proc.pid, sig)
            elif proc.poll() is None:
                proc.terminate() if sig == signal.SIGTERM else proc.kill()
    send(signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A leader's exit does not mean the worker group has exited.
    if os.name == 'posix':
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
    send(getattr(signal, 'SIGKILL', signal.SIGTERM))
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def run_process(cmd, *, cwd, log, timeout, env=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen([str(x) for x in cmd], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=(os.name == 'posix'))
        try:
            rc = proc.wait(timeout=timeout)
            if rc:
                raise subprocess.CalledProcessError(rc, cmd)
        finally:
            terminate_owned(proc)


class BenchmarkWindow:
    """GPU synchronization only at the two window boundaries; no inferred tokens."""
    def __init__(self, warmup, expected_steps, framework, config):
        if warmup < 0 or expected_steps <= warmup:
            raise ValueError('Not enough optimizer steps after warm-up')
        self.warmup, self.expected_steps = warmup, expected_steps
        self.framework, self.config = framework, config
        self.start = self.end = None
        self.steps = self.nonpad = self.padded = self.supervised = self.documents = 0
        self.finite_loss = True
        self.rows = 0
        self.length_histogram = {'0-512': 0, '513-1024': 0, '1025-1900': 0}

    def before_batch(self, optimizer_step, batch, torch):
        if optimizer_step < self.warmup:
            return
        if self.start is None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()
        # Inputs are still on CPU in both adapters. Counts are exact, including
        # the final partial accumulation window and packed document boundaries.
        self.nonpad += int(batch['attention_mask'].sum())
        self.padded += batch['input_ids'].numel()
        self.supervised += int(batch['loss_mask'].sum())
        self.rows += batch['input_ids'].shape[0]
        lengths = batch.get('_document_lengths')
        if lengths is None:
            lengths = batch['attention_mask'].sum(dim=1).tolist()
        for length in lengths:
            self.documents += 1
            bucket = '0-512' if length <= 512 else ('513-1024' if length <= 1024 else '1025-1900')
            self.length_histogram[bucket] += 1

    def after_step(self, completed_steps, torch):
        if completed_steps > self.warmup:
            self.steps += 1
        if completed_steps == self.expected_steps:
            torch.cuda.synchronize()
            self.end = time.perf_counter()

    def result(self, torch):
        if self.start is None or self.end is None or not self.finite_loss:
            raise ValueError('Incomplete timing window or nonfinite loss')
        return {'schema': SCHEMA, 'protocol': 'matched-bf16-fixed2048-v1', 'framework': self.framework,
                'complete': True, 'finite_loss': True, 'seconds': self.end - self.start,
                'optimizer_steps': self.steps, 'warmup_optimizer_steps': self.warmup,
                'nonpad_tokens': self.nonpad, 'padded_tokens': self.padded, 'supervised_tokens': self.supervised,
                'documents': self.documents, 'rows': self.rows, 'length_histogram': self.length_histogram,
                'peak_allocated_gb': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2**30, 'config': self.config}
