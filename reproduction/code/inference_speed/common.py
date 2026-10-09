"""Shared pieces of the speed package: pinned versions, file helpers, "done" markers and child processes.

Only the standard library is used here, so every script (and every virtualenv) can import this module.
"""
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

# --------------------------------------------------------------------------------------------------------------------
# What exactly is measured. Bump SCHEMA whenever a change alters a measurement: results with another SCHEMA are
# never reused. Pure refactoring (comments, names) does not need a bump.
# --------------------------------------------------------------------------------------------------------------------
SCHEMA = 4

BENCHES = ('mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'sum')

# System prompt of the authors' evaluation script (gen_ea_answer_llama3chat.py); SGLang requests reuse it.
SYSTEM = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
          "  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
          "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
          "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't "
          "know the answer to a question, please don't share false information.")

# Pinned code and models (all checked to exist on GitHub / Hugging Face).
EAGLE_COMMIT = 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b'      # SafeAILab/EAGLE
NEMO_COMMIT = '3770b1e199711ff8bef2c8616ef60c1fdef4a82e'       # NVIDIA-NeMo/Automodel
MODELS = {
    'target': ('unsloth/Meta-Llama-3.1-8B-Instruct', 'a2856192dd7c25b842431f39c179a6c2c2f627d1'),
    'draft': ('yuhuili/EAGLE3-LLaMA3.1-Instruct-8B', 'ada412b672e293d682423de84a095447bf38a637'),
}

# Training-speed protocol name, written into every training result.
TRAINING_PROTOCOL = 'matched-bf16-fixed2048-v1'


# --------------------------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_text(path, text):
    """Write a file so that readers see either the old or the complete new content, never a half-written one."""
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
    """Like atomic_text; NaN/inf are refused (allow_nan=False), so a broken number never lands in a result."""
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(value):
    """Stable sha256 of any JSON-serialisable value (key order does not matter)."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------------------
# Number checks used by the validators
# --------------------------------------------------------------------------------------------------------------------
def positive(value, label, zero=False):
    """A finite number > 0 (or >= 0 with zero=True); bools are rejected."""
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


# --------------------------------------------------------------------------------------------------------------------
# Benchmark questions and answer files
# --------------------------------------------------------------------------------------------------------------------
def questions(eagle_repo, bench, count=80):
    """The first `count` questions of an EAGLE benchmark (eagle/data/<bench>/question.jsonl)."""
    if bench not in BENCHES:
        raise ValueError('Unknown benchmark')
    path = Path(eagle_repo) / 'eagle/data' / bench / 'question.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not 0 < count <= len(rows):
        raise ValueError('Invalid question count')
    rows = rows[:count]
    if len({r['question_id'] for r in rows}) != count:
        raise ValueError('Duplicate question IDs')
    return rows


def validate_answers(path, expected, kind):
    """Check that an answer file covers every expected question and turn, with sane numbers.

    kind='eagle':  JSONL written by the authors' scripts (choices[0] with turns/wall_time/new_tokens/idxs).
    kind='sglang': JSON list written by sglang_client.py.
    """
    if kind == 'eagle':
        rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        rows = read_json(path)
    turns_per_question = {r['question_id']: len(r['turns']) for r in expected}
    if not isinstance(rows, list) or len(rows) != len(turns_per_question):
        raise ValueError('Incomplete question coverage')
    seen = set()
    for row in rows:
        qid = row['question_id']
        if qid in seen or qid not in turns_per_question:
            raise ValueError('Duplicate or unexpected question ID')
        seen.add(qid)
        n_turns = turns_per_question[qid]
        if kind == 'eagle':
            if len(row['choices']) != 1:
                raise ValueError('Expected exactly one choice')
            choice = row['choices'][0]
            for key in ('turns', 'wall_time', 'new_tokens', 'idxs'):
                if len(choice[key]) != n_turns:
                    raise ValueError('Incomplete turns')
            for seconds, tokens, idx in zip(choice['wall_time'], choice['new_tokens'], choice['idxs']):
                positive(seconds, 'seconds')
                integer(tokens, 'tokens')
                integer(idx, 'idx', zero=True)
            if not all(isinstance(t, str) for t in choice['turns']):
                raise ValueError('Invalid text')
        else:
            if len(row['turns']) != n_turns:
                raise ValueError('Incomplete turns')
            for turn in row['turns']:
                positive(turn['seconds'], 'seconds')
                integer(turn['tokens'], 'tokens')
                for key in ('steps', 'accepted_draft_tokens'):
                    if turn.get(key) is not None:
                        integer(turn[key], key, zero=True)
                if not isinstance(turn.get('prompt_ids'), list) or not turn['prompt_ids']:
                    raise ValueError('Missing prompt IDs')
                if not isinstance(turn.get('text'), str) or not turn.get('finish_reason'):
                    raise ValueError('Missing response audit data')
    return rows


def validate_speed(path):
    """Check a training-speed result written by training_common.SpeedWindow."""
    r = read_json(path)
    if r.get('schema') != SCHEMA or r.get('protocol') != TRAINING_PROTOCOL:
        raise ValueError('Unknown training protocol')
    for key in ('seconds', 'optimizer_steps', 'nonpad_tokens', 'padded_tokens', 'supervised_tokens'):
        positive(r[key], key)
    if r['padded_tokens'] < r['nonpad_tokens'] or r['nonpad_tokens'] < r['supervised_tokens']:
        raise ValueError('Inconsistent token counts')
    if r.get('finite_loss') is not True or r.get('complete') is not True:
        raise ValueError('Incomplete/nonfinite training')
    return r


# --------------------------------------------------------------------------------------------------------------------
# "Done" markers: <result>.done.json next to each result
#
# A result counts as finished only if its marker exists, was written for the same settings (manifest fingerprint)
# and the result file is unchanged since (sha256). Then a restart skips that stage.
# --------------------------------------------------------------------------------------------------------------------
def done_path(path):
    return Path(str(path) + '.done.json')


def finished(path, manifest, validator):
    """True if `path` is a complete result for exactly these settings."""
    try:
        mark = read_json(done_path(path))
        if (mark['schema'] != SCHEMA or mark['fingerprint'] != digest(manifest)
                or mark['sha256'] != file_hash(path)):
            return False
        validator(path)
        return True
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return False


def finish(path, manifest, validator):
    """Validate a fresh result and write its marker."""
    validator(path)
    atomic_json(done_path(path), {'schema': SCHEMA, 'fingerprint': digest(manifest), 'manifest': manifest,
                                  'sha256': file_hash(path), 'finished_at': time.time()})


def verified_artifact(path, validator):
    """For readers (summaries, budget): the result must have an intact marker. Returns its manifest."""
    mark = read_json(done_path(path))
    if (mark.get('schema') != SCHEMA or mark['sha256'] != file_hash(path)
            or mark['fingerprint'] != digest(mark['manifest'])):
        raise ValueError(f'Stale/uncommitted artifact: {path}')
    validator(path)
    return mark['manifest']


# --------------------------------------------------------------------------------------------------------------------
# Child processes
#
# Every child starts in its own process group (start_new_session), so stopping it also stops the workers it spawned
# (SGLang, DeepSpeed). We never look up and kill processes we did not start.
# --------------------------------------------------------------------------------------------------------------------
def terminate_owned(proc, grace=10):
    """SIGTERM the child's process group, wait up to `grace` s, then SIGKILL whatever is left."""
    if proc is None:
        return

    def send(sig):
        with contextlib.suppress(ProcessLookupError):
            if os.name == 'posix':
                os.killpg(proc.pid, sig)
            elif proc.poll() is None:
                proc.terminate() if sig == signal.SIGTERM else proc.kill()

    send(signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace)
    # The group leader may be gone while its workers still run: wait for the whole group.
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
    """Run `cmd` with all output in `log`. Raises on a non-zero exit (CalledProcessError) or timeout."""
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen([str(x) for x in cmd], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=(os.name == 'posix'))
        try:
            returncode = proc.wait(timeout=timeout)
            if returncode:
                raise subprocess.CalledProcessError(returncode, cmd)
        finally:
            terminate_owned(proc)
