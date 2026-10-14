"""Final evaluation: real tau and speed-up of the official draft, baseline and ours, in SGLang and in the authors' code.

    python final_eval.py                                   # all drafts, SGLang + the authors' code
    python final_eval.py --drafts baseline --no-authors     # e.g. baseline while ours still trains

For every draft:
  1. weights for SGLang: the official draft as in the reproduction (its weights, architectures =
     LlamaForCausalLMEagle3); ours and baseline exported from their milestone at the comparison step (export.py:
     target layers in SGLang's numbering, fc_norm loaded by sglang_eagle31 or the server refuses to start);
  2. an SGLang 0.5.9 server with the paper's tree (8 steps, top-k 10, 60 draft tokens), batch 1, temperature 0,
     and the reproduction's client (inference_speed/sglang_client.py: the authors' system prompt and chat template,
     all turns, 512 new tokens): MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM, 80 questions each;
  3. (unless --no-authors) the authors' own evaluation script (eval_tau.py: EaModel, tree 60 / depth 8 / top-k 10)
     on MT-bench, HumanEval and GSM8K.
A plain SGLang server on the same questions gives the speed of ordinary generation (the speed-up denominator).

tau = generated tokens - 1 per verification round (as in the reproduction's summary). Speeds are measured while
other GPU work runs only if nvidia-smi showed other processes: that is recorded per run ("gpu_shared"), and such
speeds are not comparable; tau is not affected.

Writes runs/final_eval/<draft>/*.json and runs/final_eval/summary.{json,md}.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import settings as S

ROOT = S.ROOT
INFER = ROOT / 'inference_speed'
SGLANG_PYTHON = INFER / 'venv_sglang' / 'bin' / 'python'
CUDA_HOME = ROOT / 'cuda-home'                     # nvcc for SGLang's start-up kernels (reproduction's setup_cuda.sh)
OUT = S.RUNS / 'final_eval'
BENCHES = ['mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'sum']
AUTHOR_BENCHES = ['mt_bench', 'humaneval', 'gsm8k']
TREE = (8, 10, 60)                                 # the paper's tree: num-steps, eagle-topk, num-draft-tokens
PORT = int(os.environ.get('SGLANG_PORT', '30123'))


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def milestone(variant):
    """The milestone at the comparison step (or, before it exists, the newest one)."""
    plan = json.loads((S.RUNS / 'plan.json').read_text(encoding='utf-8'))
    d = S.RUNS / variant / 'milestones'
    exact = d / f'step{plan["compare_step"]:07d}'
    if exact.exists():
        return exact
    newest = sorted(p for p in d.iterdir() if p.name.startswith('step'))[-1]
    log(f'{variant}: no milestone at the comparison step yet, using {newest.name}')
    return newest


def sglang_draft(name):
    """Directory SGLang loads for this draft."""
    if name == 'official':
        view = INFER / 'models' / 'eagle3-llama31-8b-sglang'   # made by the reproduction: same weights, EAGLE-3 class
        if not (view / 'config.json').exists():
            view.mkdir(parents=True, exist_ok=True)
            for f in S.OFFICIAL_DRAFT.iterdir():
                if f.is_file() and f.name != 'config.json' and not (view / f.name).exists():
                    (view / f.name).symlink_to(f.resolve())
            cfg = json.loads((S.OFFICIAL_DRAFT / 'config.json').read_text())
            cfg['architectures'] = ['LlamaForCausalLMEagle3']
            (view / 'config.json').write_text(json.dumps(cfg, indent=1))
        return view, S.OFFICIAL_DRAFT
    src = milestone(name)
    out = S.RUNS / 'exports' / f'{name}_{src.name}'
    if not (out / 'sglang' / 'config.json').exists():
        import export
        export.main(src, out)
    return out / 'sglang', src


def gpu_shared():
    """True if other processes use the GPU (then speeds are not clean)."""
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        return bool([x for x in out.split() if x.strip()])
    except (OSError, subprocess.SubprocessError):
        return None


def server_env():
    env = dict(os.environ, SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN='1', PYTHONPATH=str(S.HERE),
               SGLANG_EXTERNAL_MODEL_PACKAGE='sglang_eagle31')     # our class: adds fc_norm when the config asks
    env['PATH'] = f'{SGLANG_PYTHON.parent}{os.pathsep}{env.get("PATH", "")}'   # ninja: SGLang JIT-builds kernels
    if (CUDA_HOME / 'bin' / 'nvcc').exists():
        env['CUDA_HOME'] = str(CUDA_HOME)
        env['PATH'] = f'{CUDA_HOME / "bin"}{os.pathsep}{env.get("PATH", "")}'
        if (CUDA_HOME / 'bin' / 'g++').exists():
            env['NVCC_PREPEND_FLAGS'] = f'-ccbin {CUDA_HOME / "bin" / "g++"}'
    return env


class Server:
    """An SGLang server (plain or EAGLE-3 with a draft) for the duration of a `with` block."""

    def __init__(self, name, draft=None, mem=0.8):
        self.name, self.draft, self.mem = name, draft, mem
        self.url = f'http://127.0.0.1:{PORT}'
        self.log = OUT / 'logs' / f'server_{name}.log'

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', PORT)) == 0:
                raise RuntimeError(f'port {PORT} is busy (an old server?)')
        cmd = [str(SGLANG_PYTHON), '-m', 'sglang.launch_server', '--model-path', str(S.TARGET), '--dtype', 'float16',
               '--host', '127.0.0.1', '--port', str(PORT), '--mem-fraction-static', str(self.mem),
               '--attention-backend', 'triton', '--sampling-backend', 'pytorch', '--disable-radix-cache',
               '--random-seed', '0']
        if self.draft:
            cmd += ['--speculative-algorithm', 'EAGLE3', '--speculative-draft-model-path', str(self.draft),
                    '--speculative-num-steps', str(TREE[0]), '--speculative-eagle-topk', str(TREE[1]),
                    '--speculative-num-draft-tokens', str(TREE[2])]
        self.stream = self.log.open('w', encoding='utf-8')
        self.proc = subprocess.Popen(cmd, cwd=INFER, env=server_env(), stdout=self.stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        until = time.time() + 900
        while time.time() < until:
            if self.proc.poll() is not None:
                raise RuntimeError(f'SGLang server exited, see {self.log}')
            try:
                with urllib.request.urlopen(self.url + '/server_info', timeout=5) as r:
                    info = json.load(r)
                args = info.get('server_args', info)
                if self.draft and Path(args['speculative_draft_model_path']).resolve() != Path(self.draft).resolve():
                    raise RuntimeError('server runs another draft')
                log(f'server {self.name} ready')
                return self
            except (OSError, json.JSONDecodeError, KeyError):
                time.sleep(3)
        raise RuntimeError('SGLang server did not start in 15 min')

    def __exit__(self, *exc):
        import signal
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:                 # noqa: BLE001
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.stream.close()
        time.sleep(5)


def sglang_stats(path):
    """tau and speed of one answer file of sglang_client.py (as the reproduction's summarize.sglang_speed)."""
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    turns = [t for r in rows for t in r['turns']]
    rounds = sum(t.get('steps') or 0 for t in turns)
    speeds = [sum(t['tokens'] for t in r['turns']) / sum(t['seconds'] for t in r['turns']) for r in rows]
    return {'questions': len(rows), 'tok_s': sum(speeds) / len(speeds),
            'tau': (sum(t['tokens'] - 1 for t in turns) / rounds) if rounds else None}


def run_sglang(name, draft, benches, questions, mem):
    """All benchmarks against one server; skips files already there."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    todo = [b for b in benches if not (d / f'sglang_{b}.json').exists()]
    if not todo:
        return
    shared = gpu_shared()
    with Server(name, draft, mem) as server:
        for bench in todo:
            path = d / f'sglang_{bench}.json'
            env = dict(os.environ, MAX_Q=str(questions), SPECULATIVE='1' if draft else '0', SAMPLING_SEED='0')
            t0 = time.time()
            subprocess.run([str(SGLANG_PYTHON), str(INFER / 'sglang_client.py'), server.url, str(S.TARGET), bench,
                            str(path), '0'], cwd=INFER, env=env, check=True)
            st = sglang_stats(path)
            meta = {'draft': str(draft), 'tree': TREE if draft else None, 'gpu_shared': shared, **st}
            (d / f'sglang_{bench}.meta.json').write_text(json.dumps(meta, indent=1))
            log(f'{name} {bench}: tau {st["tau"] or 0:.3f}, {st["tok_s"]:.0f} tok/s ({time.time() - t0:.0f} s)'
                + ('  [GPU shared]' if shared else ''))


def run_authors(name, src, benches, questions):
    d = OUT / name
    for bench in benches:
        path = d / f'authors_{bench}.json'
        if path.exists():
            continue
        out = subprocess.run([str(S.PYTHON), str(S.HERE / 'eval_tau.py'), str(src), f'final_{name}', '--bench', bench,
                              '--questions', str(questions)], cwd=S.HERE, capture_output=True, text=True)
        if out.returncode:
            log(f'{name} {bench}: authors eval failed\n{out.stdout[-2000:]}{out.stderr[-2000:]}')
            continue
        result = json.loads(out.stdout.strip().splitlines()[-1])
        result['gpu_shared'] = gpu_shared()
        path.write_text(json.dumps(result, indent=1))
        log(f'{name} {bench} (authors): tau {result["tau"]:.3f}, {result["tokens_per_s"]:.0f} tok/s')


def summary(drafts):
    base = {b: json.loads((OUT / 'plain' / f'sglang_{b}.meta.json').read_text())
            for b in BENCHES if (OUT / 'plain' / f'sglang_{b}.meta.json').exists()}
    table, lines = {}, ['# Финальная оценка: реальная τ и ускорение', '',
                        f'SGLang 0.5.9, дерево статьи {TREE[0]}/{TREE[1]}/{TREE[2]}, batch 1, T = 0, 80 вопросов на '
                        'бенчмарк. Код авторов — дерево 60 / глубина 8 / top-k 10. '
                        '«*» — скорость мерилась, пока GPU делилась с другим процессом (несравнима).', '',
                        '| драфт | ' + ' | '.join(BENCHES) + ' | среднее |', '|---' * (len(BENCHES) + 2) + '|']
    for name in drafts:
        row, cells = {}, []
        for b in BENCHES:
            p = OUT / name / f'sglang_{b}.meta.json'
            if not p.exists():
                cells.append('—')
                continue
            m = json.loads(p.read_text())
            speed = m['tok_s'] / base[b]['tok_s'] if b in base else None
            row[b] = {**m, 'speedup': speed}
            mark = '*' if m.get('gpu_shared') or (b in base and base[b].get('gpu_shared')) else ''
            cells.append(f'τ {m["tau"]:.2f}' + (f', {speed:.2f}×{mark}' if speed else ''))
        taus = [r['tau'] for r in row.values() if r.get('tau')]
        cells.append(f'τ {sum(taus) / len(taus):.2f}' if taus else '—')
        authors = {b: json.loads((OUT / name / f'authors_{b}.json').read_text())
                   for b in AUTHOR_BENCHES if (OUT / name / f'authors_{b}.json').exists()}
        table[name] = {'sglang': row, 'authors': authors}
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Код авторов (τ)', '', '| драфт | ' + ' | '.join(AUTHOR_BENCHES) + ' |',
              '|---' * (len(AUTHOR_BENCHES) + 1) + '|']
    for name in drafts:
        a = table[name]['authors']
        lines.append(f'| {name} | ' + ' | '.join(f'{a[b]["tau"]:.2f}' if b in a else '—' for b in AUTHOR_BENCHES) + ' |')
    (OUT / 'summary.json').write_text(json.dumps(table, indent=1))
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drafts', default='official,baseline,ours')
    ap.add_argument('--benches', default=','.join(BENCHES))
    ap.add_argument('--questions', type=int, default=80)
    ap.add_argument('--no-plain', action='store_true', help='skip the plain server (speed-up denominator)')
    ap.add_argument('--no-authors', action='store_true', help="skip the authors' evaluation script")
    ap.add_argument('--mem', type=float, default=0.0, help='SGLang --mem-fraction-static (default: 0.8, 0.45 if the GPU is shared)')
    a = ap.parse_args()
    drafts, benches = a.drafts.split(','), a.benches.split(',')
    mem = a.mem or (0.45 if gpu_shared() else 0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.no_plain:
        run_sglang('plain', None, benches, a.questions, mem)
    for name in drafts:
        view, src = sglang_draft(name)
        log(f'{name}: SGLang draft {view} (from {src})')
        run_sglang(name, view, benches, a.questions, mem)
        if not a.no_authors:
            run_authors(name, src, [b for b in AUTHOR_BENCHES if b in benches], a.questions)
    summary(['official', 'baseline', 'ours'])


if __name__ == '__main__':
    sys.exit(main())
