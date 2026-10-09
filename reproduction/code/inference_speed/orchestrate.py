"""Owned processes, deadlines, transactional result manifests and stage ordering.
Only this module launches GPU jobs. Thin shell entrypoints never hide failures.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from runtime import (BENCHES, EAGLE_COMMIT, NEMO_COMMIT, MODELS, SCHEMA, atomic_json,
                     digest, file_hash, finish, finished, questions, read_json,
                     run_process, terminate_owned, validate_answers, validate_speed)

ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT.parent / 'nemo_speed'
TRAIN_REVISION = 'c8cf5337f3a4bed5ba9da7362446cc5111fd24f1'
# EAGLE-3 paper (appendix): tree depth 8 (EAGLE-2: 6) with the same nodes, 60 tokens / top-k 10.
# Author code counts depth as draft length - 1 (--depth 7, the EaModel default; the eval scripts' own
# --depth 5 default is the EAGLE-2 tree); SGLang's num-steps is the draft length itself (8).
AUTHOR_TREE = {'total_token': 60, 'depth': 7, 'top_k': 10}
TREE_CONFIGS = [('paper_8_10_60', 8, 10, 60), ('chain_3_1_4', 3, 1, 4), ('docs_5_8_32', 5, 8, 32),
                ('eagle2_6_10_60', 6, 10, 60)]


def http_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


class Runner:
    def __init__(self, mode):
        self.mode = mode
        self.smoke = mode == 'check' or os.environ.get('SMOKE', '0') == '1'
        self.tag = 'smoke' if self.smoke else 'main'
        self.results = ROOT / 'results' / self.tag
        self.logs = ROOT / 'logs' / self.tag
        self.train = TRAIN_ROOT / 'artifacts' / self.tag
        for p in (self.results, self.logs, self.train):
            p.mkdir(parents=True, exist_ok=True)
        self.state_path = self.results / 'stages.json'
        try:
            self.state = read_json(self.state_path)
        except (OSError, ValueError):
            self.state = {}
        seconds = int(os.environ.get('NIGHT_SECONDS', '25200'))
        if seconds <= 0:
            raise ValueError('NIGHT_SECONDS must be positive')
        self.deadline = time.monotonic() + seconds
        self.model = ROOT / 'models/llama31-8b-instruct'
        self.draft = ROOT / 'models/eagle3-llama31-8b'
        self.count = 2 if self.smoke else 80
        self.n = 40 if self.smoke else int(os.environ.get('N', '1000'))
        self.warmup = 0 if self.smoke else int(os.environ.get('WARMUP', '20'))
        if self.n <= 2 * self.warmup or self.n % 2:
            raise ValueError('N must be even and leave measured optimizer steps')
        self.env = dict(os.environ)
        self.env.update({'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'disabled', 'WANDB_DISABLED': 'true',
                         'SGLANG_ENABLE_JIT_DEEPGEMM': '0', 'SGL_ENABLE_JIT_DEEPGEMM': '0',
                         'TRAIN_DATA_REVISION': TRAIN_REVISION, 'MAX_Q': str(self.count),
                         'PYTHONPATH': str(ROOT) + os.pathsep + self.env.get('PYTHONPATH', ''),
                         'SAMPLING_SEED': '0'})
        self.ea = ROOT / 'venv_eagle/bin/python'
        self.sg = ROOT / 'venv_sglang/bin/python'
        self.orig = TRAIN_ROOT / 'venv_orig/bin/python'
        self.nemo = TRAIN_ROOT / 'Automodel/.venv/bin/python'
        self.env['PATH'] = os.pathsep.join((str(self.orig.parent), str(self.sg.parent),
                                            self.env.get('PATH', '')))
        self.failures = []
        self.environment = {}
        self.ds_cuda_home = None    # set in setup() when there is no CUDA toolkit

    def remaining(self, cap=3600):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('Overall night deadline reached; completed results are preserved')
        return min(left, cap)

    def status(self, name, status, **extra):
        self.state[name] = {'status': status, 'at': time.time(), **extra}
        atomic_json(self.state_path, self.state)
        print(f'{time.strftime("%F %T")} {name}: {status}', flush=True)

    def attempt(self, name, fn, *args, **kwargs):
        """One failed stage must not cost the rest of the night; only the deadline or a signal stops it."""
        try:
            return fn(*args, **kwargs)
        except TimeoutError:
            raise
        except Exception as exc:
            self.failures.append(name)
            if self.state.get(name, {}).get('status') != 'FAILED':
                self.status(name, 'FAILED', reason=f'{type(exc).__name__}: {exc}')
            print(f'{time.strftime("%F %T")} {name}: continuing with the next stage', flush=True)

    def command(self, name, cmd, cwd=ROOT, timeout=3600, env=None):
        self.status(name, 'RUNNING')
        try:
            run_process(cmd, cwd=cwd, log=self.logs / (name + '.log'),
                        timeout=self.remaining(timeout), env=self.env if env is None else env)
        except BaseException as exc:
            self.status(name, 'FAILED', reason=f'{type(exc).__name__}: {exc}')
            raise
        self.status(name, 'SUCCEEDED')

    def repository(self, destination, url, commit, name):
        if not (destination / '.git').is_dir():
            if destination.exists():
                raise RuntimeError(f'{destination.name} exists without .git; refusing deletion')
            self.command(name + '_clone', ['git', 'clone', url, destination], timeout=900)
        current = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
        # Old .installed markers and local virtualenvs are not source edits.
        # Git itself refuses a checkout that would overwrite an untracked file.
        dirty = subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain',
                                         '--untracked-files=no'], text=True).strip()
        if dirty:
            raise RuntimeError(f'{destination.name} has changes; refusing automatic checkout')
        if current != commit:
            self.command(name + '_fetch', ['git', '-C', destination, 'fetch', 'origin', commit], timeout=600)
            self.command(name + '_checkout', ['git', '-C', destination, 'checkout', '--detach', commit], timeout=60)
        actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise RuntimeError('Repository pin mismatch')

    def venv(self, python, name, requirements, imports):
        desired = digest({'schema': SCHEMA, 'requirements': requirements, 'imports': imports})
        marker = python.parent.parent / '.installed.json'
        if not python.exists():
            self.command(name + '_venv', [sys.executable, '-m', 'venv', python.parent.parent], timeout=120)
        matches = False
        try:
            matches = read_json(marker)['desired'] == desired
        except (OSError, ValueError, KeyError):
            pass
        if not matches:
            self.command(name + '_pip', [python, '-m', 'pip', 'install', *requirements], timeout=3600)
        # Validate complete import chain, not just the existence of the marker.
        self.command(name + '_imports', [python, '-c', imports], timeout=120)
        freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True, timeout=60)
        actual = {'desired': desired, 'freeze_sha256': digest(sorted(freeze.splitlines()))}
        atomic_json(marker, actual)
        self.environment[name] = actual

    def setup(self, inference=True, training=True, sglang=True):
        self.repository(ROOT / 'EAGLE', 'https://github.com/SafeAILab/EAGLE', EAGLE_COMMIT, 'eagle')
        if inference:
            self.venv(self.ea, 'eagle_env',
                ['torch==2.5.1', 'transformers==4.53.2', 'fschat==0.2.31', 'openai==0.28.1',
                 'anthropic==0.3.11', 'pydantic<2', 'accelerate', 'shortuuid', 'sentencepiece',
                 'protobuf', 'numpy', 'huggingface_hub', 'requests', 'datasets', 'tqdm'],
                'import torch, transformers, fastchat.llm_judge.common; assert torch.__version__.split("+")[0]=="2.5.1"; assert transformers.__version__=="4.53.2"')
        if training:
            self.venv(self.orig, 'original_env',
                ['torch==2.5.1', 'transformers==4.53.2', 'accelerate', 'datasets', 'safetensors',
                 'sentencepiece', 'tqdm', 'numpy', 'protobuf', 'huggingface_hub', 'uv', 'wheel', 'setuptools', 'ninja'],
                'import torch, transformers, datasets; assert torch.__version__.split("+")[0]=="2.5.1"; assert transformers.__version__=="4.53.2"')
            dsenv = dict(self.env)
            dsenv['DS_BUILD_OPS'] = '0'
            if not shutil.which('nvcc'):
                version = subprocess.check_output([str(self.orig), '-c', 'import torch; print(torch.version.cuda)'], text=True, timeout=30).strip()
                stub = self.orig.parent.parent / 'stub_cuda/bin/nvcc'
                stub.parent.mkdir(parents=True, exist_ok=True)
                stub.write_text('#!/bin/sh\necho "Cuda compilation tools, release ' + version + ', V' + version + '.0"\n', encoding='utf-8')
                stub.chmod(0o755)
                dsenv['CUDA_HOME'] = str(stub.parent.parent)
                # DeepSpeed 0.16.4 asks nvcc for the CUDA version on every import (op compatibility checks
                # in git_version_info), not only at install: every DeepSpeed process gets the stub.
                self.ds_cuda_home = str(stub.parent.parent)
            self.command('deepspeed_install', [self.orig, '-m', 'pip', 'install', '--no-build-isolation', 'deepspeed==0.16.4'], env=dsenv, timeout=900)
            self.command('deepspeed_import', [self.orig, '-c', 'import deepspeed; assert deepspeed.__version__.split("+")[0]=="0.16.4"'], env=dsenv, timeout=120)
            self.repository(TRAIN_ROOT / 'Automodel', 'https://github.com/NVIDIA-NeMo/Automodel', NEMO_COMMIT, 'nemo')
            uv = self.orig.parent / 'uv'
            self.command('nemo_sync', [uv, 'sync', '--frozen'], cwd=TRAIN_ROOT / 'Automodel', timeout=3600)
            if shutil.which('nvcc') or os.environ.get('TRY_FA2', '0') == '1':
                try:
                    self.command('nemo_fa2_install', [uv, 'sync', '--frozen', '--extra', 'fa'],
                                 cwd=TRAIN_ROOT / 'Automodel', timeout=900)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    self.command('nemo_restore_base', [uv, 'sync', '--frozen'], cwd=TRAIN_ROOT / 'Automodel', timeout=900)
            self.command('nemo_import', [self.nemo, '-c', 'import nemo_automodel.recipes.llm.train_eagle3'], timeout=180)
            # uv freeze includes actual package versions and is tied to the repository lockfile.
            freeze = subprocess.check_output([str(uv), 'pip', 'freeze', '--python', str(self.nemo)], text=True, timeout=60)
            self.environment['nemo_env'] = {'commit': NEMO_COMMIT, 'freeze_sha256': digest(sorted(freeze.splitlines()))}
        if sglang:
            # A dependency pulls the newest `kernels` (needs huggingface_hub>=1.10, while transformers 4.57 pins <1.0);
            # transformers imports it only if present, and the mismatch kills every SGLang server at start.
            self.venv(self.sg, 'sglang_env', ['sglang[all]==0.5.9', 'requests', 'datasets'],
                'import sglang, requests, datasets; assert sglang.__version__.split("+")[0]=="0.5.9"')
            self.command('sglang_drop_kernels', [self.sg, '-m', 'pip', 'uninstall', '-y', 'kernels'], timeout=120)
        py = self.ea if inference else self.orig
        shared = TRAIN_ROOT / 'models/llama31-8b-instruct'    # downloaded by the earlier training-speed runs
        if not self.model.exists() and (shared / 'config.json').exists():
            self.model.parent.mkdir(parents=True, exist_ok=True)
            self.model.symlink_to(shared, target_is_directory=True)
        self.command('target_snapshot', [py, ROOT / 'download.py', MODELS['target'][0], self.model, '*.json', '*.safetensors'], timeout=1800)
        if sglang or inference:
            self.command('draft_snapshot', [py, ROOT / 'download.py', MODELS['draft'][0], self.draft, '*.json', '*.bin'], timeout=1200)
        self.base_manifest = {'schema': SCHEMA, 'mode': self.tag, 'models': MODELS, 'eagle': EAGLE_COMMIT,
            'nemo': NEMO_COMMIT, 'environment': self.environment, 'count': self.count,
            'package': {str(p.relative_to(ROOT.parent)): file_hash(p)
                        for folder in (ROOT, TRAIN_ROOT) for p in sorted(folder.iterdir())
                        if p.suffix in ('.py', '.sh', '.yaml') and p.is_file()},
            'target_snapshot': read_json(self.model / 'snapshot_manifest.json'),
            'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,power.limit',
                '--format=csv,noheader'], text=True, timeout=30).strip(),
            'threads': {k: self.env.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}

    def artifact(self, name, path, manifest, validator, command, *, cwd=ROOT, env=None, timeout=3600):
        manifest = {**self.base_manifest, 'stage': name, **manifest}
        if finished(path, manifest, validator):
            self.status(name, 'REUSED')
            return
        # A complete old artifact remains inspectable; an uncommitted author JSONL
        # is cleared only for the exact requested output, never by recursive reset.
        if path.exists():
            archive = path.with_name(path.name + '.previous.' + str(time.time_ns()))
            os.replace(path, archive)
        try:
            self.command(name, command, cwd=cwd, env=env, timeout=timeout)
            server_log = getattr(self, 'active_server_log', None)
            if server_log and 'Falling back to greedy verification' in server_log.read_text(encoding='utf-8', errors='replace'):
                raise RuntimeError('T=1 degraded to greedy; result is not committed')
            validator(path)
            finish(path, manifest, validator)
        except BaseException as exc:
            self.status(name, 'FAILED', reason=f'{type(exc).__name__}: {exc}')
            raise

    def author_benches(self):
        for temperature, prefix in ((0, ''), (1, 't1_')):
            for bench in BENCHES:
                expected = questions(ROOT / 'EAGLE', bench, self.count)
                for use_eagle, script in ((True, 'gen_ea_answer_llama3chat'), (False, 'gen_baseline_answer_llama3chat')):
                    name = ('ea_' if use_eagle else 'base_') + prefix + bench
                    path = self.results / 'eagle' / (name + '.jsonl')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [self.ea, '-m', 'eagle.evaluation.' + script, '--temperature', str(temperature),
                           '--ea-model-path', self.draft, '--base-model-path', self.model,
                           '--bench-name', bench, '--question-begin', '0', '--question-end', str(self.count),
                           '--total-token', str(AUTHOR_TREE['total_token']), '--depth', str(AUTHOR_TREE['depth']),
                           '--top-k', str(AUTHOR_TREE['top_k']),
                           '--answer-file', path]
                    if use_eagle:
                        cmd += ['--use_eagle3']
                    self.attempt(name, self.artifact, name, path,
                        {'protocol': 'unchanged-author', 'temperature': temperature, 'tree': [AUTHOR_TREE['total_token'], AUTHOR_TREE['depth'], AUTHOR_TREE['top_k']]},
                        lambda p, e=expected: validate_answers(p, e, 'eagle'), cmd, cwd=ROOT / 'EAGLE',
                        timeout=int(os.environ.get('BENCH_TIMEOUT', '5400')))
                self.attempt('inference_summary', self.inference_summary)

    def inference_summary(self):
        self.command('inference_summary', [self.ea, ROOT / 'summarize.py', self.model, self.results], timeout=120)

    @contextlib.contextmanager
    def server(self, name, tree=None):
        port = int(os.environ.get('SGLANG_PORT', '30000'))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))  # fail if already owned, never kill its owner
        if not all(shutil.which(x, path=self.env['PATH']) for x in ('gcc', 'g++', 'ninja')):
            raise RuntimeError('Missing gcc/g++/ninja for runtime kernels; run setup before the allocation')
        if tree:
            self.command('tree_sampling_kernel', [self.sg, '-c',
                'from sglang.srt.speculative.spec_utils import TREE_SPEC_KERNEL_AVAILABLE; assert TREE_SPEC_KERNEL_AVAILABLE, "T=1 would fall back to greedy"'], timeout=120)
        cmd = [self.sg, '-m', 'sglang.launch_server', '--model-path', self.model, '--dtype', 'float16',
               '--host', '127.0.0.1', '--port', str(port), '--mem-fraction-static', '0.8',
               '--attention-backend', 'triton', '--sampling-backend', 'pytorch', '--disable-radix-cache']
        if tree:
            steps, topk, tokens = tree
            cmd += ['--speculative-algorithm', 'EAGLE3', '--speculative-draft-model-path', self.draft,
                    '--speculative-num-steps', str(steps), '--speculative-eagle-topk', str(topk),
                    '--speculative-num-draft-tokens', str(tokens)]
        log = self.logs / ('server_' + name + '.log')
        proc = None
        url = f'http://127.0.0.1:{port}'
        with log.open('w', encoding='utf-8') as stream:
            try:
                proc = subprocess.Popen([str(x) for x in cmd], cwd=ROOT, env=self.env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                until = time.monotonic() + self.remaining(600)
                while time.monotonic() < until:
                    if proc.poll() is not None:
                        raise RuntimeError(f'Server exited ({proc.returncode}): {log.name}')
                    try:
                        info = http_json(url + '/server_info')
                        args = info.get('server_args', info)
                        if Path(args['model_path']).resolve() != self.model.resolve():
                            raise RuntimeError('Wrong server target model')
                        if tree:
                            for key, value in zip(('speculative_num_steps', 'speculative_eagle_topk', 'speculative_num_draft_tokens'), tree):
                                if args.get(key) != value:
                                    raise RuntimeError('Wrong speculative server configuration')
                        elif args.get('speculative_algorithm') not in (None, 'NONE'):
                            raise RuntimeError('Expected a plain server')
                        atomic_json(self.logs / ('server_' + name + '.json'), info)
                        break
                    except (OSError, json.JSONDecodeError):
                        time.sleep(1)
                else:
                    raise TimeoutError('Server startup deadline')
                self.active_server_log = log
                yield url
                if 'Falling back to greedy verification' in log.read_text(encoding='utf-8', errors='replace'):
                    raise RuntimeError('T=1 degraded to greedy verification')
            finally:
                self.active_server_log = None
                terminate_owned(proc)

    def sg_benches(self):
        configs = [('base', None), *[(n, (s, k, t)) for n, s, k, t in TREE_CONFIGS]]
        # Three T=1 seeds are reported separately (no invented across-engine RNG parity).
        for name, tree in configs:
            temperatures = (0, 1) if name in ('base', 'paper_8_10_60') else (0,)
            jobs = []
            for temp in temperatures:
                for seed in ((0, 1, 2) if temp and not self.smoke else (0,)):
                    prefix = name + ('_t1' if temp else '') + (f'_seed{seed}' if seed else '')
                    for bench in BENCHES:
                        path = self.results / 'sglang' / f'{prefix}_{bench}.json'
                        expected = questions(ROOT / 'EAGLE', bench, self.count)
                        manifest = {'protocol': 'bounded-http-no-prefix-cache', 'tree': tree, 'temperature': temp, 'seed': seed}
                        full = {**self.base_manifest, 'stage': prefix + '_' + bench, **manifest}
                        validator = lambda p, e=expected: validate_answers(p, e, 'sglang')
                        if finished(path, full, validator):
                            self.status(prefix + '_' + bench, 'REUSED')
                        else:
                            jobs.append((prefix, bench, temp, seed, path, manifest, validator))
            if jobs:
                self.attempt('server_' + name, self.sg_config, name, tree, jobs)

    def sg_config(self, name, tree, jobs):
        with self.server(name, tree) as url:
            for prefix, bench, temp, seed, path, manifest, validator in jobs:
                env = {**self.env, 'SPECULATIVE': '1' if tree else '0', 'SAMPLING_SEED': str(seed)}
                self.attempt(prefix + '_' + bench, self.artifact, prefix + '_' + bench, path, manifest, validator,
                    [self.sg, ROOT / 'sglang_client.py', url, self.model, bench, path, str(temp)],
                    env=env, timeout=int(os.environ.get('BENCH_TIMEOUT', '5400')))
                self.attempt('inference_summary', self.inference_summary)

    def prepare_training(self):
        data = self.train / 'data'
        manifest = {'N': self.n, 'revision': TRAIN_REVISION, 'protocol': 'shared-author-tokens'}
        path = data / 'stats.json'
        def validate(p):
            stats = read_json(p)
            if stats['samples'] != self.n or stats['canonical_sha256'] != file_hash(data / 'canonical.jsonl'):
                raise ValueError('Wrong training data')
            if stats['mapping_sha256'] != digest(read_json(data / 'selected_tokens.json')):
                raise ValueError('Wrong mapping')
        self.artifact('training_data', path, manifest, validate,
            [self.orig, TRAIN_ROOT / 'prepare_data.py', self.model, str(self.n), data, ROOT / 'EAGLE'], timeout=1800)
        return data

    def training(self, variants=False):
        data = self.prepare_training()
        identity = {'data': file_hash(data / 'canonical.jsonl'), 'mapping': file_hash(data / 'selected_tokens.json'),
                    'N': self.n, 'warmup': self.warmup, 'protocol': 'matched-bf16-fixed2048-v1'}
        if not variants:
            work = self.train / 'work/original'
            out = self.train / 'original.speed.json'
            manifest = {**identity, 'backend': 'original', 'batch': [1, 2], 'packing': 0}
            full = {**self.base_manifest, 'stage': 'train_original', **manifest}
            if not finished(out, full, validate_speed):
                self.attempt('patch_original', self.command, 'patch_original',
                             [self.orig, TRAIN_ROOT / 'patch_original.py', ROOT / 'EAGLE', str(self.warmup), work], timeout=60)
            env = {**self.env, 'SHARED_MAPPING': str(data / 'selected_tokens.json'), 'SPEED_OUTPUT': str(out)}
            if self.ds_cuda_home:
                env['CUDA_HOME'] = self.ds_cuda_home
            self.attempt('train_original', self.artifact, 'train_original', out, manifest, validate_speed,
                [self.orig, '-m', 'deepspeed.launcher.runner', '--num_gpus', '1', 'main.py',
                 '--deepspeed_config', 'ds_config.json', '--basepath', self.model,
                 '--trainpath', data / 'canonical.jsonl', '--testpath', data / 'test_canonical.jsonl',
                 '--savedir', work / 'out'], cwd=work, env=env, timeout=1800)
        configs = [('nemo_eager_nopack', 'eager', 0, None)] if not variants else [
            ('nemo_eager_pack', 'eager', 2048, None),
            ('nemo_fa2_pack', 'flash_attention_2', 2048, None),
            ('nemo_compile', 'eager', 2048, 'compile'),
            ('nemo_compile_fp8', 'eager', 2048, 'fp8'),
            ('nemo_fa2_pack_mb4', 'flash_attention_2', 2048, 'mb4')]
        fa2 = subprocess.run([str(self.nemo), '-c', 'import flash_attn'], env=self.env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60).returncode == 0
        for name, attn, pack, extra in configs:
            if attn == 'flash_attention_2' and not fa2:
                self.status('train_' + name, 'UNSUPPORTED', reason='flash-attn 2 is absent; no eager result under FA2 label')
                continue
            if extra in ('compile', 'fp8') and not all(shutil.which(x, path=self.env['PATH']) for x in ('gcc', 'g++')):
                self.status('train_' + name, 'UNSUPPORTED', reason='Missing C/C++ compiler')
                continue
            cfg = self.train / 'work' / (name + '.yaml')
            cfg.parent.mkdir(parents=True, exist_ok=True)
            text = (ROOT / 'nemo_config.yaml').read_text(encoding='utf-8')
            for key, value in {'__MODEL__': self.model, '__DATA__': data / 'canonical.jsonl',
                               '__OUT__': self.train / 'work' / name, '__ATTN__': attn, '__PACK__': pack}.items():
                # JSON quoted strings are legal YAML; paths with spaces remain valid.
                text = text.replace(key, json.dumps(str(value)) if isinstance(value, Path) else str(value))
            if extra in ('compile', 'fp8'):
                text += '\ncompile:\n  enabled: true\n  mode: default\n'
            if extra == 'fp8':
                text += '\nfp8:\n  enabled: true\n  recipe_name: tensorwise\n  filter_fqns: ["lm_head"]\n  emulate: false\n'
            if extra == 'mb4':
                text = text.replace('micro_batch_size: 1', 'micro_batch_size: 4').replace('grad_accumulation_steps: 2', 'grad_accumulation_steps: 1')
            from runtime import atomic_text
            atomic_text(cfg, text)
            manifest = {**identity, 'backend': 'nemo', 'configuration_sha256': digest(text), 'attention': attn}
            self.attempt('train_' + name, self.artifact, 'train_' + name, self.train / (name + '.speed.json'), manifest, validate_speed,
                [self.nemo, ROOT / 'train_nemo.py', '--config', cfg, '--data', data / 'canonical.jsonl',
                 '--mapping', data / 'selected_tokens.json', '--out', self.train / (name + '.speed.json'),
                 '--warmup', str(self.warmup)], cwd=TRAIN_ROOT / 'Automodel', timeout=1800)
            self.attempt('training_summary', self.training_summary)

    def training_summary(self):
        self.command('training_summary', [self.orig, ROOT / 'stamp_and_summarize.py', 'summary', self.train], timeout=60)

    def regeneration(self):
        from regen_probe import validate_probe
        path = self.results / 'regen_probe.json'
        manifest = {'protocol': 'stratified-multiturn-fixed1900-v1', 'samples_per_bucket': 2 if self.smoke else 64,
                    'scan_limit': 64 if self.smoke else 8192, 'concurrency': 8 if self.smoke else 64}
        full = {**self.base_manifest, 'stage': 'regeneration', **manifest}
        if finished(path, full, validate_probe):
            self.status('regeneration', 'REUSED')
            return
        with self.server('regeneration') as url:
            self.artifact('regeneration', path, manifest, validate_probe,
                [self.sg, ROOT / 'regen_probe.py', url, self.model, path,
                 '--samples-per-bucket', str(manifest['samples_per_bucket']),
                 '--scan-limit', str(manifest['scan_limit']), '--concurrency', str(manifest['concurrency'])], timeout=1800)

    def budget(self):
        if (self.results / 'regen_probe.json').exists() and (self.train / 'summary.json').exists():
            self.command('budget', [self.orig if self.orig.exists() else self.ea, ROOT / 'budget.py',
                '--probe', self.results / 'regen_probe.json', '--training', self.train / 'summary.json',
                '--out', self.results / 'budget.md'], timeout=60)
        else:
            self.status('budget', 'BLOCKED', reason='Validated regeneration/training results required')

    def execute(self):
        self.setup(inference=self.mode != 'train', training=self.mode in ('night', 'train', 'check', 'setup'),
                   sglang=self.mode != 'train')
        if os.environ.get('ENV_ONLY', '0') == '1' or self.mode == 'setup':
            return
        if self.mode in ('night', 'inference', 'check'):
            self.attempt('author_benches', self.author_benches)
        if self.mode in ('night', 'train', 'check'):
            self.attempt('training', self.training)
            self.attempt('training_summary', self.training_summary)
        if self.mode in ('night', 'inference', 'check'):
            self.attempt('regeneration', self.regeneration)
            self.attempt('budget', self.budget)
            self.attempt('sglang_benches', self.sg_benches)
        if self.mode in ('night', 'train', 'check'):
            self.attempt('training_variants', self.training, variants=True)
            self.attempt('training_summary', self.training_summary)
        self.attempt('budget', self.budget)
        if self.failures:
            raise RuntimeError('Failed stages (the rest ran): ' + ', '.join(dict.fromkeys(self.failures)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('night', 'inference', 'train', 'check', 'setup'))
    args = parser.parse_args()
    if os.name != 'posix':
        raise SystemExit('GPU orchestration requires Linux; run tests locally on Windows')
    import fcntl
    with (ROOT / '.night.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}; cleaning owned processes')
        signal.signal(signal.SIGTERM, interrupted)
        runner = Runner(args.mode)
        try:
            runner.execute()
            runner.status('overall', 'SUCCEEDED')
        except BaseException as exc:
            runner.status('overall', 'FAILED', reason=f'{type(exc).__name__}: {exc}')
            print(f'FAILED: {exc}; see {runner.state_path}', file=sys.stderr)
            raise


if __name__ == '__main__':
    main()
