"""One GPU session (SLURM, at most 8 h): both variants side by side, resumed from their checkpoints.

    python session.py            normal session (bash session.sh runs it with nohup)
    python session.py --smoke    first-session check on 4000 dialogues (~30 min), see README

What a session does:
1. preflight: GPUs and free memory, disk, data, plan, patched code; stops with a clear message if something is off;
2. starts monitor.py and one trainer per variant: train.py (the authors' code) for baseline, train_nemo_full.py
   (NeMo's recipe) for ours; both on one GPU, or one GPU each if there are two;
   both get the same deadline: the end of the SLURM job (squeue) or SESSION_HOURS, minus a margin for the
   checkpoint;
3. restarts a crashed variant from its last checkpoint (up to 3 times, if more than 30 min are left);
4. at the end: a summary to Telegram / runs/monitor.log.
Secrets (Telegram, ClearML) are read from ~/.eagle_secrets (KEY=value lines, chmod 600); never put them in git.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import settings as S

RESTARTS = 3


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


def load_secrets():
    path = Path('~/.eagle_secrets').expanduser()
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip().removeprefix('export ').strip(), value.strip().strip('"\''))


def session_end():
    """Unix time when this session is killed: the SLURM job's end time, else now + SESSION_HOURS (default 8).

    The job is SLURM_JOB_ID, or, when the shell does not carry it (a terminal opened into the job, our sshd),
    the user's running job that ends first."""
    job = os.environ.get('SLURM_JOB_ID')
    query = ['-j', job] if job else ['-u', os.environ.get('USER', ''), '-t', 'R']
    try:
        lefts = subprocess.check_output(['squeue', '-h', *query, '-o', '%L'], text=True, timeout=30).split()
        ends = []
        for left in lefts:
            days, _, clock = left.rpartition('-')
            parts = [int(p) for p in clock.split(':')]
            while len(parts) < 3:
                parts.insert(0, 0)
            ends.append(time.time() + (int(days or 0) * 24 + parts[0]) * 3600 + parts[1] * 60 + parts[2])
        if ends:
            return min(ends)
        log('no running SLURM job found; using SESSION_HOURS')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        log(f'could not read the SLURM time limit ({error!r}); using SESSION_HOURS')
    return time.time() + float(os.environ.get('SESSION_HOURS', '8')) * 3600


def telegram_reachable():
    import urllib.request
    try:
        urllib.request.urlopen('https://api.telegram.org', timeout=10)
        return True
    except OSError as error:
        return getattr(error, 'code', None) is not None      # an HTTP error still means the host answered


def gpus():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
                                   '--format=csv,noheader,nounits'], text=True, timeout=60)
    return [tuple(int(x) for x in line.split(',')) for line in out.strip().splitlines()]


def preflight(data, runs, smoke):
    problems = []
    for path, what in [(S.PYTHON, 'training env (nemo_speed/venv_orig)'), (S.TARGET / 'config.json', 'target model'),
                       (S.NEMO_PYTHON, 'NeMo AutoModel env (nemo_speed/Automodel/.venv)'),
                       (S.EAGLE_REPO / 'eagle/traineagle3/cnets.py', 'EAGLE clone'),
                       (S.WORK / 'train/cnets.py', 'patched code (python patch_eagle.py)'),
                       (data / 'train', 'data (python prepare_data.py)'), (data / 'vocab.json', 'draft vocabulary')]:
        if not Path(path).exists():
            problems.append(f'missing {what}: {path}')
    if not smoke and not (runs / 'plan.json').exists():
        problems.append('missing runs/plan.json (python plan.py after the smoke test)')
    free_disk = shutil.disk_usage(S.HERE).free / 2 ** 30
    if free_disk < 60:
        problems.append(f'only {free_disk:.0f} GB free on disk (checkpoints need ~25 GB)')
    devices = gpus()
    need = 80_000 if len(devices) == 1 else 40_000            # MiB: two runs on one GPU need ~2 x 40 GB
    usable = [d for d in devices if d[1] >= need]
    if not usable:
        problems.append(f'no GPU with {need // 1000} GB free: {devices}')
    return problems, usable


def environment(variant, gpu):
    """gpu: None = keep the session's visible GPU(s) (one GPU: both variants on it), else a CUDA_VISIBLE_DEVICES value."""
    env = dict(os.environ, DS_BUILD_OPS='0', PYTHONUNBUFFERED='1',
               MASTER_ADDR='127.0.0.1', MASTER_PORT=str(29611 + sorted(S.VARIANTS).index(variant)),
               RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', TOKENIZERS_PARALLELISM='false')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    if not shutil.which('nvcc'):
        env['CUDA_HOME'] = str(S.CUDA_STUB)               # DeepSpeed only asks nvcc for its version
    return env


def start(variant, gpu, deadline, extra, runs, data, smoke):
    run_dir = runs / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    out = open(run_dir / 'train.log', 'a', encoding='utf-8')
    if S.VARIANTS[variant]['trainer'] == 'nemo':        # NeMo's recipe, from its own checkout and venv
        cmd = [str(S.NEMO_PYTHON), str(S.HERE / 'train_nemo_full.py'), '--deadline', str(deadline),
               '--runs', str(runs), '--data', str(data), *(['--smoke'] if smoke else [])]
        cwd = S.NEMO_DIR
    else:
        cmd = [str(S.PYTHON), str(S.HERE / 'train.py'), variant, '--deadline', str(deadline), '--runs', str(runs),
               *extra]
        cwd = S.HERE
    log(f'start {variant} on GPU {gpu or "(session default)"}: {" ".join(cmd[1:])}')
    return subprocess.Popen(cmd, cwd=cwd, env=environment(variant, gpu), stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args, extra = parser.parse_known_args()
    load_secrets()
    data = S.DATA if not args.smoke else S.HERE / 'data_smoke'
    runs = S.RUNS if not args.smoke else S.HERE / 'runs_smoke'
    runs.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        extra = ['--data', str(data), '--total-steps', '150', '--log-every', '10', '--val-every', '50',
                 '--val-eval', '50', '--checkpoint-minutes', '3', '--milestone-every', '150', *extra]
    problems, usable = preflight(data, runs, args.smoke)
    if problems:
        for p in problems:
            log('PREFLIGHT: ' + p)
        sys.exit(1)
    end = session_end()
    log(f'session ends {time.strftime("%Y-%m-%d %H:%M", time.localtime(end))}; GPUs {usable}')
    variants = sorted(S.VARIANTS)
    if len(usable) < 2:
        gpu_of = {v: None for v in variants}
    else:   # one GPU each; positions in the session's own CUDA_VISIBLE_DEVICES list if SLURM set one
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        ids = [visible[i] if len(visible) > i and visible[i] else str(d[0]) for i, d in enumerate(usable)]
        gpu_of = {v: ids[i] for i, v in enumerate(variants)}

    if not telegram_reachable():                 # e.g. our H200 node: then relay.py serves Telegram from outside
        log('Telegram is not reachable from this node: no bot here, messages only in runs/monitor.log')
        os.environ.pop('TELEGRAM_BOT_TOKEN', None)
    monitor = subprocess.Popen([str(S.PYTHON), str(S.HERE / 'monitor.py'), '--runs', str(runs),
                                '--until', str(end), *(['--every', '60', '--summary-every', '600'] if args.smoke else [])],
                               cwd=S.HERE, env=dict(os.environ), start_new_session=True)
    if not args.smoke and not (runs / 'reference.json').exists():
        log('scoring the official draft on the validation dialogues (reference for the forecast, ~5 min)')
        subprocess.run([str(S.PYTHON), str(S.HERE / 'reference_eval.py')], cwd=S.HERE,
                       env=environment('baseline', None), timeout=3600, check=False)
    heartbeat = runs / 'bot_heartbeat'           # the Telegram bot, unless a copy already runs (e.g. on the login node)
    bot = None
    if os.environ.get('TELEGRAM_BOT_TOKEN') and not (heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < 120):
        bot = subprocess.Popen([str(S.PYTHON), str(S.HERE / 'bot.py'), '--runs', str(runs)], cwd=S.HERE,
                               env=dict(os.environ), stdout=open(runs / 'bot.log', 'a', encoding='utf-8'),
                               stderr=subprocess.STDOUT, start_new_session=True)
    syncs = []                                   # ClearML in separate processes: it must never block training
    if os.environ.get('CLEARML_API_ACCESS_KEY') or Path('~/clearml.conf').expanduser().exists():
        for v in variants:                       # curves and status in two processes (see clearml_sync.py)
            for mode, log_name in (([], f'clearml_{v}.log'), (['--status'], f'clearml_{v}_status.log')):
                syncs.append(subprocess.Popen([str(S.PYTHON), str(S.HERE / 'clearml_sync.py'), v, *mode,
                                               '--runs', str(runs), '--until', str(end + 900)], cwd=S.HERE,
                                              env=dict(os.environ), stdout=open(runs / log_name, 'a', encoding='utf-8'),
                                              stderr=subprocess.STDOUT, start_new_session=True))
    def finished(v):                             # e.g. baseline after the comparison step: no need to load it again
        path = runs / v / 'status.json'
        return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('phase') == 'done'
    active = [v for v in variants if not finished(v)]
    for v in sorted(set(variants) - set(active)):
        log(f'{v}: its plan is finished, not started')
    procs = {v: start(v, gpu_of[v], end, extra, runs, data, args.smoke) for v in active}
    restarts = {v: 0 for v in variants}
    smoke_killed = False
    try:
        while procs:
            time.sleep(30)
            for v, p in list(procs.items()):
                code = p.poll()
                if args.smoke and not smoke_killed and v == 'baseline' and code is None:
                    status = json.loads((runs / v / 'status.json').read_text()) if (runs / v / 'status.json').exists() else {}
                    if status.get('step', 0) >= 70:          # smoke: interrupt once to test the resume path
                        log('smoke: SIGTERM to baseline to test checkpoint + resume')
                        os.kill(p.pid, signal.SIGTERM)       # the trainer only, not its data-loader workers
                        smoke_killed = True
                if code is None:
                    continue
                del procs[v]
                status = json.loads((runs / v / 'status.json').read_text()) if (runs / v / 'status.json').exists() else {}
                phase = status.get('phase')
                log(f'{v} exited with code {code} (phase {phase}, step {status.get("step")})')
                left = end - time.time()
                resumable = code != 0 or (args.smoke and phase == 'stopped' and smoke_killed and restarts[v] == 0)
                if phase == 'done' or left < 30 * 60 or not resumable:
                    continue
                if restarts[v] >= RESTARTS:
                    log(f'{v}: {RESTARTS} restarts already, giving up for this session')
                    continue
                restarts[v] += 1
                procs[v] = start(v, gpu_of[v], end, extra, runs, data, args.smoke)
    except KeyboardInterrupt:
        for p in procs.values():
            os.killpg(p.pid, signal.SIGTERM)
        for p in procs.values():
            p.wait()
    subprocess.run([str(S.PYTHON), str(S.HERE / 'monitor.py'), '--runs', str(runs), '--once'], cwd=S.HERE,
                   env=dict(os.environ), timeout=120)
    monitor.terminate()
    time.sleep(40)                               # one more sync round for the last numbers
    for p in syncs:
        p.terminate()
    if bot is not None:
        bot.terminate()
    log('session finished')


if __name__ == '__main__':
    main()
