"""CPU-only tests of the speed package: no torch, CUDA, downloads or installations.

    py -3.12 -X utf8 test_regressions.py
"""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'nemo_speed'))
from common import (atomic_json, atomic_text, finish, finished, run_process, validate_answers)
from training_common import SpeedWindow
from download import validate_files
from summarize import eagle_repo, sglang_speed
from budget import data_share, estimate
from patch_original import adapt_cnets, adapt_main
from prepare_data import selected_vocab
from sglang_client import generate


def expected():
    return [{'question_id': 1, 'turns': ['a']}, {'question_id': 2, 'turns': ['b']}]


def answers():
    return [{'question_id': i, 'choices': [{'turns': ['x'], 'wall_time': [2.0],
              'new_tokens': [6], 'idxs': [1]}]} for i in (1, 2)]


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_marker_requires_coverage_and_fingerprint(self):
        p = self.path / 'ea.jsonl'
        atomic_text(p, '\n'.join(json.dumps(r) for r in answers()))
        manifest = {'temperature': 0, 'model': 'pinned'}
        validate = lambda p: validate_answers(p, expected(), 'eagle')
        self.assertFalse(finished(p, manifest, validate))
        finish(p, manifest, validate)
        self.assertTrue(finished(p, manifest, validate))
        self.assertFalse(finished(p, {**manifest, 'temperature': 1}, validate))
        atomic_text(p, json.dumps(answers()[0]))
        self.assertFalse(finished(p, manifest, validate))
        with self.assertRaises(ValueError):
            validate(p)

    def test_duplicate_ids_and_missing_turns_rejected(self):
        for rows in ([answers()[0], answers()[0]],
                     [answers()[0], {**answers()[1], 'choices': [{'turns': [], 'wall_time': [], 'new_tokens': [], 'idxs': []}]}]):
            p = self.path / 'ea.jsonl'
            atomic_text(p, '\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                validate_answers(p, expected(), 'eagle')

    def test_author_arithmetic_and_partial_pair_rejected(self):
        a, b = self.path / 'a.jsonl', self.path / 'b.jsonl'
        for p in (a, b):
            atomic_text(p, '\n'.join(json.dumps(r) for r in answers()))
        tok = lambda text: SimpleNamespace(input_ids=list(range(7)))
        result = eagle_repo(a, b, tok, expected())
        self.assertEqual(result['speedup'], 1.0)
        self.assertEqual(result['tau'], 3.0)
        atomic_text(a, json.dumps(answers()[0]))
        with self.assertRaises(ValueError):
            eagle_repo(a, b, tok, expected())

    def test_tau_prefill_and_zero_rounds(self):
        p = self.path / 'sg.json'
        atomic_json(p, [{'question_id': 1, 'turns': [
            {'tokens': 7, 'steps': 2, 'accepted_draft_tokens': 4, 'seconds': 1},
            {'tokens': 1, 'steps': 0, 'accepted_draft_tokens': 0, 'seconds': 1}]}])
        r = sglang_speed(p)
        self.assertEqual(r['verification_tokens_per_round'], 3)
        self.assertEqual(r['completion_over_verify'], 4)
        self.assertEqual(r['zero_round_turns'], 1)
        self.assertEqual(r['tau'], 3)          # (7 - 1 + 1 - 1) generated after prefill / 2 rounds

    def test_empty_partial_download_rejected(self):
        atomic_json(self.path / 'config.json', {})
        (self.path / 'weight.bin').touch()
        with self.assertRaises(ValueError):
            validate_files(self.path, {'config.json': None, 'weight.bin': 10})
        (self.path / 'weight.bin').write_bytes(b'abc')
        with self.assertRaises(ValueError):
            validate_files(self.path, {'config.json': None, 'weight.bin': 10})

    def test_sharded_model_without_all_shards_rejected(self):
        atomic_json(self.path / 'config.json', {})
        atomic_json(self.path / 'model.index.json', {'weight_map': {'x': 'missing.bin'}})
        with self.assertRaises(ValueError):
            validate_files(self.path, {'config.json': None, 'model.index.json': None})

    def test_nonfinite_atomic_output_rejected(self):
        with self.assertRaises(ValueError):
            atomic_json(self.path / 'bad.json', {'seconds': float('nan')})
        self.assertFalse((self.path / 'bad.json').exists())

    def test_child_failure_propagates(self):
        with self.assertRaises(subprocess.CalledProcessError):
            run_process([sys.executable, '-c', 'raise SystemExit(7)'], cwd=self.path,
                        log=self.path / 'fail.log', timeout=3)

    def test_child_timeout_propagates(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_process([sys.executable, '-c', 'import time; time.sleep(20)'], cwd=self.path,
                        log=self.path / 'timeout.log', timeout=0.1)

    def test_sunk_regeneration_and_reserve(self):
        self.assertAlmostEqual(data_share(100, 20, 10, 10, 10), 90 / 120)
        self.assertAlmostEqual(data_share(100, 20, 10, 10, 10, True), .7)
        self.assertEqual(data_share(10, 20, 10, 1, 0, True), 0)

    def test_shared_vocab_exact_size_and_tie_order(self):
        rows = [{'input_ids': [9, 3, 9, 4], 'loss_mask': [1, 1, 1, 0]}]
        self.assertEqual(selected_vocab(rows, 20, size=2), [3, 9])
        self.assertEqual(len(selected_vocab(rows, 20, size=10)), 10)

    def test_original_patch_pinned_ast_and_accumulation(self):
        candidates = [ROOT / 'EAGLE', ROOT.parent.parent / 'eagle-work']     # clone made by the run, or a local one
        main_py = next((c / 'eagle/traineagle3/main.py' for c in candidates
                        if (c / 'eagle/traineagle3/main.py').exists()), None)
        if main_py is None:
            self.skipTest('no SafeAILab/EAGLE checkout next to the package')
        source = main_py.read_text(encoding='utf-8')
        patched = adapt_main(source, 20)
        ast.parse(patched)
        self.assertNotIn('model.zero_grad()', patched)
        self.assertIn('max_length = 2048', patched)
        self.assertIn('SpeedWindow(20, len(train_loader) // 2', patched)
        self.assertIn('from training_common import original_dataset', patched)
        self.assertIn('shuffle=False)', patched)
        with self.assertRaises(ValueError):
            adapt_main(source.replace('"num_epochs": 40,', '"num_epochs": 39,'), 20)
        as_is = adapt_main(source, 20, 'asis')
        ast.parse(as_is)
        self.assertIn('model.zero_grad()', as_is)                       # the authors' loop is kept
        self.assertNotIn('max_length = 2048', as_is)                    # no padding to 2048
        self.assertIn('"gradient_checkpoint": True', as_is)
        self.assertIn("protocol='author-as-is-fp16-v1'", as_is)

    def test_trainer_dict_access_bug_fixed(self):
        pinned = '\n'.join(['gradient_checkpointing = self.train_config.gradient_checkpointing',
                            'if len(input_ids) > self.train_config.max_len:',
                            '    x = AutoModel.from_pretrained(path, torch_dtype=torch.float16)', ''])
        fixed = adapt_cnets(pinned, 'asis')
        self.assertIn('self.train_config["gradient_checkpoint"]', fixed)
        self.assertIn('self.train_config["max_len"]', fixed)
        self.assertIn('torch.float16', fixed)
        self.assertIn('torch.bfloat16', adapt_cnets(pinned, 'matched'))

    def test_budget_weights_each_split(self):
        group = lambda w, sec, tok: {'weight': w, 'seconds_per_dialogue': sec, 'mean_training_tokens': tok}
        probe = {'sources': {'ultrachat': {'splits': {
            'train_sft': {'dialogues': 100, 'groups': {'a': group(1.0, 3600, 10)}},
            'train_gen': {'dialogues': 300, 'groups': {'a': group(0.5, 3600, 0), 'b': group(0.5, 7200, 20)}}}}}}
        regen_h, h_epoch, tokens = estimate(probe, 1.0)
        self.assertAlmostEqual(regen_h, 100 + 150 + 300)
        self.assertAlmostEqual(tokens, 100 * 10 + 150 * 20)

    def test_sglang_sampling_contract(self):
        out = {'text': 'answer', 'meta_info': {'completion_tokens': 5, 'spec_verify_ct': 2,
               'spec_accept_token_num': 2, 'finish_reason': {'type': 'stop'}}}
        requests = SimpleNamespace(post=lambda *a, **kw: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: out))
        with patch.dict(sys.modules, {'requests': requests}):
            r = generate('http://localhost', [1, 2], 1, stops=[9], seed=2, speculative=True)
        self.assertEqual(r['sampling_params']['sampling_seed'], 2)
        self.assertEqual(r['sampling_params']['temperature'], 1)
        self.assertEqual(r['sampling_params']['stop_token_ids'], [9])
        self.assertEqual(r['timing'], 'http_round_trip')
        with patch.dict(sys.modules, {'requests': requests}):
            with self.assertRaises(ValueError):
                generate('http://localhost', list(range(1980)), 0, stops=[9])

    def test_warmup_requires_measured_optimizer_steps(self):
        with self.assertRaises(ValueError):
            SpeedWindow(20, 20, 'nemo', {})
        w = SpeedWindow(2, 4, 'nemo', {})
        gpu = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        w.after_step(1, gpu)
        w.after_step(2, gpu)
        self.assertEqual(w.steps, 0)
        w.after_step(3, gpu)
        w.after_step(4, gpu)
        self.assertEqual(w.steps, 2)

    def test_no_destructive_reset_or_error_masking_in_launchers(self):
        for name in ('night.sh', 'run.sh', 'check.sh', 'nemo_run.sh'):
            text = (ROOT / name).read_text(encoding='utf-8')
            self.assertIn('set -Eeuo pipefail', text)
            self.assertNotIn('rm -rf', text)
            self.assertIn('exec python3', text)


if __name__ == '__main__':
    unittest.main()
