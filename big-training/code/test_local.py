"""CPU tests without the server: formulas, forecast, monitor, SLURM time parsing, export names, DeepSpeed config.

    BT_WORK=<patched work dir> py -3 -X utf8 test_local.py
"""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import monitor
import settings as S


class Formulas(unittest.TestCase):
    def test_tau_estimate(self):
        from train import tau_estimate
        self.assertEqual(tau_estimate([0.0] * 7), 1.0)
        self.assertEqual(tau_estimate([1.0] * 7), 8.0)
        self.assertAlmostEqual(tau_estimate([0.5, 0.5]), 1.75)

    def test_variants(self):
        self.assertEqual(S.VARIANTS['baseline'], {'layers': [2, 16, 29], 'fc_norm': False})
        self.assertEqual(S.VARIANTS['ours'], {'layers': [16, 22, 29], 'fc_norm': True})

    def test_fit_recovers_curve(self):
        xs = [2e4 * k for k in range(1, 12)]
        ys = [5.0 - 40 * x ** -0.4 for x in xs]
        fit = monitor.fit_curve(list(zip(xs, ys)))
        self.assertAlmostEqual(fit[0], 5.0, delta=0.05)
        self.assertAlmostEqual(monitor.predict(fit, 1e6), 5.0 - 40 * 1e6 ** -0.4, delta=0.03)
        need = monitor.samples_to_reach(fit, 4.0)
        self.assertAlmostEqual(need, (40 / 1.0) ** (1 / 0.4), delta=need * 0.1)
        self.assertIsNone(monitor.samples_to_reach(fit, 6.0))
        self.assertIsNone(monitor.fit_curve([(1, 1), (2, 2)]))

    def test_export_names(self):
        from export import nemo_name
        self.assertEqual(nemo_name('midlayer.self_attn.q_proj.weight'), 'model.layers.0.self_attn.q_proj.weight')
        self.assertEqual(nemo_name('midlayer.hidden_norm.weight'), 'model.layers.0.hidden_norm.weight')
        self.assertEqual(nemo_name('fc.weight'), 'model.fc.weight')
        self.assertEqual(nemo_name('fc_norm.2.weight'), 'model.fc_norm.2.weight')
        self.assertEqual(nemo_name('norm.weight'), 'model.norm.weight')
        self.assertEqual(nemo_name('lm_head.weight'), 'lm_head.weight')
        self.assertEqual(nemo_name('d2t'), 'd2t')


class Session(unittest.TestCase):
    def test_slurm_time_left(self):
        import session
        for left, seconds in (('7:59:30', 7 * 3600 + 59 * 60 + 30), ('1-02:00:00', 26 * 3600), ('45:00', 2700)):
            with patch.dict(os.environ, {'SLURM_JOB_ID': '1'}), \
                    patch('subprocess.check_output', return_value=left + '\n'):
                self.assertAlmostEqual(session.session_end() - time.time(), seconds, delta=2)


class DeepSpeedConfig(unittest.TestCase):
    def test_authors_config_with_our_changes(self):
        from train import deepspeed_config
        cfg = deepspeed_config(150000)
        self.assertEqual(cfg['gradient_accumulation_steps'], 8)
        self.assertEqual(cfg['scheduler']['params']['total_num_steps'], 150000)
        self.assertEqual(cfg['scheduler']['params']['warmup_num_steps'], 12000)
        self.assertEqual(deepspeed_config(50000)['scheduler']['params']['warmup_num_steps'], 5000)
        self.assertTrue(cfg['fp16']['enabled'] in (True, 'true'))
        self.assertEqual(cfg['gradient_clipping'], 0.5)


class Monitor(unittest.TestCase):
    def status(self, **kw):
        base = {'variant': 'baseline', 'phase': 'running', 'pid': os.getpid(), 'updated': time.time(), 'step': 1000,
                'total_steps': 100000, 'epoch': 0, 'position': 8000, 'train_dialogues': 1400000, 'samples': 8000,
                'tokens': 3e6, 'nonfinite': 0, 'skipped_steps': 3, 'lr': 1e-5, 'loss_scale': 1024.0,
                'window': {'samples_per_s': 5.0, 'tokens_per_s': 2200}, 'last_val': None, 'best_val_tau': None,
                'last_checkpoint': None, 'deadline': None, 'train_seconds': 100, 'gpu_memory_gb': 38}
        base.update(kw)
        return base

    def run_check(self, statuses, previous=None):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for v, st in statuses.items():
                (runs / v).mkdir()
                (runs / v / 'status.json').write_text(json.dumps(st))
            n = monitor.Notifier(runs)
            monitor.check(runs, n, previous or {})
            text = (runs / 'monitor.log').read_text(encoding='utf-8') if (runs / 'monitor.log').exists() else ''
            return text, monitor.summary(runs, {v: st for v, st in statuses.items()})

    def test_healthy_run_is_quiet(self):
        text, summary = self.run_check({'baseline': self.status(), 'ours': self.status(variant='ours')})
        self.assertEqual(text, '')
        self.assertIn('шаг 1 000 из 100 000', summary)

    def test_alerts(self):
        dead = self.status(pid=999999999)
        stalled = self.status(variant='ours', updated=time.time() - 3600, loss_scale=1.0, nonfinite=50)
        text, _ = self.run_check({'baseline': dead, 'ours': stalled}, previous={'ours': self.status(variant='ours')})
        self.assertIn('процесс обучения пропал', text)
        self.assertIn('нет нового шага', text)
        self.assertIn('loss scale', text)
        self.assertIn('NaN/inf', text)

    def test_forecast_in_summary(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            (runs / 'ours').mkdir()
            vals = [{'step': s, 'samples': 8 * s, 'tau_estimate': 4.5 - 30 * (8 * s) ** -0.4, 'acc': [0.7] * 7}
                    for s in range(2000, 20001, 2000)]
            (runs / 'ours' / 'val.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in vals))
            (runs / 'reference.json').write_text(json.dumps({'tau_estimate': 4.2}))
            st = self.status(variant='ours', last_val=vals[-1], best_val_tau=vals[-1]['tau_estimate'])
            text = monitor.summary(runs, {'baseline': None, 'ours': st})
            self.assertIn('к концу плана', text)
            self.assertIn('официальный драфт: 4.20', text)


class TelegramBot(unittest.TestCase):
    """bot.py with a fake Telegram API: every view renders, access is checked, subscriptions toggle."""

    def setUp(self):
        import bot
        self.bot_module = bot
        self.dir = tempfile.TemporaryDirectory()
        runs = Path(self.dir.name)
        st = Monitor().status()
        for v, shift in (('baseline', 0.0), ('ours', 0.08)):
            (runs / v).mkdir()
            vals = [{'step': s, 'samples': 8 * s, 'tau_estimate': 4.5 + shift - 30 * (8 * s) ** -0.4,
                     'acc': [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4]} for s in range(2000, 20001, 2000)]
            (runs / v / 'val.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in vals))
            (runs / v / 'metrics.jsonl').write_text(''.join(json.dumps(
                {'step': 10 * i, 'ploss': [3 - i * 0.05] * 7, 'tokens_per_s': 4000 + i}) + '\n' for i in range(30)))
            (runs / v / 'status.json').write_text(json.dumps(dict(
                st, variant=v, last_val=vals[-1], best_val_tau=vals[-1]['tau_estimate'],
                last_checkpoint={'time': time.time() - 600, 'step': 900}, deadline=time.time() + 5 * 3600)))
        (runs / 'reference.json').write_text(json.dumps({'tau_estimate': 4.4}))
        (runs / 'plan.json').write_text(json.dumps({'deadline': '2026-10-15 10:00', 'total_steps': 100000,
                                                    'share_of_data': 0.57}))
        (runs / 'monitor.log').write_text('2026 🟠 baseline: no new step for 25 min\n\n2026 summary\n\n')
        sent = []
        with patch.dict(os.environ, {'TELEGRAM_CHAT_ID': '1', 'TELEGRAM_ALLOWED_IDS': '2,-100'}):
            self.bot = bot.Bot('TOKEN', runs)
        self.bot.call = lambda method, **p: sent.append((method, p)) or {'result': []}
        self.bot.send_photo = lambda chat, png, caption, html=True: sent.append(
            ('sendPhoto', {'chat_id': str(chat), 'text': caption, 'png': len(png)}))
        self.sent, self.runs = sent, runs

    def tearDown(self):
        self.dir.cleanup()

    def test_every_view_renders(self):
        for command in ('start', 'status', 'quality', 'forecast', 'curves', 'gpu', 'alerts', 'all'):
            self.sent.clear()
            self.bot.answer(1, 1, '/' + command)
            text = self.sent[-1][1]['text']
            self.assertNotIn('Traceback', text)
            self.assertLessEqual(len(text), 4096)
        self.bot.answer(1, 1, '/quality')
        self.assertIn('ours − baseline: +0.08', self.sent[-1][1]['text'])
        self.bot.answer(1, 1, '/forecast')
        self.assertIn('до уровня официального драфта', self.sent[-1][1]['text'])

    def test_access_request_and_approval(self):
        self.bot.answer(777, 777, '/status', who={'first_name': 'Иван', 'username': 'ivan'})
        texts = {p['chat_id']: p['text'] for _, p in self.sent}
        self.assertIn('Запрос на доступ отправлен', texts[777])
        self.assertIn('Иван', texts['1'])                          # the owner got the request with buttons
        request = [p for _, p in self.sent if p['chat_id'] == '1'][-1]
        allow = request['reply_markup']['inline_keyboard'][0][0]['callback_data']
        self.bot.answer(555, 555, allow)                             # not the owner: ignored
        self.assertFalse(self.bot.allowed(777, 777))
        self.bot.answer(1, 1, allow, edit=10)                        # the owner presses ✅
        self.assertTrue(self.bot.allowed(777, 777))
        self.bot.answer(777, 777, '/status')
        self.assertIn('📊 Статус', self.sent[-1][1]['text'])
        self.bot.answer(1, 1, '/remove_777')
        self.assertFalse(self.bot.allowed(777, 777))

    def test_messages_go_to_owner_only(self):
        with patch.dict(os.environ, {'TELEGRAM_CHAT_ID': '1'}):
            self.assertEqual(monitor.Notifier(self.runs).chats(), ['1'])
        self.bot.answer(-100, 555, '/status')                       # a fixed allowed group chat still works
        self.assertIn('📊 Статус', self.sent[-1][1]['text'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
