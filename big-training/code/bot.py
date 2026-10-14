"""Telegram bot for watching the training: buttons, progress bars, quality, forecast, GPU — read-only.

    python bot.py [--runs DIR]        (session.py starts it; it can also run on the login node if runs/ is shared)

Who may use it: the owner (TELEGRAM_CHAT_ID) and colleagues. A colleague just writes to the bot; the owner gets a
request with ✅ / ❌ buttons; approved colleagues can press every button but get no messages on their own
(runs/bot_access.json; /users and /remove_<id> for the owner; TELEGRAM_ALLOWED_IDS adds fixed ids).
Alerts and the hourly digest of monitor.py go to the owner only. The bot never changes the training: control
stays on the server (runs/<variant>/STOP, CKPT).
Secrets come from ~/.eagle_secrets (TELEGRAM_BOT_TOKEN); never put the token in code or git.
Only one copy polls Telegram at a time: runs/bot_heartbeat is refreshed every poll, a second copy exits if it is fresh.
"""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import monitor
import settings as S

NAMES = {'baseline': '📄 baseline · слои статьи', 'ours': '🧪 ours · наши слои + fc_norm'}
BUTTONS = [[('📊 Статус', 'status'), ('🎯 Качество', 'quality')],
           [('🔮 Прогноз', 'forecast'), ('🚨 Тревоги', 'alerts')]]
REMOTE = False        # relay.py: data mirrored from ClearML (no process ids, no nvidia-smi of the server)
BAR = '▰▱'
SPARK = '▁▂▃▄▅▆▇█'


# ---------- formatting helpers ----------

def bar(fraction, width=12):
    fraction = max(0.0, min(1.0, fraction))
    full = round(fraction * width)
    return BAR[0] * full + BAR[1] * (width - full)


def spark(values):
    if not values:
        return ''
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return ''.join(SPARK[min(7, int((v - lo) / span * 7.999))] for v in values)


def ago(seconds):
    if seconds is None:
        return '—'
    seconds = max(0, seconds)
    if seconds < 90:
        return f'{seconds:.0f} с назад'
    if seconds < 5400:
        return f'{seconds / 60:.0f} мин назад'
    return f'{seconds / 3600:.1f} ч назад'


def hours(h):
    if h is None:
        return '—'
    return f'{h * 60:.0f} мин' if h < 1 else f'{h:.1f} ч'


def phase_icon(st):
    if st is None:
        return '⚪️ не начинался'
    fresh = time.time() - st['updated'] < monitor.STALL_MINUTES * 60
    if st['phase'] == 'done':
        return '✅ план выполнен'
    if st['phase'] in ('starting', 'running'):
        if fresh:          # judged by the status file: the bot may run on another node than the training
            return '🟡 запускается' if st['phase'] == 'starting' else '🟢 учится'
        if REMOTE:
            return f'🟠 нет вестей {ago(time.time() - st["updated"])}'
        return '🟠 завис?' if monitor.alive(st['pid']) else '🔴 остановился без сохранения'
    return '⏸ ждёт следующей сессии'


def esc(x):
    return html.escape(str(x))


# ---------- views ----------

def read_runs(runs):
    return {v: monitor.read(runs / v / 'status.json') for v in S.VARIANTS}


def table(rows, head):
    """A monospace table for Telegram (<pre>): first column labels, then one column per variant."""
    width = max(len(r[0]) for r in rows + [head])
    cols = [max(len(str(r[i])) for r in rows + [head]) for i in range(1, len(head))]
    fmt = lambda r: r[0].ljust(width) + ''.join('  ' + str(c).rjust(w) for c, w in zip(r[1:], cols))
    return '<pre>' + esc('\n'.join([fmt(head)] + [fmt(r) for r in rows])) + '</pre>'


def view_status(runs):
    statuses = read_runs(runs)
    plan = monitor.read(runs / 'plan.json') or {}
    rows = {k: [k] for k in ('состояние', 'диалогов', 'из', 'готово', 'ток/с', 'осталось GPU', 'чекпоинт')}
    left_by = monitor.hours_left(statuses, plan)        # the second one speeds up once the first stops
    for v, st in statuses.items():
        if st is None:
            for r in rows.values():
                r.append('—')
            continue
        done, stop, _ = monitor.progress(st)          # in dialogues: the two trainers' steps differ
        left = left_by.get(v)
        w = st.get('window') or {}
        speed = w.get('samples_per_s')
        ck = st.get('last_checkpoint')
        rows['состояние'].append(' '.join(phase_icon(st).split()[1:2]))     # a word: emoji break the columns
        rows['диалогов'].append(f'{done / 1e3:,.0f}k'.replace(',', ' '))
        rows['из'].append(f'{stop / 1e3:,.0f}k'.replace(',', ' '))
        rows['готово'].append(f'{done / max(1, stop):.1%}')
        rows['ток/с'].append(f'{w["tokens_per_s"] / 1000:.1f}k' if speed else '—')
        rows['осталось GPU'].append(hours(left))
        rows['чекпоинт'].append(ago(time.time() - ck['time']).replace(' назад', '') if ck else '—')
    lines = ['<b>📊 Статус</b>', table(list(rows.values()), ['', *statuses])]
    any_st = next((s for s in statuses.values() if s), None)
    if any_st and any_st.get('deadline') and any_st['phase'] in ('starting', 'running'):
        lines.append(f'⌛️ сессия закончится через {hours((any_st["deadline"] - time.time()) / 3600)}')
    if plan.get('compare_step'):
        lines.append(f'⚖️ сравнение на {plan["compare_step"] * S.GLOBAL_BATCH / 1e3:,.0f} тыс. диалогов'.replace(',', ' ')
                     + f' · финиш ~{esc(plan["deadline"].split(" (")[0])}')
    return '\n'.join(lines)


def view_quality(runs):
    statuses = read_runs(runs)
    ref = monitor.read(runs / 'reference.json')
    vals = {v: (st or {}).get('last_val') for v, st in statuses.items()}
    head = ['', *statuses] + (['офиц.'] if ref else [])
    rows = [['τ-оценка'] + [f'{x["tau_estimate"]:.2f}' if x else '—' for x in vals.values()]
            + ([f'{ref["tau_estimate"]:.2f}'] if ref else []),
            ['диалогов'] + [f'{x["samples"] / 1e3:,.0f}k'.replace(',', ' ') if x else '—' for x in vals.values()]
            + (['—'] if ref else [])]
    for k in range(7):
        rows.append([f'точн. {k + 1}'] + [f'{x["acc"][k]:.2f}' if x else '—' for x in vals.values()]
                    + ([f'{ref["acc"][k]:.2f}' if ref.get('acc') else '—'] if ref else []))
    lines = ['<b>🎯 Качество</b> <i>(валидация, τ-оценка цепочкой)</i>', table(rows, head)]
    if monitor.compare_line(runs, html=True):              # at equal dialogues, not at the last validations
        lines.append(monitor.compare_line(runs, html=True))
    return '\n'.join(lines)


def view_forecast(runs):
    statuses = read_runs(runs)
    ref = monitor.read(runs / 'reference.json')
    lines = ['<b>🔮 Прогноз</b>', '<i>кривая τ = a − b·N^(−c) по точкам валидации, N — диалоги</i>']
    for v, st in statuses.items():
        lines.append('')
        lines.append(f'<b>{NAMES[v]}</b>')
        hist = monitor.val_history(runs / v)
        fit = monitor.fit_curve([(h['samples'], h['tau_estimate']) for h in hist]) if hist else None
        if st is None or fit is None:
            lines.append(f'   нужно ≥ 4 валидаций (сейчас {len(hist)})')
            continue
        end = monitor.progress(st)[1]
        lines.append(f'   🎯 к концу плана: <b>{monitor.predict(fit, end):.2f}</b>')
        lines.append(f'   📚 после полной эпохи: {monitor.predict(fit, st["train_dialogues"]):.2f}')
        lines.append(f'   🧱 потолок кривой: ~{fit[0]:.2f}')
        if ref:
            need = monitor.samples_to_reach(fit, ref['tau_estimate'])
            lines.append('   🏁 до уровня официального драфта: '
                         + (f'~{need / 1e6:.1f} млн диалогов' if need else 'кривая не дотягивает'))
    return '\n'.join(lines)


def view_curves(runs):
    lines = ['<b>📉 Кривые</b> (последние точки, слева старые)']
    for v in S.VARIANTS:
        lines.append('')
        lines.append(f'<b>{NAMES[v]}</b>')
        rows = []
        path = runs / v / 'metrics.jsonl'
        if path.exists():
            rows = monitor.jsonl(path)[-24:]
        vals = monitor.val_history(runs / v)[-24:]
        if rows:
            loss = [r['ploss'][0] for r in rows]
            speed = [r['tokens_per_s'] for r in rows]
            lines.append(f'   loss шага 1  <code>{spark(loss)}</code> {loss[-1]:.3f}')
            lines.append(f'   ток/с        <code>{spark(speed)}</code> {speed[-1]:,.0f}'.replace(',', ' '))
        if vals:
            tau = [x['tau_estimate'] for x in vals]
            lines.append(f'   τ валидации  <code>{spark(tau)}</code> {tau[-1]:.3f}')
        if not rows and not vals:
            lines.append('   данных пока нет')
    return '\n'.join(lines)


def view_gpu(runs):
    lines = ['<b>🖥 GPU и сервер</b>']
    if REMOTE:
        for v, st in read_runs(runs).items():
            if st:
                lines.append(f'\n<b>{NAMES[v]}</b>\n   🧠 пик памяти GPU {st.get("gpu_memory_gb", 0):.0f} ГБ'
                             f'\n   💽 свободно на диске сервера {st.get("disk_free_gb", 0):.0f} ГБ'
                             f'\n   🔄 данные {ago(time.time() - st["updated"])}')
        lines.append('\n<i>загрузку и температуру GPU смотри в ClearML: вкладка Scalars → :monitor:gpu</i>')
        return '\n'.join(lines)
    if shutil.which('nvidia-smi'):
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,'
                                           'memory.total,temperature.gpu,power.draw',
                                           '--format=csv,noheader,nounits'], text=True, timeout=20)
            for line in out.strip().splitlines():
                i, name, util, used, total, temp, power = [x.strip() for x in line.split(',')]
                lines.append(f'\n<b>GPU {i}</b> · {esc(name)}')
                lines.append(f'   ⚙️ загрузка <code>{bar(float(util) / 100, 10)}</code> {util}%')
                lines.append(f'   🧠 память  <code>{bar(float(used) / float(total), 10)}</code> '
                             f'{float(used) / 1024:.0f} / {float(total) / 1024:.0f} ГБ')
                lines.append(f'   🌡 {temp} °C · 🔌 {float(power):.0f} Вт')
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            lines.append(f'nvidia-smi не ответил: {esc(error)}')
    else:
        lines.append('бот запущен не на GPU-узле — смотрю по статусам обучения:')
        for v, st in read_runs(runs).items():
            if st:
                lines.append(f'   {NAMES[v]}: пик памяти {st.get("gpu_memory_gb", 0):.0f} ГБ')
    free = shutil.disk_usage(runs).free / 2 ** 30
    lines.append(f'\n💽 свободно на диске: <b>{free:.0f} ГБ</b>' + (' ⚠️' if free < 30 else ''))
    return '\n'.join(lines)


def view_alerts(runs):
    path = runs / 'monitor.log'
    if not path.exists():
        return '<b>🚨 Тревоги</b>\n\nпока тихо 🌿'
    blocks = [b for b in path.read_text(encoding='utf-8').split('\n\n') if b.strip()]
    alerts = [b for b in blocks if any(m in b for m in ('🔴', '🟠'))][-6:]
    if not alerts:
        return '<b>🚨 Тревоги</b>\n\nза всё время ни одной 🌿'
    return '<b>🚨 Последние тревоги</b>\n\n' + '\n\n'.join(f'<code>{esc(a.strip())}</code>' for a in alerts)


VIEWS = {'status': view_status, 'quality': view_quality, 'forecast': view_forecast, 'curves': view_curves,
         'gpu': view_gpu, 'alerts': view_alerts}


# ---------- Telegram ----------

class Bot:
    """Owner: TELEGRAM_CHAT_ID (gets alerts and the hourly digest from monitor.py).
    Colleagues: ask by writing to the bot; the owner gets ✅ / ❌ buttons. Approved colleagues can press every button
    but get no messages on their own. Approvals live in runs/bot_access.json; TELEGRAM_ALLOWED_IDS adds fixed ones.
    """

    def __init__(self, token, runs):
        self.api = f'https://api.telegram.org/bot{token}/'
        self.runs = runs
        self.owner = {x.strip() for x in os.environ.get('TELEGRAM_CHAT_ID', '').split(',') if x.strip()}
        self.fixed = {x.strip() for x in os.environ.get('TELEGRAM_ALLOWED_IDS', '').split(',') if x.strip()}
        self.access_file = runs / 'bot_access.json'

    def call(self, method, **params):
        body = urllib.parse.urlencode({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                                       for k, v in params.items()}).encode()
        with urllib.request.urlopen(self.api + method, body, timeout=70) as r:
            return json.loads(r.read().decode())

    def keyboard(self):
        return {'inline_keyboard': [[{'text': t, 'callback_data': d} for t, d in row] for row in BUTTONS]}

    def send(self, chat, text, edit=None, keyboard=None):
        params = {'chat_id': chat, 'text': text[:4096], 'parse_mode': 'HTML', 'disable_web_page_preview': True,
                  'reply_markup': keyboard if keyboard is not None else self.keyboard()}
        try:
            self.call('editMessageText', message_id=edit, **params) if edit else self.call('sendMessage', **params)
        except OSError as error:      # e.g. "message is not modified" when 🔄 shows the same numbers
            print('telegram:', error, flush=True)

    def send_photo(self, chat, png, caption, html=True):
        """A picture with a caption (multipart upload; Telegram limits captions to 1024 characters)."""
        import uuid
        boundary = uuid.uuid4().hex
        fields = {'chat_id': chat, 'caption': caption[:1024], 'reply_markup': json.dumps(self.keyboard())}
        if html:
            fields['parse_mode'] = 'HTML'
        crlf = '\r\n'
        body = b''.join(f'--{boundary}{crlf}Content-Disposition: form-data; name="{k}"{crlf}{crlf}{v}{crlf}'.encode()
                        for k, v in fields.items())
        body += (f'--{boundary}{crlf}Content-Disposition: form-data; name="photo"; filename="dashboard.png"{crlf}'
                 f'Content-Type: image/png{crlf}{crlf}').encode() + png + f'{crlf}--{boundary}--{crlf}'.encode()
        request = urllib.request.Request(self.api + 'sendPhoto', data=body,
                                         headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        try:
            urllib.request.urlopen(request, timeout=60).read()
        except OSError as error:
            print('telegram photo:', error, flush=True)

    # ---- access ----

    def approved(self):
        """{chat id: name} of colleagues the owner let in."""
        return monitor.read(self.access_file) or {}

    def set_approved(self, approved):
        self.access_file.write_text(json.dumps(approved, ensure_ascii=False, indent=1), encoding='utf-8')

    def allowed(self, chat, user):
        ids = self.owner | self.fixed | set(self.approved())
        return str(chat) in ids or str(user) in ids

    def request_access(self, chat, who):
        name = ' '.join(x for x in (who.get('first_name'), who.get('last_name')) if x) or 'без имени'
        handle = f' (@{who["username"]})' if who.get('username') else ''
        self.send(chat, '🔐 Запрос на доступ отправлен владельцу бота. Как только он подтвердит, '
                        'здесь появятся кнопки.', keyboard={'inline_keyboard': []})
        buttons = {'inline_keyboard': [[{'text': '✅ Пустить', 'callback_data': f'allow:{chat}:{name}'[:64]},
                                        {'text': '❌ Отказать', 'callback_data': f'deny:{chat}'}]]}
        for owner in self.owner:
            self.send(owner, f'🔐 <b>{esc(name)}</b>{esc(handle)} просит доступ к боту '
                             f'(id <code>{chat}</code>).\nПосле подтверждения сможет только смотреть, '
                             f'без уведомлений.', keyboard=buttons)

    def decide(self, owner_chat, data, edit):
        action, chat, *name = data.split(':', 2)
        approved = self.approved()
        if action == 'allow':
            approved[chat] = name[0] if name else chat
            self.set_approved(approved)
            self.send(owner_chat, f'✅ Доступ открыт: {esc(approved[chat])}', edit=edit, keyboard={'inline_keyboard': []})
            self.send(chat, '✅ Доступ открыт! Жми кнопки 👇\n<i>Только просмотр, уведомления не приходят.</i>')
        else:
            approved.pop(chat, None)
            self.set_approved(approved)
            self.send(owner_chat, f'❌ Отказано (id <code>{chat}</code>)', edit=edit, keyboard={'inline_keyboard': []})

    def users(self):
        approved = self.approved()
        if not approved:
            return '👥 Доступ есть только у тебя.'
        rows = '\n'.join(f'• {esc(n)} — <code>{c}</code>  /remove_{c}' for c, n in approved.items())
        return f'👥 <b>Кто может смотреть бота</b>\n\n{rows}'

    # ---- commands ----

    def answer(self, chat, user, command, edit=None, who=None):
        is_owner = str(chat) in self.owner or str(user) in self.owner
        if command and command.startswith(('allow:', 'deny:')):
            if is_owner:
                self.decide(chat, command, edit)
            return
        if not self.allowed(chat, user):
            self.request_access(chat, who or {})
            return
        command = command.lstrip('/').split('@')[0].split()[0] if command else 'start'
        if command in ('start', 'help', 'menu'):
            text = ('<b>🦅 EAGLE-3 · обучение драфтов с нуля</b>\n\n'
                    '📄 <b>baseline</b> — слои статьи (2, 16, 29)\n'
                    '🧪 <b>ours</b> — наши слои (16, 22, 29) + fc_norm\n\n'
                    '📊 <b>Статус</b> — дашборд и прогресс\n'
                    '🎯 <b>Качество</b> — τ и точность по шагам драфта\n'
                    '🔮 <b>Прогноз</b> — чего ждать к концу обучения\n'
                    '🚨 <b>Тревоги</b> — последние предупреждения\n\n'
                    'Жми кнопки 👇'
                    + ('\n\n<i>Для владельца: /users — кто имеет доступ</i>' if is_owner else ''))
        elif command == 'all':
            text = '\n\n━━━━━━━━━━\n\n'.join(VIEWS[k](self.runs) for k in ('status', 'quality', 'forecast'))
        elif command == 'status':                  # the dashboard picture with the status table under it
            import dashboard
            self.send_photo(chat, dashboard.render(self.runs), view_status(self.runs))
            return
        elif command in VIEWS:
            text = VIEWS[command](self.runs)
        elif command == 'users' and is_owner:
            text = self.users()
        elif command.startswith('remove_') and is_owner:
            approved = self.approved()
            gone = approved.pop(command[len('remove_'):], None)
            self.set_approved(approved)
            text = f'🚫 Доступ закрыт: {esc(gone)}' if gone else 'Такого id в списке нет'
        else:
            text = 'Не знаю такую команду 🤷 /help'
        self.send(chat, text, edit=edit)

    def run(self):
        offset, heartbeat = None, self.runs / 'bot_heartbeat'
        while True:
            if heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < 120 and \
                    heartbeat.read_text().strip() != str(os.getpid()):
                print('another copy of the bot is running; exiting', flush=True)
                return
            heartbeat.write_text(str(os.getpid()))
            try:
                updates = self.call('getUpdates', timeout=50, offset=offset or 0,
                                    allowed_updates=['message', 'callback_query'])['result']
            except OSError as error:
                print('telegram:', error, flush=True)
                time.sleep(15)
                continue
            for u in updates:
                offset = u['update_id'] + 1
                try:
                    if 'callback_query' in u:
                        q = u['callback_query']
                        self.call('answerCallbackQuery', callback_query_id=q['id'])
                        # under a picture there is no text to edit: answer with a new message
                        edit = None if 'photo' in q['message'] else q['message']['message_id']
                        self.answer(q['message']['chat']['id'], q['from']['id'], q['data'], edit=edit, who=q['from'])
                    elif 'message' in u and 'text' in u['message']:
                        m = u['message']
                        self.answer(m['chat']['id'], m['from']['id'], m['text'], who=m['from'])
                except Exception as error:          # noqa: BLE001 - one bad update must not stop the bot
                    print('update failed:', repr(error), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=str(S.RUNS))
    args = parser.parse_args()
    from session import load_secrets
    load_secrets()
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        sys.exit('TELEGRAM_BOT_TOKEN is not set (~/.eagle_secrets)')
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    Bot(token, runs).run()


if __name__ == '__main__':
    main()
