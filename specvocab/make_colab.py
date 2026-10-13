"""Builds colab.ipynb (run: python make_colab.py). The notebook clones this branch, keeps data, models and results
on Google Drive (so a disconnected session resumes), and runs the same scripts as run.sh, one step per cell."""
import json
from pathlib import Path

BRANCH = "specvocab-languages"
REPO = "https://github.com/stepankulibaba/yandex-research-program-eagle3"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md(f"""
# SpecVocab на языках — Colab

Динамический словарь драфта EAGLE-3 ([SpecVocab, arXiv:2602.13836](https://arxiv.org/abs/2602.13836)) против фиксированных 32K токенов на Qwen3-1.7B, проверка на шести языках. Подробности — [README]({REPO}/tree/{BRANCH}/specvocab).

**Запуск:** `Среда выполнения → Сменить тип среды → GPU` (лучше **A100**; **L4** подойдёт с двумя вариантами вместо трёх; T4 не подходит), затем `Среда выполнения → Выполнить все`.

**Если сессия оборвалась** — снова «Выполнить все»: данные, модели и результаты лежат на Google Drive, каждый шаг пропускает сделанное, обучение продолжается с последнего чекпоинта.
""")

md("## 1. Настройки")
code("""
#@title Настройки { display-mode: "form" }
DATA = "en"  #@param ["en", "ml"]
#@markdown `en` — английские данные, как в наших прошлых запусках; `ml` — плюс многоязычные запросы (oasst2).
EPOCHS = 1  #@param {type:"integer"}
#@markdown Эпох обучения (8000 шагов каждая): 1 укладывается в сессию Colab; в прошлых запусках было 3.
SMOKE = False  #@param {type:"boolean"}
#@markdown `SMOKE` — 15-минутная проверка всех шагов на 24 текстах.
DRIVE_DIR = "/content/drive/MyDrive/specvocab_run"  #@param {type:"string"}
""")

md("## 2. GPU, Google Drive и код")
code(f"""
import os, shutil, subprocess, torch
assert torch.cuda.is_available(), "Нужна GPU: Среда выполнения → Сменить тип среды → GPU"
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {{gpu}}, {{mem:.0f}} ГБ")
assert mem > 20, "Нужна GPU от 24 ГБ (A100 или L4); T4 не подходит"

from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)

if not os.path.isdir("/content/repo"):
    subprocess.run(["git", "clone", "-q", "-b", "{BRANCH}", "{REPO}", "/content/repo"], check=True)
else:
    subprocess.run(["git", "-C", "/content/repo", "pull", "-q"], check=True)
WORK = "/content/repo/specvocab"
os.chdir(WORK)
for d in ["data", "data_smoke", "models", "results", "logs", "EAGLE"]:     # all on Drive: survives a disconnect
    target = os.path.join(DRIVE_DIR, d)
    os.makedirs(target, exist_ok=True)
    if os.path.islink(d):
        os.unlink(d)
    if not os.path.exists(d):
        os.symlink(target, d)

# the resume checkpoint is ~12 GB (three variants with optimiser state): on Drive if there is room
free = shutil.disk_usage(DRIVE_DIR).free / 2**30
ckpt_dir = os.path.join(DRIVE_DIR, "checkpoints") if free > 30 else "/content/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
print(f"на Drive свободно {{free:.0f}} ГБ; чекпоинт продолжения: {{ckpt_dir}}")
if free <= 30:
    print("   на Drive мало места: чекпоинт на локальном диске, после обрыва сессии обучение начнётся заново")

big = mem > 34                          # A100: all three variants; L4 (24 GB): without the full head
VARIANTS = "static32k,full,specvocab" if big else "static32k,specvocab"
env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           SPECVOCAB_CKPT_DIR=ckpt_dir, SPECVOCAB_GEN_BATCH="96" if big else "32",
           SPECVOCAB_LOSS_CHUNK="128" if big else "64")
if SMOKE:
    env.update(SPECVOCAB_DATA="data_smoke", SPECVOCAB_RESULTS="results/smoke")
D = env.get("SPECVOCAB_DATA", "data")
RES = os.path.join(env.get("SPECVOCAB_RESULTS", "results"), DATA)
LIMIT = ["--limit", "24"] if SMOKE else []
print("варианты:", VARIANTS)


def run(args, log, tail=None):
    \"\"\"python <args> in the package folder: output live (and to logs/<log>), stops the notebook on an error.\"\"\"
    lines = []
    with open(os.path.join("logs", log), "a", encoding="utf-8") as f:
        p = subprocess.Popen(["python", *args], cwd=WORK, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line)
            if "Warning" in line or "warn(" in line:
                continue
            lines.append(line)
            if tail is None:
                print(line, end="")
    if tail:
        print("".join(lines[-tail:]), end="")
    assert p.wait() == 0, f"{{args[0]}} упал, лог: logs/{{log}}"
""")

md("## 3. Пакеты\nPyTorch в Colab уже есть; ставим остальное в версиях наших запусков и проверяем код на CPU.")
code("""
reqs = [r for r in open("requirements.txt").read().split() if not r.startswith("torch")]
subprocess.run(["pip", "install", "-q", *reqs], check=True)
run(["tests/test_cpu.py"], "test_cpu.log", tail=2)
""")

md("## 4. Вопросы бенчмарков (репозиторий EAGLE) и модели (~4 ГБ)")
code("""
if not os.path.exists("EAGLE/eagle/data/mt_bench/question.jsonl"):
    src = os.path.join(DRIVE_DIR, "EAGLE")
    shutil.rmtree(src, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "https://github.com/SafeAILab/EAGLE", src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "-q", "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"], check=True)
from huggingface_hub import snapshot_download
for repo, path in [("Qwen/Qwen3-1.7B", "models/qwen3-1.7b/target"),
                   ("AngelSlim/Qwen3-1.7B_eagle3", "models/qwen3-1.7b/draft")]:
    if not os.path.exists(os.path.join(path, "config.json")):
        snapshot_download(repo, local_dir=path, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"])
print("вопросы и модели на месте")
""")

md("## 5. Тексты: ответы самой модели\n240 английских вопросов, 660 языковых (m-ArenaHard + MGSM на en, ru, uk, de, zh, ja) и 8000 обучающих. На A100 ~1–1,5 ч.")
code("""
if not os.path.exists(f"{D}/eval_en.pt"):
    run(["prepare_data.py", "eval", *LIMIT], "prepare_eval.log", tail=3)
if not os.path.exists(f"{D}/eval_lang.pt"):
    run(["prepare_data.py", "lang", *LIMIT], "prepare_lang.log", tail=3)
if not os.path.exists(f"{D}/train_{DATA}.pt"):
    run(["prepare_data.py", "train", *(["--ml"] if DATA == "ml" else []), *LIMIT], f"prepare_train_{DATA}.log", tail=3)
print("тексты готовы")
""")

md("## 6. Проверка: наш код воспроизводит официальный драфт\nДолжно быть `SANITY OK`, иначе ноутбук остановится.")
code("""
run(["sanity.py"], "sanity.log", tail=3)
""")

md("## 7. Обучение\nСтрока каждые 100 шагов (скорость, память, loss), оценка τ каждые 2000 шагов. После обрыва сессии этот же код продолжит с чекпоинта.")
code("""
train_args = ["--steps-per-epoch", "60", "--epochs", "1", "--eval-every", "30", "--ckpt-every", "40"] if SMOKE \\
    else ["--epochs", str(EPOCHS), "--ckpt-every", "1000"]
if not os.path.exists(f"{RES}/specvocab.pt"):
    run(["train.py", DATA, "--variants", VARIANTS, *train_args], f"train_{DATA}.log")
else:
    print("уже обучено")
""")

md("## 8. Итоговая оценка, стоимость головы, таблицы и графики")
code("""
if not os.path.exists(f"{RES}/final.json"):
    run(["evaluate.py", DATA], f"evaluate_{DATA}.log", tail=12)
run(["head_speed.py"], "head_speed.log", tail=3)
run(["analyze.py", DATA], f"analyze_{DATA}.log", tail=0)
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{RES}/summary.md", encoding="utf-8").read().split("![")[0]))
for name in ["fig_tau_by_language.png", "fig_specvocab_k.png", "fig_training.png"]:
    if os.path.exists(f"{RES}/{name}"):
        display(Image(f"{RES}/{name}"))
""")

md("## 9. Архив результатов (без весов): на Drive и для скачивания")
code("""
import glob, tarfile
out = os.path.join(DRIVE_DIR, f"results_{DATA}{'_smoke' if SMOKE else ''}.tgz")
with tarfile.open(out, "w:gz") as tar:
    for f in glob.glob(f"{RES}/*.json") + glob.glob(f"{RES}/*.md") + glob.glob(f"{RES}/*.png") + glob.glob("logs/*.log"):
        tar.add(f)
print(out, os.path.getsize(out) // 1024, "КБ")
from google.colab import files
files.download(out)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
Path(__file__).with_name("colab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote colab.ipynb")
