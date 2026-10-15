"""96 fixed prompts, 16 per domain: general / math / code / medicine (from prompts_base.jsonl: the held-out
test+validation splits of the geometry pilot), tool use (ToolACE: its system prompt with the tool list + the first
user turn) and Russian (m-ArenaHard ru). Writes prompts.jsonl. Only prompts, never reference answers.
"""
import json
import random

import pandas as pd
from huggingface_hub import hf_hub_download

from config import CODE

N = 16
rows = [json.loads(x) for x in (CODE / "prompts_base.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

tool = json.load(open(hf_hub_download("Team-ACE/ToolACE", "data.json", repo_type="dataset"), encoding="utf-8"))
rng = random.Random(0)
picked = []
for i in rng.sample(range(len(tool)), 400):
    ex = tool[i]
    first = next((t for t in ex["conversations"] if t["from"] == "user"), None)
    if first and len(ex["system"]) < 4000:
        picked.append({"id": f"tool:{i}", "domain": "tool",
                       "messages": [{"role": "system", "content": ex["system"]},
                                    {"role": "user", "content": first["value"]}]})
    if len(picked) == N:
        break
rows += picked

ru = pd.read_parquet(hf_hub_download("CohereLabs/m-ArenaHard", "ru/test-00000-of-00001.parquet", repo_type="dataset"))
ru = ru.sort_values("question_id").iloc[:N]
rows += [{"id": f"ru:{r.question_id}", "domain": "ru", "messages": [{"role": "user", "content": r.prompt}]}
         for r in ru.itertuples()]

(CODE / "prompts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(len(rows), {d: sum(r["domain"] == d for r in rows) for d in sorted({r["domain"] for r in rows})})
