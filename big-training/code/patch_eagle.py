"""Patched copies of the authors' code (SafeAILab/EAGLE, pinned commit) for training and for evaluation.

    python patch_eagle.py [EAGLE repo] [work dir]

work/train  = eagle/traineagle3 (the trainer's model code), work/eval/eagle = the eagle package (generation).
Every edit is an exact text replacement that must match once, so a different EAGLE version fails loudly.
With EAGLE_CAPTURE_LAYERS unset and fc_norm absent from the draft config, both copies behave exactly as the
authors' code; the edits only add what our variant needs:

    capture layers   modeling_llama_kv.py hard-codes the target layers 2, N/2, N-3; now they come from
                     EAGLE_CAPTURE_LAYERS ("16,22,29"; index i = input of layer i), default unchanged
    fc_norm          if the draft config has "fc_norm": true, one RMSNorm per captured state before fc
                     (EAGLE-3.1 fc_norm, the layout of NeMo / vLLM)
    bug fix          traineagle3/cnets.py reads self.train_config.gradient_checkpointing and .max_len, but
                     main.py passes a dict: the published trainer crashes (AttributeError)
    loading check    eagle/model/ea_model.py loads the draft with strict=False: a draft whose fc_norm weights
                     were not picked up would silently run without them. For our drafts (config has
                     "eagle_capture_layers") every weight must load
"""
import shutil
import sys
from pathlib import Path

from settings import EAGLE_REPO, WORK

CAPTURE_HELPER = '''import os as _os


def capture_layers(n_layers):
    """Target states fed to the draft (index i = input of layer i). EAGLE-3's default: 2, N/2, N-3."""
    spec = _os.environ.get("EAGLE_CAPTURE_LAYERS")
    if not spec:
        return [2, n_layers // 2, n_layers - 3]
    layers = [int(x) for x in spec.split(",")]
    if len(layers) != 3 or sorted(set(layers)) != layers or not 0 <= layers[0] < layers[-1] < n_layers:
        raise ValueError(f"EAGLE_CAPTURE_LAYERS must be 3 increasing layer ids, got {spec}")
    return layers


'''
CAPTURE_OLD = '            if idx==len(self.layers)-3 or idx==len(self.layers)//2 or idx==2:\n'
CAPTURE_NEW = '            if idx in capture_layers(len(self.layers)):\n'
MODULE_START = '""" PyTorch LLaMA model."""\n'

def fc_norm_apply(indent):
    return (f'{indent}if hasattr(self, "fc_norm"):   # EAGLE-3.1 fc_norm: each captured state normalised on its own\n'
            f'{indent}    parts = hidden_states.chunk(3, dim=-1)\n'
            f'{indent}    hidden_states = torch.cat([norm(part) for norm, part in zip(self.fc_norm, parts)], dim=-1)\n')


def fc_norm_modules(indent):
    return (f'{indent}if getattr(config, "fc_norm", False):\n'
            f'{indent}    self.fc_norm = nn.ModuleList(\n'
            f'{indent}        [LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(3)])\n')


def replace_once(source, old, new, where):
    if source.count(old) != 1:
        raise ValueError(f'{where}: pinned EAGLE source changed near {old.strip()[:70]!r}')
    return source.replace(old, new, 1)


def patch_capture(source, where):
    source = replace_once(source, MODULE_START, MODULE_START + CAPTURE_HELPER, where)
    return replace_once(source, CAPTURE_OLD, CAPTURE_NEW, where)


def patch_train_cnets(s):
    w = 'traineagle3/cnets.py'
    s = replace_once(s, 'self.train_config.gradient_checkpointing', 'self.train_config["gradient_checkpoint"]', w)
    s = replace_once(s, 'self.train_config.max_len', 'self.train_config["max_len"]', w)
    fc = '        self.fc=nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)\n'
    s = replace_once(s, fc, fc + fc_norm_modules('        '), w)
    apply = '        hidden_states=self.fc(hidden_states)\n'
    return replace_once(s, apply, fc_norm_apply('        ') + apply, w)


def patch_eval_cnets(s):
    w = 'model/cnets.py'
    fc = ('            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)\n'
          '        self.norm=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)\n')
    s = replace_once(s, fc, fc + fc_norm_modules('        '), w)
    apply = ('        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:\n'
             '            hidden_states = self.fc(hidden_states)\n')
    new = ('        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:\n'
           + fc_norm_apply('            ') + '            hidden_states = self.fc(hidden_states)\n')
    return replace_once(s, apply, new, w)


def patch_ea_model(s):
    w = 'model/ea_model.py'
    old = '        config = EConfig.from_pretrained(ea_model_path)\n'
    new = old + ('        if hasattr(config, "eagle_capture_layers"):   # our drafts say which target states they use\n'
                 '            os.environ["EAGLE_CAPTURE_LAYERS"] = ",".join(str(i) for i in config.eagle_capture_layers)\n')
    s = replace_once(s, old, new, w)
    old = '        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)\n'
    new = old + ('        if hasattr(config, "eagle_capture_layers"):   # our drafts: every weight must be used\n'
                 '            missing = [k for k in load_.missing_keys\n'
                 '                       if not k.startswith("embed_tokens") and not k.endswith("inv_freq")]\n'
                 '            unexpected = [k for k in load_.unexpected_keys   # d2t/t2d are dropped above\n'
                 '                          if not (k in ("d2t", "t2d") and config.vocab_size == config.draft_vocab_size)]\n'
                 '            if missing or unexpected:\n'
                 '                raise ValueError(f"Draft weights do not match the model: missing {missing}, "\n'
                 '                                 f"unexpected {unexpected}")\n')
    s = replace_once(s, old, new, w)
    if '\nimport os\n' not in s:
        s = 'import os\n' + s
    return s


def build(repo=EAGLE_REPO, work=WORK):
    repo, work = Path(repo), Path(work)
    train = work / 'train'
    if train.exists():
        shutil.rmtree(train)
    shutil.copytree(repo / 'eagle/traineagle3', train, ignore=shutil.ignore_patterns('__pycache__'))
    (train / 'cnets.py').write_text(patch_train_cnets((train / 'cnets.py').read_text(encoding='utf-8')),
                                    encoding='utf-8')
    path = train / 'modeling_llama_kv.py'
    path.write_text(patch_capture(path.read_text(encoding='utf-8'), 'traineagle3/modeling_llama_kv.py'),
                    encoding='utf-8')

    evaluation = work / 'eval'
    if evaluation.exists():
        shutil.rmtree(evaluation)
    shutil.copytree(repo / 'eagle', evaluation / 'eagle', ignore=shutil.ignore_patterns('__pycache__'))
    model = evaluation / 'eagle/model'
    (model / 'cnets.py').write_text(patch_eval_cnets((model / 'cnets.py').read_text(encoding='utf-8')),
                                    encoding='utf-8')
    path = model / 'modeling_llama_kv.py'
    path.write_text(patch_capture(path.read_text(encoding='utf-8'), 'model/modeling_llama_kv.py'), encoding='utf-8')
    (model / 'ea_model.py').write_text(patch_ea_model((model / 'ea_model.py').read_text(encoding='utf-8')),
                                       encoding='utf-8')
    for f in [train / 'cnets.py', train / 'modeling_llama_kv.py', model / 'cnets.py',
              model / 'modeling_llama_kv.py', model / 'ea_model.py']:
        compile(f.read_text(encoding='utf-8'), str(f), 'exec')
    print('patched copies of the authors\' code in', work)


if __name__ == '__main__':
    build(*sys.argv[1:])
