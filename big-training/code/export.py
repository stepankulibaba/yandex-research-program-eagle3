"""A trained draft (train.py milestone, the authors' layout) -> directories for SGLang and for NeMo / vLLM.

    python export.py runs/ours/milestones/step0100000 exports/ours

Writes <out>/sglang/ and <out>/nemo/, each config.json + model.safetensors.

Target layers. We count "input of layer i" (= HF hidden_states[i], the authors' convention), SGLang and NeMo count
"output of layer i": eagle_aux_hidden_state_layer_ids = our ids - 1. So the baseline 2, 16, 29 is [1, 15, 28]
(exactly SGLang's and NeMo's defaults) and ours 16, 22, 29 is [15, 21, 28].

sglang/  the authors' key names (midlayer.*, fc.*, norm.*, lm_head.*, d2t, t2d), which SGLang 0.5.9 loads;
         architectures = LlamaForCausalLMEagle3. With fc_norm, run SGLang with sglang_eagle31 (see README).
nemo/    the layout of NeMo's LlamaEagle3 draft and of vLLM's EAGLE-3.1: model.layers.0.*, model.fc.*,
         model.fc_norm.{0,1,2}.weight, model.norm.*, lm_head.*, plus "fc_norm": true in config.json.
"""
import json
from pathlib import Path
import sys


def nemo_name(key):
    for old, new in (('midlayer.', 'model.layers.0.'), ('fc_norm.', 'model.fc_norm.'), ('fc.', 'model.fc.'),
                     ('norm.', 'model.norm.')):
        if key.startswith(old):
            return new + key[len(old):]
    return key                                    # lm_head.weight, d2t, t2d


def main(milestone, out):
    import torch
    from safetensors.torch import save_file
    milestone, out = Path(milestone), Path(out)
    config = json.loads((milestone / 'config.json').read_text(encoding='utf-8'))
    weights = torch.load(milestone / 'pytorch_model.bin', map_location='cpu')
    layers = config['eagle_capture_layers']
    if config['fc_norm'] != any(k.startswith('fc_norm.') for k in weights):
        raise ValueError('fc_norm in config.json and in the weights disagree')
    config.update({'architectures': ['LlamaForCausalLMEagle3'], 'num_aux_hidden_states': 3,
                   'target_hidden_size': config['hidden_size'],
                   'eagle_config': {'eagle_aux_hidden_state_layer_ids': [i - 1 for i in layers],
                                    'use_aux_hidden_state': True}})
    for name, rename in (('sglang', lambda k: k), ('nemo', nemo_name)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        save_file({rename(k): v.contiguous() for k, v in weights.items()}, str(d / 'model.safetensors'))
        (d / 'config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')
    print(f'exported {milestone} -> {out}/sglang, {out}/nemo; target layers (output ids) {[i - 1 for i in layers]}')


if __name__ == '__main__':
    main(*sys.argv[1:])
