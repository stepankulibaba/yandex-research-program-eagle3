"""GPU check of the patched authors' code on a tiny random 32-layer Llama (a minute; run before the smoke test).

    python test_tiny_gpu.py [work dir] [tokenizer dir]      (defaults: work/ and the target model)

1. the target states the draft gets are HF hidden_states[i] for each variant's layers (and not a neighbour);
2. fc_norm exists only in "ours", gets gradients, and a few training steps lower the loss for both variants;
3. a draft saved the way train.py exports it loads into the authors' generation code (EaModel) and generates;
   the same draft without its fc_norm weights is refused instead of silently running without them.
"""
import os
import sys
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import settings as S                                     # noqa: E402
WORK = sys.argv[1] if len(sys.argv) > 1 else str(S.WORK)
TOKENIZER = sys.argv[2] if len(sys.argv) > 2 else str(S.TARGET)
sys.path.insert(0, WORK + '/train')
torch.manual_seed(0)
tmp = tempfile.mkdtemp()
tiny = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=32, num_attention_heads=4,
                   num_key_value_heads=2, vocab_size=1000, max_position_embeddings=512, rms_norm_eps=1e-5)
reference = LlamaForCausalLM(tiny).eval()
with torch.no_grad():
    reference.lm_head.weight.mul_(30)      # a confident target, so there is something to learn
reference.save_pretrained(tmp)
import json as _json
_json.dump({'weight_map': {'model.embed_tokens.weight': 'model.safetensors'}}, open(tmp + '/model.safetensors.index.json', 'w'))

from cnets import Model            # noqa: E402
from configs import EConfig        # noqa: E402

draft_cfg = EConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=1000, draft_vocab_size=1000, max_position_embeddings=512,
                    rms_norm_eps=1e-5, pad_token_id=0)
draft_cfg.rope_scaling = None          # transformers 5 fills a default; the server's 4.53 does not
train_config = {'max_len': 512, 'gradient_checkpoint': False}
ids = torch.randint(5, 1000, (1, 40))
mask = torch.ones_like(ids)
loss_mask = torch.zeros_like(ids)
loss_mask[:, 10:] = 1

with torch.no_grad():
    ref_states = reference(input_ids=ids, output_hidden_states=True).hidden_states

for variant, layers, fc_norm in (('baseline', [2, 16, 29], False), ('ours', [16, 22, 29], True)):
    os.environ['EAGLE_CAPTURE_LAYERS'] = ','.join(map(str, layers))
    draft_cfg.fc_norm = fc_norm
    torch.manual_seed(0)
    model = Model(draft_cfg, {'zero_optimization': {'stage': 2}}, train_config, path=tmp, load_emb=False)
    model.target_model.float()
    selected = torch.arange(1000)
    model.register_buffer('d2t', selected - torch.arange(1000))
    t2d = torch.zeros(1000, dtype=torch.bool)
    t2d[selected] = True
    model.register_buffer('t2d', t2d)
    model.l1smooth = torch.nn.SmoothL1Loss(reduction='none')
    model = model.cuda()
    # 1. the captured states are HF hidden_states[i] for our ids (input of layer i)
    hs, _, _, _ = model.dataprepare(ids.cuda(), mask.cuda(), loss_mask)
    def rel(index_shift):
        expect = torch.cat([ref_states[i + index_shift] for i in layers], dim=-1).cuda()
        return ((hs - expect).norm() / expect.norm()).item()
    err, off_by_one = rel(0), min(rel(-1), rel(1))
    assert err < 1e-2 and off_by_one > 1e-1, (variant, err, off_by_one)
    # 2. fc_norm present exactly for ours
    assert hasattr(model, 'fc_norm') == fc_norm, variant
    # 3. a few steps on one batch: loss goes down, fc_norm gets gradients
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    losses = []
    for _ in range(60):
        plosses, _, acces = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), loss_mask=loss_mask)
        loss = sum(0.8 ** i * p for i, p in enumerate(plosses))
        opt.zero_grad()
        loss.backward()
        if fc_norm:
            assert all(n.weight.grad is not None and n.weight.grad.abs().sum() > 0 for n in model.fc_norm)
        opt.step()
        losses.append(loss.item())
    assert len(plosses) == 7 and losses[-1] < 0.5 * losses[0], (variant, losses[0], losses[-1])
    keys = [k for k in model.state_dict() if k.startswith('fc_norm')]
    print(f'{variant}: capture rel. error {err:.1e} (neighbouring layers {off_by_one:.1e}), loss {losses[0]:.3f} -> {losses[-1]:.3f}, fc_norm keys {keys}')
print('tiny check passed')

# --- 4. the exported draft (train.py's export) loads into the authors' generation code and runs
import json, shutil                                       # noqa: E401,E402
for f in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
    shutil.copy(os.path.join(TOKENIZER, f), tmp)
sys.path.insert(0, WORK + '/eval')
from eagle.model.ea_model import EaModel                  # noqa: E402
draft_dir = tempfile.mkdtemp()
weights = {k: v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu()
           for k, v in model.state_dict().items()
           if not k.startswith(('target_model.', 'embed_tokens.')) and not k.endswith('inv_freq')}
cfg = draft_cfg.to_dict()
cfg.update({'eagle_capture_layers': layers, 'fc_norm': True, 'rope_scaling': None})
json.dump(cfg, open(os.path.join(draft_dir, 'config.json'), 'w'))
torch.save(weights, os.path.join(draft_dir, 'pytorch_model.bin'))
ea = EaModel.from_pretrained(use_eagle3=True, base_model_path=tmp, ea_model_path=draft_dir, total_token=20,
                             depth=4, top_k=4, torch_dtype=torch.float16, device_map='cuda')
assert os.environ['EAGLE_CAPTURE_LAYERS'] == '16,22,29' and hasattr(ea.ea_layer, 'fc_norm')
out = ea.eagenerate(ids.cuda(), temperature=0.0, max_new_tokens=16, log=True, is_llama3=True)
print('generation ok: new tokens', int(out[1]), 'rounds', int(out[2]) + 1)
broken = {k: v for k, v in weights.items() if not k.startswith('fc_norm')}
torch.save(broken, os.path.join(draft_dir, 'pytorch_model.bin'))
try:
    EaModel.from_pretrained(use_eagle3=True, base_model_path=tmp, ea_model_path=draft_dir, total_token=20,
                            depth=4, top_k=4, torch_dtype=torch.float16, device_map='cuda')
    raise SystemExit('a draft without its fc_norm weights loaded silently')
except ValueError as e:
    print('missing fc_norm weights refused:', str(e)[:90])
print('eval check passed')
