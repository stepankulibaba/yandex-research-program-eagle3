"""SGLang 0.5.9's EAGLE-3 draft (LlamaForCausalLMEagle3) plus EAGLE-3.1 fc_norm, for our variant.

Use: PYTHONPATH=<big_training> SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_eagle31 python -m sglang.launch_server ...
SGLang imports this package in every server process and lets its class replace the built-in one.

Without this, SGLang 0.5.9 would load our draft and silently drop the fc_norm weights (its loader skips unknown
names): wrong results with no error. Here, when the draft config says "fc_norm": true, one RMSNorm per captured
target state is applied in front of fc (a forward pre-hook, so the parameter names and the loader stay SGLang's),
and loading fails unless all three fc_norm weights arrive. Drafts without fc_norm behave exactly as before.
"""
import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import llama_eagle3


def _normalise_parts(norms):
    def hook(module, args):
        parts = args[0].chunk(len(norms), dim=-1)
        return (torch.cat([norm(part.contiguous()) for norm, part in zip(norms, parts)], dim=-1),) + args[1:]
    return hook


class LlamaForCausalLMEagle3(llama_eagle3.LlamaForCausalLMEagle3):
    def __init__(self, config, quant_config=None, prefix=''):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.uses_fc_norm = bool(getattr(config, 'fc_norm', False))
        if self.uses_fc_norm:
            model = self.model
            model.fc_norm = nn.ModuleList([RMSNorm(model.hidden_size_in, eps=config.rms_norm_eps) for _ in range(3)])
            model.fc.register_forward_pre_hook(_normalise_parts(model.fc_norm))

    def load_weights(self, weights):
        seen = []

        def tracked():
            for name, tensor in weights:
                if 'fc_norm' in name:
                    seen.append(name)
                yield name, tensor

        super().load_weights(tracked())
        if self.uses_fc_norm and len(set(seen)) != 3:
            raise ValueError(f'fc_norm draft: expected 3 fc_norm weights, got {seen}')


EntryClass = [LlamaForCausalLMEagle3]
