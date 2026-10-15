"""SGLang 0.5.9's EAGLE-3 draft (LlamaForCausalLMEagle3) plus EAGLE-3.1 fc_norm, for our variant.

Use: PYTHONPATH=<big_training> SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_eagle31 python -m sglang.launch_server ...
SGLang imports this package in every server process and lets its class replace the built-in one.

Without this, SGLang 0.5.9 would load our draft and silently drop the fc_norm weights (its loader skips unknown
names): wrong results with no error. Here, when the draft config says "fc_norm": true, one RMSNorm per captured
target state is applied in front of fc (a forward pre-hook, so the parameter names and the loader stay SGLang's),
and loading fails unless all three fc_norm weights arrive. Drafts without fc_norm behave exactly as before.

Also fixes SGLang 0.5.9 for drafts whose target layers are not the default 2, N/2, N-3 (ours: 16, 22, 29). The CUDA
graph runner calls set_eagle3_layers_to_capture() without ids before capturing, so the graphs (decode and target
verify) record the default layers; the draft's ids are applied only afterwards and reach just the eager prefill.
Our draft then got the baseline's layers in every verification round (MT-bench tau 4.36 instead of 5.04). Here a call
without ids uses the draft config's ids, which the model runner has read before the graphs are captured.
"""
import logging

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor import model_runner
from sglang.srt.models import llama, llama_eagle3

logger = logging.getLogger(__name__)
_set_layers = llama.LlamaForCausalLM.set_eagle3_layers_to_capture
_init_device_graphs = model_runner.ModelRunner.init_device_graphs


def set_eagle3_layers_to_capture(self, layer_ids=None):
    if layer_ids is None:
        layer_ids = getattr(self, 'eagle3_draft_layer_ids', None)
    return _set_layers(self, layer_ids)


def init_device_graphs(self):
    ids = getattr(self, 'eagle_aux_hidden_state_layer_ids', None)
    if ids is not None and getattr(self, 'model', None) is not None:
        self.model.eagle3_draft_layer_ids = list(ids)
        logger.info(f'EAGLE-3: target layers to capture (output ids) {list(ids)}, also in CUDA graphs')
    return _init_device_graphs(self)


llama.LlamaForCausalLM.set_eagle3_layers_to_capture = set_eagle3_layers_to_capture
model_runner.ModelRunner.init_device_graphs = init_device_graphs


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
