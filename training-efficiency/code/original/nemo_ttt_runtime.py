"""Reviewed single-GPU eager subset of pinned NeMo, without package installation.

Executed AST definitions are unchanged. Disabled P-EAGLE mixins are empty
capability stubs; the harness rejects those modes. Source hashes are verified.
"""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN

REF = Path(__file__).resolve().parents[2]/'Automodel'


def definitions(path, names, namespace):
    source=(REF/path).read_bytes()
    manifest=json.loads((REF/'source_hashes.json').read_text())
    assert hashlib.sha256(source).hexdigest()==manifest[path], path
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names), (path,names)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
    exec(compile(module,str(REF/path),'exec'),namespace)


def load_runtime():
    module=types.ModuleType('_eagle_pinned_nemo_ttt');sys.modules[module.__name__]=module
    ns=module.__dict__
    ns.update(torch=torch,nn=nn,math=math,dataclass=dataclasses.dataclass,
              logger=logging.getLogger(module.__name__),PretrainedConfig=PretrainedConfig,PreTrainedModel=PreTrainedModel,
              _HAS_FA=False,_SUPPORTED_ATTN_IMPLEMENTATIONS=('eager','flash_attention_2'),
              _LK_LOSS_TYPES=('alpha','lambda'),HAVE_TRITON=False,ACT2FN=ACT2FN,F=F)
    if torch.cuda.is_available() and importlib.util.find_spec('triton') is not None:
        raise RuntimeError('Triton is present: load the actual NeMo fused CE before GPU timing, do not silently choose fallback')
    for name in ('_PeagleAttentionMixin','_PeagleDecoderLayerMixin','_PeagleDraftMixin'):
        ns[name]=type(name,(),{})
    definitions('nemo_automodel/components/models/common/utils.py',['initialize_rms_norm_module'],ns)
    definitions('nemo_automodel/components/models/llama/rope_utils.py',
                ['rotate_half','apply_rotary_pos_emb','_get_rope_config','_compute_default_inv_freq','_compute_llama3_inv_freq','LlamaRotaryEmbedding'],ns)
    definitions('nemo_automodel/components/datasets/llm/packed_sequence.py',['build_block_causal_additive_mask'],ns)
    definitions('nemo_automodel/components/speculative/eagle/draft_llama.py',
                ['_build_causal_mask','Eagle3LlamaAttention','Eagle3LlamaMLP','Eagle3LlamaDecoderLayer','Eagle3LlamaModel','LlamaEagle3DraftModel'],ns)
    definitions('nemo_automodel/components/loss/soft_ce.py',['masked_soft_cross_entropy'],ns)
    definitions('nemo_automodel/components/speculative/eagle/core.py',
                ['_shift_left_with_zero','_compute_target_distribution','_masked_mean','Eagle3StepMetrics','Eagle3TrainerModule'],ns)
    return module
