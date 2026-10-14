"""Explicit CUDA/Triton factory for unchanged pinned eager AST model definitions."""
import ast,copy,dataclasses,hashlib,json,logging,math,sys,types
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig,PreTrainedModel
from transformers.activations import ACT2FN
from nemo_ttt_runtime import definitions,REF

def load_fused_runtime(nemo_root):
    assert torch.cuda.is_available(),'Real CUDA required'
    root=Path(nemo_root).resolve();rel='nemo_automodel/components/loss/triton/soft_cross_entropy.py'
    expected=json.loads((REF/'source_hashes.json').read_text())[rel]
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected
    sys.path.insert(0,str(root))
    from nemo_automodel.components.loss.triton import soft_cross_entropy as fused
    assert fused.HAVE_TRITON and Path(fused.__file__).resolve()==(root/rel).resolve()
    module=types.ModuleType('_eagle_pinned_nemo_fused');sys.modules[module.__name__]=module;ns=module.__dict__
    ns.update(torch=torch,nn=nn,math=math,dataclass=dataclasses.dataclass,
        logger=logging.getLogger(module.__name__),PretrainedConfig=PretrainedConfig,PreTrainedModel=PreTrainedModel,
        _HAS_FA=False,_SUPPORTED_ATTN_IMPLEMENTATIONS=('eager','flash_attention_2'),
        _LK_LOSS_TYPES=('alpha','lambda'),HAVE_TRITON=True,ACT2FN=ACT2FN,F=F,
        fused_soft_cross_entropy=fused.fused_soft_cross_entropy)
    for name in ('_PeagleAttentionMixin','_PeagleDecoderLayerMixin','_PeagleDraftMixin'):ns[name]=type(name,(),{})
    definitions('nemo_automodel/components/models/common/utils.py',['initialize_rms_norm_module'],ns)
    definitions('nemo_automodel/components/models/llama/rope_utils.py',
        ['rotate_half','apply_rotary_pos_emb','_get_rope_config','_compute_default_inv_freq','_compute_llama3_inv_freq','LlamaRotaryEmbedding'],ns)
    definitions('nemo_automodel/components/datasets/llm/packed_sequence.py',['build_block_causal_additive_mask'],ns)
    definitions('nemo_automodel/components/speculative/eagle/draft_llama.py',
        ['_build_causal_mask','Eagle3LlamaAttention','Eagle3LlamaMLP','Eagle3LlamaDecoderLayer','Eagle3LlamaModel','LlamaEagle3DraftModel'],ns)
    definitions('nemo_automodel/components/loss/soft_ce.py',['masked_soft_cross_entropy'],ns)
    definitions('nemo_automodel/components/speculative/eagle/core.py',
        ['_shift_left_with_zero','_compute_target_distribution','_masked_mean','Eagle3StepMetrics','Eagle3TrainerModule'],ns)
    return module,fused

def guard(self,target_logits,target_probs,position_mask):
    assert self.cp_group is None and self.lk_loss_type is None
    assert target_logits is None and target_probs is not None and position_mask is not None
    assert not target_probs.requires_grad and target_probs.dtype==torch.float32
    B,S,V=target_probs.shape;capacity=S+self.ttt_steps-1
    assert target_probs.storage_offset()==0 and target_probs.stride()==(capacity*V,V,1)
    assert target_probs.untyped_storage().nbytes()==B*capacity*V*4

def offset_trainer(runtime):
    rel='nemo_automodel/components/speculative/eagle/core.py';source=(REF/rel).read_bytes()
    assert hashlib.sha256(source).hexdigest()==json.loads((REF/'source_hashes.json').read_text())[rel]
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Eagle3TrainerModule')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'));fn.name='_offset_forward';count=[0]
    class Patch(ast.NodeTransformer):
        def visit_Assign(self,n):
            if ast.unparse(n)=='cur_target_probs = shift(cur_target_probs)':
                count[0]+=1
                return ast.parse('cur_target_probs = target_probs.as_strided(target_probs.shape, target_probs.stride(), target_probs.storage_offset() + (step_idx + 1) * target_probs.shape[-1])').body[0]
            return self.generic_visit(n)
    fn=Patch().visit(fn);assert count==[1];fn.body.insert(1,ast.parse('_offset_guard(self,target_logits,target_probs,position_mask)').body[0])
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    runtime.__dict__['_offset_guard']=guard
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[])),__file__,'exec'),runtime.__dict__)
    class OffsetTrainer(runtime.Eagle3TrainerModule):forward=runtime.__dict__['_offset_forward']
    return OffsetTrainer
