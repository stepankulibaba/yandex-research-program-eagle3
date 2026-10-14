"""Pinned NeMo eager recurrent forward/backward gate and paired measurement.

Frozen target backbone is captured before measurement; optimizer is excluded.
No target or draft weights are changed, including in memory.
"""
import argparse
import gc
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig
from nemo_ttt_runtime import load_runtime
from probe_nemo_supervision import chunked, shift, distribution, dump, NEMO_SHA, QWEN_SHA

R = load_runtime()
DRAFT_SHA = '94441b48acc5804677ae12259617c83323b543a9'


def config(**overrides):
    fields=dict(hidden_size=16,target_hidden_size=16,intermediate_size=32,vocab_size=19,
        draft_vocab_size=5,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        num_hidden_layers=1,num_aux_hidden_states=3,hidden_act='silu',rms_norm_eps=1e-6,
        rope_theta=10000.,rope_scaling=None,max_position_embeddings=40960,
        attn_implementation='eager',tie_word_embeddings=False,pad_token_id=0,
        torch_dtype=torch.float32,fc_norm=False,norm_output=False,parallel_drafting=False)
    fields.update(overrides); c=PretrainedConfig(**fields);c._attn_implementation='eager';return c


def digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pack(documents, length, device):
    parts=[];remain=length
    for doc in documents:
        n=min(remain,doc['tokens'].shape[1]);parts.append({k:v[:,:n] for k,v in doc.items()})
        remain-=n
        if not remain:break
    assert remain==0
    lens=[p['tokens'].shape[1] for p in parts]
    tensors={k:torch.cat([p[k] for p in parts],1).to(device) for k in parts[0]}
    tensors['position_ids']=torch.cat([torch.arange(n) for n in lens]).view(1,-1).to(device)
    tensors['doc_remaining']=torch.cat([torch.arange(n-1,-1,-1) for n in lens]).view(1,-1).to(device)
    tensors['seq_lens']=torch.tensor([lens],device=device)
    tensors['attention_mask']=torch.ones_like(tensors['tokens'])
    return tensors


def prepare(mode,batch,weight,ids,membership,chunk):
    if mode=='dense':
        with torch.no_grad():raw=F.linear(batch['hidden'],weight);logits=shift(raw);del raw
        return {'target_logits':logits}
    probs,mask=chunked(batch['hidden'],weight,ids,membership,batch['loss_mask'],chunk)
    return {'target_probs':probs,'position_mask':mask}


def forward(trainer,batch,supervision):
    return trainer(input_ids=shift(batch['tokens']),attention_mask=batch['attention_mask'],
        loss_mask=shift(batch['loss_mask']),aux_hidden_states=batch['aux'],
        position_ids=batch['position_ids'],seq_lens=batch['seq_lens'],
        doc_remaining=batch['doc_remaining'],**supervision)


def metrics(result):
    return {name:getattr(result,name).detach().cpu() for name in
        ('loss','accuracy','valid_tokens','step_prefix_hits','step_valid')}


def grads(model):
    return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}


def compare(trainer,batch,weight,ids,membership,chunk,atol,rtol):
    values=[]
    for mode in ('dense','chunk256'):
        trainer.zero_grad(set_to_none=True)
        sup=prepare(mode,batch,weight,ids,membership,chunk)
        result=forward(trainer,batch,sup);result.loss.backward()
        values.append((metrics(result),grads(trainer.draft_model)))
        del sup,result;gc.collect()
    m0,g0=values[0];m1,g1=values[1]
    expected={n for n,p in trainer.draft_model.named_parameters() if p.requires_grad}
    names_exact=set(g0)==set(g1)==expected
    details=[]
    for name in sorted(g0):
        x,y=g0[name],g1[name];finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        details.append({'name':name,'finite':finite,'close':bool(torch.allclose(x,y,atol=atol,rtol=rtol)),
            'max_abs':float((x-y).abs().max()) if finite else None,
            'reference_norm':float(x.float().norm()) if finite else None})
    finite_loss=bool(torch.isfinite(m0['loss']) and torch.isfinite(m1['loss']))
    # Aggregate metrics are exact; only the floating loss is tolerance-based.
    exact=all(torch.equal(m0[k],m1[k]) for k in m0 if k!='loss')
    probs0,mask0=distribution(prepare('dense',batch,weight,ids,membership,chunk)['target_logits'],ids,membership,shift(batch['loss_mask']))
    probs1,mask1=chunked(batch['hidden'],weight,ids,membership,batch['loss_mask'],chunk)
    row={'loss_dense':float(m0['loss']) if finite_loss else None,'loss_chunk':float(m1['loss']) if finite_loss else None,
        'loss_close':finite_loss and bool(torch.allclose(m0['loss'],m1['loss'],atol=atol,rtol=rtol)),
        'metrics_exact':exact,'parameter_names_exact':names_exact,'all_gradients_finite':all(d['finite'] for d in details),
        'all_gradients_close':all(d['close'] for d in details),'gradients':details,
        'mask_exact':torch.equal(mask0,mask1),'probs_close':torch.allclose(probs0,probs1,atol=atol,rtol=rtol),
        'probs_max_abs':float((probs0-probs1).abs().max()),
        'step_valid':m0['step_valid'].tolist(),'step_prefix_hits':m0['step_prefix_hits'].tolist(),
        'valid_tokens':float(m0['valid_tokens'])}
    row['passed']=all(row[k] for k in ('loss_close','metrics_exact','parameter_names_exact','all_gradients_finite','all_gradients_close','mask_exact','probs_close'))
    trainer.zero_grad(set_to_none=True)
    return row


def late_cache_gradient(trainer,batch,weight,ids,membership,chunk):
    """Last loss alone must backpropagate into the first attention K branch."""
    losses=[];cache0=[];original=R.masked_soft_cross_entropy
    def record_loss(*args,**kw):
        value=original(*args,**kw);losses.append(value);return value
    def hook(module,args,kw,output):
        k=kw['cache_hidden'][0][0]
        if not cache0:k.retain_grad();cache0.append(k)
    handle=trainer.draft_model.model.layers[0].self_attn.register_forward_hook(hook,with_kwargs=True)
    R.masked_soft_cross_entropy=record_loss
    trainer.zero_grad(set_to_none=True)
    try:
        sup=prepare('chunk256',batch,weight,ids,membership,chunk)
        result=forward(trainer,batch,sup);losses[-1].backward()
        g=cache0[0].grad
        finite=g is not None and bool(torch.isfinite(g).all())
        norm=float(g.float().norm()) if finite else None
        return {'ttt_steps':trainer.ttt_steps,'last_loss_only':True,'k0_gradient_finite':finite,
            'k0_gradient_norm':norm,'passed':finite and norm>0}
    finally:
        handle.remove();R.masked_soft_cross_entropy=original;trainer.zero_grad(set_to_none=True)


def cpu_gate():
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    torch.manual_seed(20261006);draft=R.LlamaEagle3DraftModel(config()).float();draft.freeze_embeddings()
    ids=torch.tensor([18,2,0,13,7]);membership=torch.zeros(19,dtype=torch.bool);membership[ids]=True
    weight=torch.randn(19,16)*.01;weight[ids[0]].fill_(1.0);docs=[]
    for n in (9,5):docs.append({'tokens':torch.randint(0,19,(1,n)), 'hidden':torch.randn(1,n,16).abs()+1,
        'aux':torch.randn(1,n,48),'loss_mask':torch.ones(1,n,dtype=torch.long)})
    batch=pack(docs,14,'cpu');checks=[]
    for steps in (1,3,7):
        trainer=R.Eagle3TrainerModule(draft,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=steps)
        for pattern in ('all','sparse','empty'):
            batch['loss_mask'].fill_(1)
            if pattern=='sparse':batch['loss_mask'][:,::2]=0
            if pattern=='empty':batch['loss_mask'].zero_()
            row=compare(trainer,batch,weight,ids,membership,4,2e-6,2e-5)
            row.update(case=pattern,ttt_steps=steps);checks.append(row)
        if steps>1:
            batch['loss_mask'].fill_(1)
            checks.append(dict(case='last_loss_cache_gradient',**late_cache_gradient(trainer,batch,weight,ids,membership,4)))
    torch.set_num_threads(previous_threads)
    return checks


def capture(root,out):
    import run_pilot as pilot
    from loading import read_only_safetensors
    from eagle.model.modeling_qwen3_kv import Qwen3ForCausalLM
    pilot.verify_vendor()
    base=root/'hf_cache'/'hub'/'models--Qwen--Qwen3-1.7B'/'snapshots'/QWEN_SHA
    data=root/'night_20261006'/'ablation_dev'
    prompts=torch.load(data/'encoded_prompts.pt',map_location='cpu',weights_only=True)
    generated=torch.load(data/'generated_tokens.pt',map_location='cpu',weights_only=True)
    previous=torch.load(root/'nemo_supervision_20261006'/'real_activations'/'inputs.pt',weights_only=True,map_location='cpu')
    with read_only_safetensors():
        model=Qwen3ForCausalLM.from_pretrained(str(base),torch_dtype=torch.float16,device_map={'':0},attn_implementation='eager').eval()
    layers=[1,len(model.model.layers)//2-1,len(model.model.layers)-4];docs=[];capture_checks=[]
    saved={}
    def hook(i):
        def save(module,args,output):saved[i]=output[0].detach()
        return save
    handles=[model.model.layers[i].register_forward_hook(hook(i)) for i in layers]
    with torch.no_grad():
        for name,prompt in prompts.items():
            seq=torch.cat((prompt,torch.tensor(generated[(0,name,'upstream')]).view(1,-1)),1).cuda()
            hidden=model.model(input_ids=seq,use_cache=False).last_hidden_state
            lm=torch.zeros_like(seq);lm[:,prompt.shape[1]:]=1
            docs.append({'tokens':seq.cpu(),'hidden':hidden.cpu(),'aux':torch.cat([saved[i] for i in layers],-1).cpu(),'loss_mask':lm.cpu()})
            capture_checks.append({'name':name,'length':seq.shape[1],'previous_hidden_exact':torch.equal(hidden.cpu(),previous['hidden'][name])})
            print('CAPTURE',name,seq.shape[1],flush=True);saved.clear()
    head=model.lm_head.weight.detach();embedding=model.model.embed_tokens.weight.detach()
    for h in handles:h.remove()
    del model,hidden,seq,lm,previous;gc.collect();torch.cuda.empty_cache()
    torch.save({'documents':docs,'capture_checks':capture_checks,'aux_layers':layers},out/'inputs.pt')
    assert all(c['previous_hidden_exact'] for c in capture_checks),'Capture differs from validated stage-1 input'
    return docs,head,embedding,base,capture_checks


def load_draft(root,embedding,out):
    path=root/'hf_cache'/'hub'/'models--AngelSlim--Qwen3-1.7B_eagle3'/'snapshots'/DRAFT_SHA
    source=torch.load(path/'pytorch_model.bin',weights_only=True,map_location='cpu')
    cfg=json.loads((path/'config.json').read_text());cfg.update(torch_dtype=torch.float16,tie_word_embeddings=False,
        target_hidden_size=embedding.shape[1],num_aux_hidden_states=3)
    assert source['fc.weight'].shape==(cfg['hidden_size'],embedding.shape[1]*3)
    with torch.device('meta'):draft=R.LlamaEagle3DraftModel(config(**cfg))
    draft=draft.to(dtype=torch.float16).to_empty(device='cuda')
    mapping={}
    for name,value in source.items():
        target=name.replace('midlayer.','model.layers.0.').replace('fc.','model.fc.') if name.startswith(('midlayer.','fc.')) else name
        if name.startswith('norm.'):target='model.'+name
        mapping[target]=value
    mapping['model.embed_tokens.weight']=embedding
    draft.load_state_dict(mapping,strict=True);draft.freeze_embeddings();draft.train()
    ids=(torch.arange(draft.d2t.numel(),device='cuda')+draft.d2t)
    membership=torch.zeros(draft.config.vocab_size,dtype=torch.bool,device='cuda');membership[ids]=True
    assert torch.equal(draft.t2d,membership) and ids.unique().numel()==ids.numel()
    assert all(p.dtype==torch.float16 for p in draft.parameters())
    dump(out/'loading.json',{'checkpoint':DRAFT_SHA,'strict':True,'keys':list(mapping),
        'embedding_frozen':not draft.model.embed_tokens.weight.requires_grad,'num_trainable':sum(p.numel() for p in draft.parameters() if p.requires_grad),
        'config_overrides':{'torch_dtype':'float16','tie_word_embeddings':False,'attn_implementation':'eager'},
        'mapping':'checkpoint keys renamed to pinned NeMo names; base embedding copied; no random weights retained'})
    del source,mapping;gc.collect();torch.cuda.empty_cache()
    return draft,ids,membership


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work-root');parser.add_argument('--out');parser.add_argument('--cpu-only',action='store_true');a=parser.parse_args()
    root=Path(a.work_root) if a.work_root else None;out=Path(a.out) if a.out else root/'nemo_ttt_20261006';out.mkdir(parents=True,exist_ok=True)
    state={'state':'starting','stage':'cpu_correctness','checks_done':0,'checks_total':19,'runs_done':0,'runs_total':48,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(out/'status.json',state)
    try:
        checks=cpu_gate();dump(out/'cpu_checks.json',checks);state['checks_done']=sum(c['passed'] for c in checks)
        assert all(c['passed'] for c in checks),'CPU gate failed; see cpu_checks.json'
        if a.cpu_only:
            state.update(state='complete',stage='cpu_done',checks_total=len(checks),runs_total=0);dump(out/'status.json',state);return
        assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
        torch.manual_seed(20261006);state['stage']='capture';dump(out/'status.json',state)
        docs,head,embedding,base,capture_checks=capture(root,out)
        draft,ids,membership=load_draft(root,embedding,out);del embedding
        initial_digest=digest(draft)
        dump(out/'environment.json',{'torch':torch.__version__,'python':platform.python_version(),'os':platform.platform(),
            'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'nemo_commit':NEMO_SHA,'target_revision':QWEN_SHA,'draft_revision':DRAFT_SHA,
            'attention':'actual pinned NeMo eager','soft_ce':'actual NeMo PyTorch fallback; Triton unavailable',
            'precision':'fp16','chunk':256,'repeats':4,'lengths':[166,512,1067],'ttt_steps':[3,7],
            'scope':'target output preparation + actual recurrent draft forward/backward; frozen target backbone and optimizer excluded',
            'atol':2e-4,'rtol':2e-3,'initial_weights_sha256':initial_digest,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime_sha256':hashlib.sha256(Path(__file__).with_name('nemo_ttt_runtime.py').read_bytes()).hexdigest(),
            'inputs_sha256':hashlib.sha256((out/'inputs.pt').read_bytes()).hexdigest(),'capture_checks':capture_checks})
        cases=[(n,t) for n in (166,512,1067) for t in (3,7)];eligible=[];gpu_checks=[];ooms=[]
        state['stage']='gpu_correctness';dump(out/'status.json',state)
        for length,steps in cases:
            batch=pack(docs,length,'cuda');trainer=R.Eagle3TrainerModule(draft,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=steps)
            try:
                row=compare(trainer,batch,head,ids,membership,256,2e-4,2e-3);row.update(length=length,ttt_steps=steps)
                gpu_checks.append(row);dump(out/'gpu_checks.json',gpu_checks)
                if not row['passed']:raise AssertionError(f'TTT correctness failed for {length}/{steps}')
                eligible.append((length,steps));state['checks_done']+=1
                print('GPU_CHECK',length,steps,'PASS',row['loss_dense'],flush=True)
                if length==166:
                    late=late_cache_gradient(trainer,batch,head,ids,membership,256);late.update(length=length,case='last_loss_cache_gradient');gpu_checks.append(late)
                    dump(out/'gpu_checks.json',gpu_checks)
                    if not late['passed']:raise AssertionError('Last-loss cache gradient gate failed')
                    state['checks_done']+=1
            except torch.cuda.OutOfMemoryError:
                ooms.append({'length':length,'ttt_steps':steps,'stage':'correctness','traceback':traceback.format_exc()});dump(out/'oom.json',ooms)
                print('OOM',length,steps,'timing ineligible',flush=True)
            finally:
                trainer.zero_grad(set_to_none=True);del trainer,batch;gc.collect();torch.cuda.empty_cache();dump(out/'status.json',state)
        state.update(stage='timing',runs_eligible=len(eligible)*8,oom_cases=len(ooms));dump(out/'status.json',state);rows=[]
        for length,steps in eligible:
            batch=pack(docs,length,'cuda');trainer=R.Eagle3TrainerModule(draft,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=steps)
            for repeat in range(4):
                order=('dense','chunk256') if repeat%2==0 else ('chunk256','dense')
                for mode in order:
                    trainer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                    # One warmup before each measured run, outside timing.
                    sup=prepare(mode,batch,head,ids,membership,256);result=forward(trainer,batch,sup);result.loss.backward()
                    del sup,result;trainer.zero_grad(set_to_none=True);torch.cuda.synchronize()
                    allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    sup=prepare(mode,batch,head,ids,membership,256);result=forward(trainer,batch,sup);result.loss.backward();torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-start)*1000
                    rows.append({'length':length,'ttt_steps':steps,'repeat':repeat,'mode':mode,'ms':elapsed,
                        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                        'incremental_peak_mib':(torch.cuda.max_memory_allocated()-allocated)/2**20,'loss':float(result.loss.detach())})
                    dump(out/'rows.json',rows);state['runs_done']=len(rows);dump(out/'status.json',state)
                    print('RUN',len(rows),length,steps,mode,round(elapsed,2),flush=True)
                    del sup,result
            trainer.zero_grad(set_to_none=True);del trainer,batch;gc.collect();torch.cuda.empty_cache()
        final_digest=digest(draft);assert final_digest==initial_digest,'In-memory weights changed'
        dump(out/'weights_integrity.json',{'initial':initial_digest,'final':final_digest,'unchanged':True})
        state.update(state='complete',stage='done',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));dump(out/'status.json',state)
    except Exception as e:
        state.update(state='failed',error=str(e));dump(out/'status.json',state);(out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
