"""Real pinned Triton and pretrained recurrent B2: gates before full-step timing."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import argparse,functools,gc,hashlib,json,time,traceback
from pathlib import Path
import torch
from transformers import PretrainedConfig
from nemo_fused_runtime import load_fused_runtime,offset_trainer
from nemo_batched_soft_ce import batch_sliced_soft_ce
from fp32_master_adamw import MasterAdamW,LOSS_SCALE,OPTIMIZER_KWARGS,cpu_gate

ROOT=Path(__file__).resolve().parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*2**20):h.update(b)
    return h.hexdigest()

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def digest(model):
    h=hashlib.sha256()
    for n,x in model.state_dict().items():h.update(n.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def restore(model,saved):
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad:p.copy_(saved[n])
    gc.collect();torch.cuda.empty_cache()

def configuration(cfg):
    fields=dict(hidden_size=16,target_hidden_size=16,intermediate_size=32,vocab_size=19,
        draft_vocab_size=5,num_attention_heads=2,num_key_value_heads=1,head_dim=8,num_hidden_layers=1,
        num_aux_hidden_states=3,hidden_act='silu',rms_norm_eps=1e-6,rope_theta=10000.,rope_scaling=None,
        max_position_embeddings=40960,attn_implementation='eager',tie_word_embeddings=False,pad_token_id=0,
        torch_dtype=torch.float16,fc_norm=False,norm_output=False,parallel_drafting=False)
    fields.update(cfg);fields.update(torch_dtype=torch.float16,tie_word_embeddings=False,target_hidden_size=2048,num_aux_hidden_states=3)
    c=PretrainedConfig(**fields);c._attn_implementation='eager';return c

def load_model(runtime,inputmeta,out):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    draftid='AngelSlim/Qwen3-1.7B_eagle3';targetid='Qwen/Qwen3-1.7B';cache=str(ROOT/'hf_cache')
    cp=Path(hf_hub_download(draftid,'pytorch_model.bin',revision=inputmeta['draft_revision'],cache_dir=cache))
    configpath=Path(hf_hub_download(draftid,'config.json',revision=inputmeta['draft_revision'],cache_dir=cache))
    assert sha(cp)==inputmeta['draft_checkpoint_sha256'] and sha(configpath)==inputmeta['draft_config_sha256']
    idx=Path(hf_hub_download(targetid,'model.safetensors.index.json',revision=inputmeta['target_revision'],cache_dir=cache))
    shard=json.loads(idx.read_text())['weight_map']['model.embed_tokens.weight']
    weight=hf_hub_download(targetid,shard,revision=inputmeta['target_revision'],cache_dir=cache)
    with safe_open(weight,framework='pt',device='cpu') as f:embedding=f.get_tensor('model.embed_tokens.weight').to(torch.float16)
    assert embedding.shape==(151936,2048)
    source=torch.load(cp,weights_only=True,map_location='cpu');cfg=configuration(json.loads(configpath.read_text()))
    with torch.device('meta'):model=runtime.LlamaEagle3DraftModel(cfg)
    model=model.to(dtype=torch.float16).to_empty(device='cuda');mapping={}
    for n,value in source.items():
        key=n.replace('midlayer.','model.layers.0.').replace('fc.','model.fc.') if n.startswith(('midlayer.','fc.')) else n
        if n.startswith('norm.'):key='model.'+n
        mapping[key]=value
    mapping['model.embed_tokens.weight']=embedding;model.load_state_dict(mapping,strict=True);model.freeze_embeddings();model.train()
    ids=torch.arange(model.d2t.numel(),device='cuda')+model.d2t
    membership=torch.zeros(151936,dtype=torch.bool,device='cuda');membership[ids]=True
    assert torch.equal(model.t2d,membership) and ids.unique().numel()==32000
    assert all(p.dtype==torch.float16 for p in model.parameters())
    assert not model.model.embed_tokens.weight.requires_grad
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==136847360
    assert len([p for p in model.parameters() if p.requires_grad])==13
    dump(out/'loading.json',{'strict':True,'embedding_frozen':True,'draft_revision':inputmeta['draft_revision'],
        'target_revision':inputmeta['target_revision'],'embedding_shard':shard,'num_trainable':136847360,
        'config_overrides':{'torch_dtype':'float16','tie_word_embeddings':False,'attn_implementation':'eager'},'keys':list(mapping)})
    del source,mapping,embedding;gc.collect();torch.cuda.empty_cache();return model,ids,membership

def transfer(cache,mode,steps):
    kw={k:v.to('cuda') for k,v in cache.items() if k!='target_probs'};p=cache['target_probs']
    if mode=='shift':kw['target_probs']=p.to('cuda')
    else:
        B,S,V=p.shape;buffer=torch.empty(B,S+steps-1,V,device='cuda',dtype=torch.float32)
        buffer[:,:S].copy_(p);buffer[:,S:].zero_();kw['target_probs']=buffer[:,:S]
    return kw

def mask_case(cache,pattern):
    kw=dict(cache)
    if pattern!='natural':
        kw['position_mask']=cache['position_mask'].clone();kw['loss_mask']=cache['loss_mask'].clone()
        if pattern=='empty':kw['position_mask'].zero_();kw['loss_mask'].zero_()
        else:
            keep=kw['position_mask'][0,:,0].nonzero().flatten()[:8]
            kw['position_mask'][0].zero_();kw['loss_mask'][0].zero_()
            kw['position_mask'][0,keep,0]=True;kw['loss_mask'][0,keep]=1
    return kw

def metric_values(res):return {k:getattr(res,k).detach().cpu().clone() for k in ('loss','accuracy','valid_tokens','step_prefix_hits','step_valid')}
def parameter_grads(model):return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def compare_grads(g0,g1,cfg):
    assert set(g0)==set(g1) and len(g0)==13;details=[]
    for n in sorted(g0):
        x,y=g0[n],g1[n];finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        details.append({'name':n,'finite':finite,'close':bool(torch.allclose(x,y,atol=cfg['grad_atol'],rtol=cfg['rtol'])),
            'max_abs':float((x-y).abs().max()) if finite else None,'shift_sha256':thash(x),'views_sha256':thash(y)})
    return details

def fingerprint(opt,with_grads=True):
    result={}
    for (n,p),m in zip(opt.named,opt.masters):
        st=opt.optimizer.state[m];tensors={'weight_fp16':p,'master_fp32':m,'exp_avg':st['exp_avg'],'exp_avg_sq':st['exp_avg_sq']}
        if with_grads:tensors['grad_fp32']=m.grad
        result[n]={}
        for k,x in tensors.items():
            cpu=x.detach().cpu();assert bool(torch.isfinite(cpu).all()),(n,k)
            result[n][k]=thash(cpu)
        result[n]['step']=int(st['step'].item())
    return result

def state_record(opt,res,norm,with_grads=True):
    return {'fingerprint':fingerprint(opt,with_grads),'metrics':{k:thash(v) for k,v in metric_values(res).items() if k!='loss'},
        'loss':float(res.loss),'loss_sha256':thash(res.loss),'grad_norm':float(norm)}

def records_match(a,b,cfg,ignore_grad=False):
    fpa=a['fingerprint'];fpb=b['fingerprint']
    if ignore_grad:fpb={n:{k:v for k,v in x.items() if k!='grad_fp32'} for n,x in fpb.items()}
    lossclose=abs(a['loss']-b['loss'])<=cfg['loss_atol']+cfg['rtol']*abs(b['loss'])
    return fpa==fpb and a['metrics']==b['metrics'] and a['grad_norm']==b['grad_norm'] and lossclose

def full_step(tr,opt,cache,mode):
    kw=transfer(cache,mode,7);transfer_peak=torch.cuda.max_memory_allocated()/2**20
    res=tr(**kw);(res.loss*LOSS_SCALE).backward();backward_peak=torch.cuda.max_memory_allocated()/2**20
    norm=opt.step();update_peak=torch.cuda.max_memory_allocated()/2**20
    return kw,res,norm,{'after_transfer_mib':transfer_peak,'after_backward_mib':backward_peak,'after_update_mib':update_peak}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nemo-root',required=True);ap.add_argument('--inputs',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);assert not out.exists();out.mkdir()
    cfg=json.loads((ROOT/'code/original/fused_b2_protocol_20261007.json').read_text());dump(out/'protocol_before_probe.json',cfg)
    status={'state':'running','stage':'cpu_optimizer','checks_done':0,'checks_total':29,'runs_done':0,'runs_total':8,
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':cfg['scope']}
    dump(out/'status.json',status);model=saved=initial=None
    try:
        torch.set_num_threads(4);dump(out/'cpu_checks.json',cpu_gate());status['checks_done']=13
        torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261007)
        inputs=Path(args.inputs);meta=json.loads((inputs/'input_manifest.json').read_text())
        assert sha(inputs/'b2_teacher_inputs.pt')==meta['data_sha256']
        assert meta['parent_capture_sha_verified'] and all(c['passed'] for c in meta['teacher_checks'])
        data=torch.load(inputs/'b2_teacher_inputs.pt',weights_only=True,map_location='cpu')
        runtime,fused=load_fused_runtime(args.nemo_root);Offset=offset_trainer(runtime)
        original=runtime.masked_soft_cross_entropy
        candidate=functools.partial(batch_sliced_soft_ce,fused_apply=fused.SoftCrossEntropyFunction.apply)
        def make_trainer(mode,steps):
            runtime.masked_soft_cross_entropy=original if mode=='shift' else candidate
            cls=runtime.Eagle3TrainerModule if mode=='shift' else Offset
            return cls(model,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=steps)
        status['stage']='pretrained_loading';dump(out/'status.json',status)
        model,ids,membership=load_model(runtime,meta,out);initial=digest(model)
        assert initial==meta['expected_initial_draft_sha256'],'Full model differs from previous FP16 checkpoint mapping'
        saved={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        free,total=torch.cuda.mem_get_info();budget=free/2**20+torch.cuda.memory_allocated()/2**20
        admitted=cfg['admission_estimate_mib']+cfg['margin_mib']<=budget
        dump(out/'admission.json',{'admitted':admitted,'budget_mib':budget,'estimate_mib':cfg['admission_estimate_mib'],'margin_mib':cfg['margin_mib'],'scope':'engineering estimate, not a strict bound'})
        assert admitted,'Insufficient memory; no dependent measurements'
        import transformers,triton
        env={'torch':torch.__version__,'transformers':transformers.__version__,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(0),
            'initial_draft_sha256':initial,'data_sha256':meta['data_sha256'],'input_manifest_sha256':sha(inputs/'input_manifest.json'),
            'nemo_commit':meta['nemo_commit'],'target_revision':meta['target_revision'],'draft_revision':meta['draft_revision'],
            'actual_triton':True,'precision':'FP16 computation / FP32 master AdamW','loss_scale':LOSS_SCALE,'optimizer_kwargs':OPTIMIZER_KWARGS,
            'target_resident':False,'runtime_policy':'unchanged pinned eager AST subset; actual original Triton loss; no CUDA loss fallback',
            'source_sha256':{n:sha(ROOT/'code/original'/n) for n in ['probe_nemo_fused_b2_training.py','nemo_fused_runtime.py','nemo_ttt_runtime.py','nemo_batched_soft_ce.py','fp32_master_adamw.py','fused_b2_protocol_20261007.json']},
            'pinned_fused_source_sha256':sha(Path(args.nemo_root)/'nemo_automodel/components/loss/triton/soft_cross_entropy.py')}
        dump(out/'environment.json',env);dump(out/'input_manifest.json',meta);status.update(stage='recurrent_backward_gates',checks_done=14);dump(out/'status.json',status)
        checks=[]
        cases=[('main',d,m) for d in cfg['depths'] for m in cfg['masks']]+[('packed',7,'natural')]
        for name,depth,pattern in cases:
            cache=mask_case(data[name],pattern);values=[];audits=[]
            for mode in ('shift','views'):
                restore(model,saved);tr=make_trainer(mode,depth);kw=transfer(cache,mode,depth);teachers=[]
                def hook(t):
                    if t.ndim==2 and t.shape[-1]==32000 and t.dtype==torch.float32 and not t.requires_grad:
                        teachers.append({'ptr':t.untyped_storage().data_ptr(),'bytes':t.untyped_storage().nbytes(),'shape':list(t.shape)})
                    return t
                with torch.autograd.graph.saved_tensors_hooks(hook,lambda t:t):res=tr(**kw)
                res.loss.backward();values.append((metric_values(res),parameter_grads(model)));audits.append(teachers)
                assert digest(model)==initial
                del tr,kw,res;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            m0,g0=values[0];m1,g1=values[1];details=compare_grads(g0,g1,cfg)
            lossclose=bool(torch.allclose(m0['loss'],m1['loss'],atol=cfg['loss_atol'],rtol=cfg['rtol']))
            exact=all(torch.equal(m0[k],m1[k]) for k in m0 if k!='loss')
            expected_calls=depth*(1 if depth==1 else 2)
            storageok=len(audits[1])==expected_calls and len({x['ptr'] for x in audits[1]})==1
            assert len(audits[0])==depth,'Baseline did not use real fused CE for every step'
            row={'case':name,'ttt_steps':depth,'mask':pattern,'loss_shift':float(m0['loss']),'loss_views':float(m1['loss']),
                'loss_close':lossclose,'metrics_exact':exact,'gradients':details,'teacher_storage_audits':{'shift':audits[0],'views':audits[1]},
                'shared_teacher_storage':storageok,'expected_fused_calls':expected_calls,
                'passed':lossclose and exact and storageok and all(x['finite'] and x['close'] for x in details)}
            checks.append(row);dump(out/'recurrent_checks.json',checks);assert row['passed'],row
            status['checks_done']=14+len(checks);dump(out/'status.json',status);print('RECURRENT',len(checks),name,depth,pattern,'PASS',flush=True)
            del values,audits,m0,m1,g0,g1,details;gc.collect()
        status['stage']='late_cache_gradient';dump(out/'status.json',status);late=[]
        for mode in ('shift','views'):
            restore(model,saved);tr=make_trainer(mode,7);lossfn=runtime.masked_soft_cross_entropy;losses=[];cache0=[]
            def loss_hook(*a,**kw):
                loss=lossfn(*a,**kw);losses.append(loss);return loss
            def attn_hook(module,a,kw,result):
                if not cache0:k=kw['cache_hidden'][0][0];k.retain_grad();cache0.append(k)
            handle=model.model.layers[0].self_attn.register_forward_hook(attn_hook,with_kwargs=True)
            runtime.masked_soft_cross_entropy=loss_hook
            try:
                kw=transfer(data['main'],mode,7);res=tr(**kw);losses[-1].backward()
                g=cache0[0].grad;assert g is not None and bool(torch.isfinite(g).all()) and float(g.float().norm())>0
                late.append(g.detach().cpu().clone())
            finally:
                handle.remove();runtime.masked_soft_cross_entropy=lossfn;tr=kw=res=losses=cache0=g=None
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        latecheck={'finite_nonzero_both':True,'gradient_close':bool(torch.allclose(late[0],late[1],atol=cfg['grad_atol'],rtol=cfg['rtol'])),
            'max_abs':float((late[0]-late[1]).abs().max()),'shift_sha256':thash(late[0]),'views_sha256':thash(late[1])}
        latecheck['passed']=latecheck['gradient_close'];dump(out/'late_cache_check.json',latecheck);assert latecheck['passed'];del late
        status.update(checks_done=25,stage='full_step_trajectory');dump(out/'status.json',status);references=[];trajectory=[]
        for mode in ('shift','views'):
            restore(model,saved);tr=make_trainer(mode,7);opt=MasterAdamW(model)
            try:
                for i in range(3):
                    kw,res,norm,_=full_step(tr,opt,data['main'],mode);rec=state_record(opt,res,norm)
                    if mode=='shift':references.append(rec)
                    else:
                        row={'step':i+1,'passed':records_match(rec,references[i],cfg),'fingerprints_exact':rec['fingerprint']==references[i]['fingerprint'],
                            'loss_shift':references[i]['loss'],'loss_views':rec['loss'],'loss_hash_exact':rec['loss_sha256']==references[i]['loss_sha256']}
                        trajectory.append(row);dump(out/'trajectory_checks.json',trajectory);assert row['passed'],row
                        status['checks_done']=25+len(trajectory);dump(out/'status.json',status);print('UPDATE',i+1,'PASS',flush=True)
                    opt.zero_grad();kw=res=norm=rec=None;gc.collect()
            finally:
                opt.zero_grad();tr=opt=kw=res=norm=None;restore(model,saved)
            assert digest(model)==initial
        dump(out/'trajectory_fingerprints.json',references)
        status['stage']='full_step_measurements';dump(out/'status.json',status);rows=[]
        for repeat in range(4):
            for mode in (('shift','views') if repeat%2==0 else ('views','shift')):
                restore(model,saved);tr=make_trainer(mode,7);opt=MasterAdamW(model)
                try:
                    kw,res,norm,_=full_step(tr,opt,data['main'],mode);opt.zero_grad();kw=res=norm=None;gc.collect();torch.cuda.synchronize()
                    persistent=opt.persistent_bytes();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    kw,res,norm,peaks=full_step(tr,opt,data['main'],mode);opt.zero_grad();torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                    peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
                    rec=state_record(opt,res,norm,with_grads=False);assert records_match(rec,references[1],cfg,ignore_grad=True)
                    row={'repeat':repeat,'mode':mode,'batch':2,'length':166,'ttt_steps':7,'ms':ms,
                        'peak_allocated_mib':peak,'peak_reserved_mib':reserved,'phase_peaks':peaks,'loss':rec['loss'],
                        'update_matches_gate':True,'persistent_optimizer_bytes':persistent,'target_resident':False}
                    rows.append(row);dump(out/'rows.json',rows);status['runs_done']=len(rows);dump(out/'status.json',status);print('RUN',len(rows),mode,round(ms,2),round(peak,2),flush=True)
                finally:
                    opt.zero_grad();tr=opt=kw=res=norm=rec=None;restore(model,saved)
        final=digest(model);assert final==initial;dump(out/'weights_integrity.json',{'initial':initial,'final':final,'restored':True,'checkpoint_written':False})
        status.update(state='complete',stage='done',checks_done=29,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));dump(out/'status.json',status)
    except Exception:
        if model is not None and saved is not None:
            restore(model,saved);dump(out/'weights_on_failure.json',{'restored':digest(model)==initial,'checkpoint_written':False})
        status.update(state='failed',error=traceback.format_exc());dump(out/'status.json',status);raise

if __name__=='__main__':main()
