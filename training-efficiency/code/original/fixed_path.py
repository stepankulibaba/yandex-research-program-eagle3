"""Finite fixed-continuation tree replay; production correctness failures remain unchanged."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'code'/'original';OUT=ROOT/'fixed_path_20261008'
PREVIOUS=ROOT/'draft_acceptance_20261008';PILOT=ROOT/'data_budget_dense_20261008'
sys.path.insert(0,str(REPO))
import torch
import run_pilot as pilot
from loading import read_only_safetensors
from fixed_path_matcher import match_prefix,posterior_oracle,cpu_gate


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(4*2**20):h.update(chunk)
    return h.hexdigest()


def dump(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def now():return datetime.now(timezone.utc).isoformat()


def tensor_digest(values):
    h=hashlib.sha256()
    for x in values:
        x=x.detach().cpu().contiguous()
        h.update(str(x.dtype).encode());h.update(str(list(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for key in ['runner','protocol','matcher','inputs']:ap.add_argument('--'+key+'-sha',required=True)
    a=ap.parse_args();assert not OUT.exists();OUT.mkdir()
    begin=time.perf_counter()
    state={'state':'running','stage':'integrity','started_utc':now(),'counts':{
        'cpu_cases':[0,64],'teacher_capture':[0,8],'production_replay':[0,16],
        'cache_checks':[0,8],'fixed_replay':[0,64],'repeat_checks':[0,32]}}
    def status(stage=None):
        if stage:state['stage']=stage
        state['updated_utc']=now();dump(OUT/'status.json',state)
    status()
    try:
        assert sha(__file__)==a.runner_sha
        for name,value in [('FIXED_PATH_PROTOCOL.md',a.protocol_sha),('fixed_path_matcher.py',a.matcher_sha),('FIXED_PATH_INPUTS.json',a.inputs_sha)]:
            assert sha(REPO/name)==value
        identities=json.loads((REPO/'FIXED_PATH_INPUTS.json').read_text(encoding='utf-8'))
        assert all(sha(ROOT/name)==value for name,value in identities.items())
        pilot.verify_vendor();torch.manual_seed(20261008)
        from eagle.model.ea_model import EaModel
        import eagle.model.ea_model as ea
        cpu=cpu_gate(ea.evaluate_posterior);dump(OUT/'cpu_gate.json',cpu)
        state['counts']['cpu_cases'][0]=64;status('model_loading')
        prompts=json.loads((PREVIOUS/'prompts.json').read_text(encoding='utf-8'))
        refs=json.loads((PREVIOUS/'ar_reference.json').read_text(encoding='utf-8'))
        checks=json.loads((PREVIOUS/'generation_checks.json').read_text(encoding='utf-8'))
        assert len(prompts)==len(refs)==8 and len(checks)==16
        checkmap={(x['id'],x['variant']):x for x in checks};refmap={x['id']:x for x in refs}
        original_path=ROOT/'hf_cache/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots/94441b48acc5804677ae12259617c83323b543a9'
        target_path=ROOT/'hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
        source_hashes=json.loads((PREVIOUS/'source_weights_after.json').read_text(encoding='utf-8'))
        assert all(sha(ROOT/p)==v for p,v in source_hashes.items())
        weights={v:torch.load(p/'pytorch_model.bin',map_location='cpu',weights_only=True) for v,p in
                 [('original',original_path),('trained',PREVIOUS/'exported_draft')]}
        with read_only_safetensors():
            model=EaModel.from_pretrained(use_eagle3=True,base_model_path=str(target_path),ea_model_path=str(original_path),
                total_token=64,depth=2,top_k=10,torch_dtype=torch.float16,device_map={'':0}).eval().requires_grad_(False)
        embedding_sha=tensor_digest([model.ea_layer.embed_tokens.weight])
        def swap(variant):
            obj=model.ea_layer.load_state_dict(weights[variant],strict=False)
            assert not obj.unexpected_keys and set(obj.missing_keys)<={'embed_tokens.weight'}
            model.ea_layer.reset_kv();model.ea_layer.init_tree()
        for v in weights:
            swap(v)
            assert all(torch.equal(model.ea_layer.state_dict()[n].cpu(),x) for n,x in weights[v].items())
        swap('original')
        limit=max(len(x['prompt_ids']) for x in prompts)+128+64+96
        env={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
             'python':platform.python_version(),'precision':'FP16','eagle_commit':pilot.COMMIT,
             'requested_tree':[64,2,10],'internal_total_tokens':model.ea_layer.total_tokens,
             'runner_sha256':a.runner_sha,'protocol_sha256':a.protocol_sha,'matcher_sha256':a.matcher_sha,
             'inputs_sha256':a.inputs_sha,'aux_layers':[1,13,24],'scope':'diagnostic fixed trajectory; no performance timing'}
        dump(OUT/'environment.json',env);dump(OUT/'source_weights_before.json',source_hashes)
        status('sequential_teacher_capture')
        teacher={};capture_rows=[]
        for item in prompts:
            chunks={i:[] for i in [1,13,24]}
            def hook(i):
                def record(module,args,result):
                    x=result[0] if isinstance(result,tuple) else result
                    assert x.ndim==3 and x.shape[0]==1
                    chunks[i].append(x.detach().cpu().clone())
                return record
            handles=[model.base_model.model.layers[i].register_forward_hook(hook(i)) for i in chunks]
            ids=torch.tensor([item['prompt_ids']],device='cuda')
            try:output,new,idx=model.naivegenerate(ids,temperature=0,max_new_tokens=127,max_length=limit,log=True)
            finally:
                for h in handles:h.remove()
            tokens=output[0,len(item['prompt_ids']):].cpu().tolist()
            expected=refmap[item['id']]
            assert tokens==expected['tokens'] and int(new)==expected['raw_new_tokens'] and int(idx)+1==expected['cycles']
            aux=torch.cat([torch.cat(chunks[i],1) for i in chunks],-1)
            assert aux.shape[1]==len(item['prompt_ids'])+len(tokens) and aux.dtype==torch.float16 and torch.isfinite(aux).all()
            teacher[item['id']]=aux
            capture_rows.append({'id':item['id'],'tokens_exact':True,'shape':list(aux.shape),'sha256':tensor_digest([aux])})
            dump(OUT/'teacher_capture.json',capture_rows)
            state['counts']['teacher_capture'][0]+=1;status()
            print('TEACHER_CAPTURE',len(capture_rows),flush=True)
        torch.save({'aux_layers':[1,13,24],'features':teacher},OUT/'sequential_teacher.pt')
        status('production_trace')
        productions=[];oracle_cycle_gates=[]
        original_td,original_ep=ea.tree_decoding,ea.evaluate_posterior
        for i,item in enumerate(prompts):
            for variant in (('original','trained') if i%2==0 else ('trained','original')):
                swap(variant);expected=checkmap[item['id'],variant];events=[];current={}
                def tree_call(m,tokens,kv,positions,input_ids,retrieve):
                    current.clear();current['start']=input_ids.shape[1]-len(item['prompt_ids'])
                    return original_td(m,tokens,kv,positions,input_ids,retrieve)
                def posterior(logits,candidates,processor):
                    best,accept,p=original_ep(logits,candidates,processor)
                    paths=candidates.detach().cpu().tolist();n=current['start']
                    continuation=expected['tokens'][n:]
                    matched=match_prefix(paths,continuation)
                    oracle=posterior_oracle(paths,continuation,original_ep)
                    assert matched==oracle,'Actual-tree oracle gate failed'
                    events.append({'start':n,'candidates':paths,'actual_best':int(best),'actual_length':int(accept)+1,
                                   'oracle_best':matched[0],'oracle_length':matched[1],
                                   'length_equal':matched[1]==int(accept)+1})
                    return best,accept,p
                ea.tree_decoding,ea.evaluate_posterior=tree_call,posterior
                try:
                    ids=torch.tensor([item['prompt_ids']],device='cuda')
                    output,new,idx=model.eagenerate(ids,temperature=0,max_new_tokens=127,max_length=limit,log=True)
                finally:ea.tree_decoding,ea.evaluate_posterior=original_td,original_ep
                tokens=output[0,len(item['prompt_ids']):].cpu().tolist()
                assert tokens==expected['tokens'] and int(idx)+1==expected['cycles'] and int(new)==expected['raw_new_tokens']
                assert len(events)==expected['cycles'] and sum(e['actual_length'] for e in events)==len(tokens)
                row={'id':item['id'],'variant':variant,'tokens_exact':True,'cycles_exact':True,
                     'ar_exact':expected['passed'],'cycles':int(idx)+1,'events':events}
                productions.append(row);dump(OUT/'production_trace.json',productions)
                state['counts']['production_replay'][0]+=1;status()
                print('PRODUCTION_TRACE',len(productions),variant,flush=True)
        status('full_cache_checks')
        caches=torch.load(PILOT/'teacher_cache.pt',map_location='cpu',weights_only=True)
        assert caches['aux_layers']==[1,13,24]
        cachemap={r['id']:r['cache'] for r in caches['caches']}
        full={};cache_rows=[]
        generated=json.loads((PILOT/'generated.json').read_text(encoding='utf-8'));genmap={r['id']:r for r in generated}
        for item in prompts:
            key=item['id'];entry=cachemap[key];cached_tokens=genmap[key]['tokens'];p=len(item['prompt_ids'])
            assert cached_tokens[p:]==refmap[key]['tokens']
            assert torch.equal(entry['input_ids'][0,:-1],torch.tensor(cached_tokens[1:])) and int(entry['input_ids'][0,-1])==0
            assert entry['aux_hidden_states'].shape==teacher[key].shape
            full[key]=entry['aux_hidden_states']
            cache_rows.append({'id':key,'tokens_exact':True,'shape':list(full[key].shape),
                'full_vs_sequential_max_abs':float((full[key]-teacher[key]).abs().max()),
                'full_vs_sequential_exact':torch.equal(full[key],teacher[key]),'sha256':tensor_digest([full[key]])})
            state['counts']['cache_checks'][0]+=1
        dump(OUT/'cache_checks.json',cache_rows);del caches,cachemap,entry;gc.collect()
        status('fixed_tree_replay')
        results=[];first={}
        @torch.no_grad()
        def replay(item,features):
            target=refmap[item['id']]['useful_tokens'];n=0;previous_n=0;events=[]
            model.ea_layer.reset_kv();model.ea_layer.init_tree()
            while n<len(target):
                prefix=item['prompt_ids']+target[:n+1]
                ids=torch.tensor([prefix],device='cuda')
                lo=0 if n==0 else len(item['prompt_ids'])+previous_n
                hi=len(item['prompt_ids'])+n
                hidden=features[:,lo:hi].cuda()
                assert hidden.shape[1]==(len(item['prompt_ids']) if n==0 else n-previous_n)
                tree_tokens,retrieve,mask,positions=model.ea_layer.topK_genrate(hidden,ids,model.base_model.lm_head,None)
                assert int(tree_tokens[0,0])==target[n]
                assert model.ea_layer.stable_kv[0][0].shape[2]==hi, 'Draft KV/feature alignment differs'
                extended=torch.cat([tree_tokens.cpu(),torch.tensor([[-1]])],1)
                paths=extended[0,retrieve.cpu()].tolist()
                best,count=match_prefix(paths,target[n:])
                assert (best,count)==posterior_oracle(paths,target[n:],original_ep)
                events.append({'start':n,'accepted_length':count,'best':best,
                     'tree_sha256':tensor_digest([tree_tokens,retrieve,mask,positions])})
                previous_n=n;n+=count
                assert n<=len(target) and len(events)<=len(target)
            assert n==len(target)
            return {'useful_tokens':n,'cycles':len(events),'tau_fixed':n/len(events),
                    'draft_accepted':n-len(events),'events':events}
        for i,item in enumerate(prompts):
            for mode,features in [('sequential_AR',teacher),('full_teacher_cache',full)]:
                for repeat in range(2):
                    for variant in (('original','trained') if (i+repeat)%2==0 else ('trained','original')):
                        swap(variant);row=replay(item,features[item['id']])
                        row.update(id=item['id'],variant=variant,teacher=mode,repeat=repeat)
                        key=(item['id'],mode,variant)
                        if repeat==0:first[key]=row
                        else:
                            previous=first[key]
                            row['repeat_exact']=all(row[k]==previous[k] for k in ['useful_tokens','cycles','tau_fixed','draft_accepted','events'])
                            assert row['repeat_exact'],'Fixed trajectory replay differs across repeats'
                            state['counts']['repeat_checks'][0]+=1
                        results.append(row);state['counts']['fixed_replay'][0]+=1
                        dump(OUT/'fixed_replay.json',results);status()
                    print('FIXED_PAIR',i+1,mode,repeat+1,flush=True)
        status('final_integrity');swap('original')
        assert tensor_digest([model.ea_layer.embed_tokens.weight])==embedding_sha
        after={p:sha(ROOT/p) for p in source_hashes};assert after==source_hashes
        assert all(sha(ROOT/name)==value for name,value in identities.items())
        dump(OUT/'source_weights_after.json',after)
        dump(OUT/'completion.json',{'source_weights_unchanged':True,'embedding_unchanged':True,'inherited_inputs_unchanged':True,
            'sequential_teacher_sha256':sha(OUT/'sequential_teacher.pt'),
            'total_wall_seconds':time.perf_counter()-begin,'finished_utc':now()})
        state['state']='complete';status('done')
    except Exception:
        state['state']='failed';state['error']=traceback.format_exc();status();print(state['error'],flush=True);raise


if __name__=='__main__':main()
