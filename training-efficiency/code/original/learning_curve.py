"""Finite 0/1/4/16 epoch learnability control, cached teacher and strict inherited gates."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import argparse,gc,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'code'/'original';OUT=ROOT/'learning_curve_20261008'
PILOT=ROOT/'data_budget_dense_20261008';FIXED=ROOT/'fixed_path_20261008'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'remote'))
import torch
import probe_nemo_ttt as base
import run_pilot as pilot
from loading import ReadOnlySafeOpen
from fp32_master_adamw import MasterAdamW,LOSS_SCALE,OPTIMIZER_KWARGS,cpu_gate as optimizer_gate
from fixed_path_matcher import cpu_gate,match_prefix,posterior_oracle
from fixed_path import sha,dump,now,tensor_digest
from data_budget_pilot import metrics,evaluate

@torch.no_grad()
def replay(layer,item,features,oracle):
    target=item['tokens'][len(item['prompt_ids']):];n=0;previous_n=0;events=[]
    assert 0<len(target)<=128
    layer.reset_kv();layer.init_tree()
    while n<len(target):
        ids=torch.tensor([item['prompt_ids']+target[:n+1]],device='cuda')
        lo=0 if n==0 else len(item['prompt_ids'])+previous_n;hi=len(item['prompt_ids'])+n
        hidden=features[:,lo:hi].cuda()
        tree,retrieve,mask,positions=layer.topK_genrate(hidden,ids,None,None)
        assert int(tree[0,0])==target[n] and layer.stable_kv[0][0].shape[2]==hi
        paths=torch.cat([tree.cpu(),torch.tensor([[-1]])],1)[0,retrieve.cpu()].tolist()
        best,count=match_prefix(paths,target[n:]);assert (best,count)==posterior_oracle(paths,target[n:],oracle)
        events.append({'start':n,'accepted_length':count,'best':best,'tree_sha256':tensor_digest([tree,retrieve,mask,positions])})
        previous_n=n;n+=count;assert n<=len(target)
    return {'useful_tokens':n,'cycles':len(events),'tau_fixed':n/len(events),'draft_accepted':n-len(events),'events':events}


def main():
    ap=argparse.ArgumentParser()
    for key in ['runner','protocol','matcher','inputs']:ap.add_argument('--'+key+'-sha',required=True)
    a=ap.parse_args();assert not OUT.exists();OUT.mkdir();begin=time.perf_counter()
    state={'state':'running','stage':'integrity','started_utc':now(),'counts':{
        'cpu_cases':[0,77],'gates':[0,4],'updates':[0,384],'ttt_evaluation':[0,128],
        'fixed_replay':[0,320],'repeat_checks':[0,160],'checkpoints':[0,3]}}
    def status(stage=None):
        if stage:state['stage']=stage
        state['updated_utc']=now();dump(OUT/'status.json',state)
    def gate(name,details):
        dump(OUT/('gate_'+name+'.json'),details);assert details['passed'],name
        state['counts']['gates'][0]+=1;status()
    status()
    try:
        assert sha(__file__)==a.runner_sha
        for name,value in [('LEARNING_CURVE_PROTOCOL.md',a.protocol_sha),('fixed_path_matcher.py',a.matcher_sha),('LEARNING_CURVE_INPUTS.json',a.inputs_sha)]:assert sha(REPO/name)==value
        identities=json.loads((REPO/'LEARNING_CURVE_INPUTS.json').read_text(encoding='utf-8'))
        assert all(sha(ROOT/name)==value for name,value in identities.items())
        pilot.verify_vendor();torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261008)
        from eagle.model.ea_model import evaluate_posterior
        from eagle.model.cnets import Model
        from eagle.model.configs import EConfig
        cp={'optimizer':optimizer_gate(),'oracle':cpu_gate(evaluate_posterior)};dump(OUT/'cpu_gate.json',cp)
        state['counts']['cpu_cases'][0]=77;torch.manual_seed(20261008);status('draft_loading')
        generated=json.loads((PILOT/'generated.json').read_text(encoding='utf-8'))
        caches=torch.load(PILOT/'teacher_cache.pt',map_location='cpu',weights_only=True)['caches']
        assert len(caches)==len(generated)==32 and all(x['id']==y['id'] and x['split']==y['pilot_split'] for x,y in zip(caches,generated))
        train=[x for x in caches if x['split']=='train'];val=[x for x in caches if x['split']=='validation']
        assert len(train)==24 and len(val)==8 and len({x['id'] for x in generated})==32
        for item,entry in zip(generated,caches):
            assert torch.equal(entry['cache']['input_ids'][0,:-1],torch.tensor(item['tokens'][1:]))
            assert int(entry['cache']['input_ids'][0,-1])==0
        seq=torch.load(FIXED/'sequential_teacher.pt',map_location='cpu',weights_only=True)['features']
        previous=json.loads((FIXED/'fixed_replay.json').read_text(encoding='utf-8'))
        prior=json.loads((PILOT/'completion.json').read_text(encoding='utf-8'))
        source=json.loads((PILOT/'source_weights_after.json').read_text(encoding='utf-8'))
        assert all(sha(ROOT/p)==v for p,v in source.items());dump(OUT/'source_weights_before.json',source)
        target=ROOT/'hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
        original=ROOT/'hf_cache/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots'/base.DRAFT_SHA
        index=json.loads((target/'model.safetensors.index.json').read_text())
        with ReadOnlySafeOpen(target/index['weight_map']['model.embed_tokens.weight']) as f:
            embedding=f.get_tensor('model.embed_tokens.weight').to(device='cuda',dtype=torch.float16)
        draft,ids,membership=base.load_draft(ROOT,embedding,OUT);initial=base.digest(draft)
        embedding_sha=tensor_digest([embedding])
        gate('initial',{'passed':initial==prior['initial_draft_sha256'] and torch.equal(draft.model.embed_tokens.weight,embedding),
                        'full_draft_sha256':initial,'embedding_sha256':embedding_sha})
        trainer=base.R.Eagle3TrainerModule(draft,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=7)
        config=EConfig.from_pretrained(str(original));conf=json.loads((original/'config.json').read_text())
        layer=Model(config,load_emb=False,bias=conf.get('bias',True),total_tokens=64,depth=2,top_k=10,threshold=1.0).to(device='cuda',dtype=torch.float16).eval().requires_grad_(False)
        weights=torch.load(original/'pytorch_model.bin',map_location='cpu',weights_only=True)
        weights['embed_tokens.weight']=embedding.detach().cpu();del embedding;gc.collect();torch.cuda.empty_cache()
        def swap():
            mapping={}
            for name,p in draft.named_parameters():
                if not p.requires_grad:continue
                if name.startswith('model.layers.0.'):name='midlayer.'+name[len('model.layers.0.'):]
                elif name.startswith('model.fc.'):name='fc.'+name[len('model.fc.'):]
                elif name.startswith('model.norm.'):name='norm.'+name[len('model.norm.'):]
                assert name in weights;mapping[name]=p.detach().cpu().clone()
            assert len(mapping)==13
            layer.load_state_dict({**weights,**mapping},strict=True);layer.reset_kv();layer.init_tree()
        results=[];curves=[];updates=[];checkpoints=[]
        env={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'precision':'FP16/eager, FP32 master AdamW',
             'optimizer':OPTIMIZER_KWARGS,'loss_scale':LOSS_SCALE,'seed':20261008,'tree':[64,2,10],'ttt_steps':7,
             'target_loaded':False,'initial_draft_sha256':initial,'source_hashes':identities,
             'runner_sha256':a.runner_sha,'protocol_sha256':a.protocol_sha,'inputs_sha256':a.inputs_sha}
        dump(OUT/'environment.json',env)
        def checkpoint_evaluation(update):
            status('evaluation_'+str(update));swap()
            for split,items in [('train',train),('validation',val)]:
                sub=OUT/split;sub.mkdir(exist_ok=True)
                summary=evaluate(trainer,draft,items,sub,update)
                obj=json.loads((sub/f'validation_{update:03d}.json').read_text(encoding='utf-8'))
                dump(OUT/f'ttt_{split}_{update:03d}.json',obj)
                curves.append({'split':split,**summary});state['counts']['ttt_evaluation'][0]+=len(items);status()
                if split=='validation' and update in (0,24):
                    old=json.loads((PILOT/f'validation_{update:03d}.json').read_text(encoding='utf-8'))['rows']
                    assert obj['rows']==old,'Previous TTT validation metrics differ'
            for item,entry in zip(generated,caches):
                modes=[('full_teacher_cache',entry['cache']['aux_hidden_states'])]
                if item['pilot_split']=='validation':modes.append(('sequential_AR',seq[item['id']]))
                for mode,features in modes:
                    first=None
                    for repeat in range(2):
                        row=replay(layer,item,features,evaluate_posterior)
                        row.update(id=item['id'],split=item['pilot_split'],update=update,epoch=update//24,teacher=mode,repeat=repeat)
                        if repeat==0:first=row
                        else:
                            row['repeat_exact']=all(row[k]==first[k] for k in ['useful_tokens','cycles','tau_fixed','draft_accepted','events'])
                            assert row['repeat_exact'];state['counts']['repeat_checks'][0]+=1
                        results.append(row);state['counts']['fixed_replay'][0]+=1
                        dump(OUT/'fixed_replay.json',results);status()
            if update in (0,24):
                variant='original' if update==0 else 'trained';checks=[]
                for old in previous:
                    if old['variant']!=variant or old['teacher']!='full_teacher_cache' or old['repeat']!=0:continue
                    new=next(r for r in results if r['id']==old['id'] and r['update']==update and r['teacher']==old['teacher'] and r['repeat']==0)
                    same=all(old[k]==new[k] for k in ['useful_tokens','cycles','tau_fixed','draft_accepted','events'])
                    checks.append({'id':old['id'],'trees_and_counts_exact':same})
                gate('draft_only_'+str(update),{'passed':len(checks)==8 and all(x['trees_and_counts_exact'] for x in checks),'cases':checks,'ttt_validation_exact':True})
            dump(OUT/'ttt_curve.json',curves);print('EVALUATED',update,flush=True)
        checkpoint_evaluation(0)
        opt=MasterAdamW(draft)
        for update in range(1,385):
            status('training');item=train[(update-1)%24];opt.zero_grad();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begun=time.perf_counter()
            kw={k:v.cuda() for k,v in item['cache'].items()};res=trainer(**kw)
            assert torch.isfinite(res.loss) and res.loss>0 and res.valid_tokens>0
            (res.loss*LOSS_SCALE).backward();norm=opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-begun
            for (_,p),master in zip(opt.named,opt.masters):
                st=opt.optimizer.state[master]
                assert all(bool(torch.isfinite(v).all()) for v in (p,master,st['exp_avg'],st['exp_avg_sq']))
            updates.append({'update':update,'epoch':(update-1)//24+1,'id':item['id'],'update_seconds':elapsed,
                'grad_norm_before_clip':float(norm),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'all_weights_and_moments_finite':True,**metrics(res)})
            state['counts']['updates'][0]=update;dump(OUT/'updates.json',updates);status();opt.zero_grad();del kw,res,norm
            if update in (8,16):
                # Match the inherited first-epoch evaluation lifecycle; auxiliary gates, not curve cases.
                sub=OUT/'inherited_lifecycle';sub.mkdir(exist_ok=True);evaluate(trainer,draft,val,sub,update)
                old=json.loads((PILOT/f'validation_{update:03d}.json').read_text(encoding='utf-8'))['rows']
                actual=json.loads((sub/f'validation_{update:03d}.json').read_text(encoding='utf-8'))['rows']
                assert actual==old,'Inherited lifecycle metrics differ'
            if update in (24,96,384):
                params={n:p.detach().cpu().clone() for n,p in draft.named_parameters() if p.requires_grad};full=base.digest(draft)
                if update==24:
                    old=torch.load(PILOT/'draft_trainable_024.pt',map_location='cpu',weights_only=True)['trainable_parameters']
                    gate('epoch1_exact',{'passed':full==prior['final_draft_sha256'] and set(params)==set(old) and all(torch.equal(params[n],old[n]) for n in params),
                                        'full_draft_sha256':full,'parameters':len(params)})
                file=OUT/f'draft_trainable_{update:03d}.pt'
                torch.save({'trainable_parameters':params,'updates':update,'initial_draft_sha256':initial,'draft_revision':base.DRAFT_SHA},file)
                checkpoints.append({'updates':update,'sha256':sha(file),'full_draft_sha256':full,'path':str(file)});dump(OUT/'checkpoints.json',checkpoints)
                state['counts']['checkpoints'][0]+=1;del params;checkpoint_evaluation(update)
            if update%24==0:print('EPOCH',update//24,flush=True)
        status('final_integrity')
        assert tensor_digest([draft.model.embed_tokens.weight])==tensor_digest([layer.embed_tokens.weight])==embedding_sha
        after={p:sha(ROOT/p) for p in source};assert after==source
        assert all(sha(ROOT/name)==value for name,value in identities.items())
        assert all(d==n for d,n in state['counts'].values())
        dump(OUT/'source_weights_after.json',after)
        dump(OUT/'completion.json',{'source_weights_unchanged':True,'inherited_inputs_unchanged':True,'embedding_unchanged':True,
             'first_epoch_bitwise_exact':True,'total_wall_seconds':time.perf_counter()-begin,'update_seconds_sum':sum(x['update_seconds'] for x in updates),
             'checkpoints':checkpoints,'finished_utc':now()})
        state['state']='complete';status('done')
    except Exception:
        state.update(state='failed',error=traceback.format_exc());status();print(state['error'],flush=True);raise

if __name__=='__main__':main()
