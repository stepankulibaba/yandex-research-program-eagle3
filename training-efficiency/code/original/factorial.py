"""Finite staged factorial: frozen selection, teacher preparation, baseline, four cells."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,gc,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'code'/'original';EXPERIMENT=ROOT/'factorial_20261008';PILOT=ROOT/'data_budget_dense_20261008';CURVE=ROOT/'learning_curve_20261008'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'remote'))
import torch
import probe_nemo_ttt as base
import run_pilot as pilot
from loading import ReadOnlySafeOpen,read_only_safetensors
from fixed_path import sha,dump as original_dump,now,tensor_digest
from learning_curve import replay
from data_budget_pilot import metrics,evaluate,cache_from_doc,dense_parity
from factorial_utils import token_schedule,crop_cache,scheduler_gate,optimizer_gate
from fp32_master_adamw import MasterAdamW,LOSS_SCALE
from probe_nemo_supervision import distribution,shift
TARGET=ROOT/'hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
ORIGINAL=ROOT/'hf_cache/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots'/base.DRAFT_SHA
BUDGET=79712

def dump(path,obj):
    # Windows readers may briefly deny delete-sharing on status.json.
    for attempt in range(20):
        try:return original_dump(path,obj)
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.025*(attempt+1),.25))

def qualify_crop(trainer,draft,doc,cache,head,ids,membership,prompt_length):
    batch=base.pack([doc],doc['tokens'].shape[1],'cuda')
    with torch.no_grad():raw=torch.nn.functional.linear(batch['hidden'],head)
    checks=[]
    candidates=[max(prompt_length+8,doc['tokens'].shape[1]-11),max(prompt_length+8,doc['tokens'].shape[1]-31)]
    for n in candidates:
        assert n<doc['tokens'].shape[1]
        cropped=crop_cache(cache,n);trunc={k:v[:,:n] for k,v in doc.items()};sub=base.pack([trunc],n,'cuda')
        expected_probs,expected_mask=distribution(shift(raw[:,:n]),ids,membership,shift(sub['loss_mask']))
        check={'length':n,'probabilities_exact':torch.equal(expected_probs.cpu(),cropped['target_probs']),
               'mask_exact':torch.equal(expected_mask.cpu(),cropped['position_mask'])}
        results=[]
        for mode in range(2):
            trainer.zero_grad(set_to_none=True)
            if mode==0:res=base.forward(trainer,sub,{'target_logits':shift(raw[:,:n])})
            else:res=trainer(**{k:v.cuda() for k,v in cropped.items()})
            res.loss.backward();results.append((base.metrics(res),base.grads(draft)));del res
        m0,g0=results[0];m1,g1=results[1]
        check.update(metrics_exact=all(torch.equal(m0[k],m1[k]) for k in m0),
                     gradients_exact=set(g0)==set(g1) and len(g0)==13 and all(torch.equal(g0[k],g1[k]) for k in g0),
                     finite=all(bool(torch.isfinite(x).all()) for x in g0.values()))
        check['passed']=all(check[k] for k in ['probabilities_exact','mask_exact','metrics_exact','gradients_exact','finite']);assert check['passed'],check
        checks.append(check);del sub,results,expected_probs,expected_mask,cropped
    trainer.zero_grad(set_to_none=True);return checks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True)
    for k in ['runner','protocol','helper','inputs']:ap.add_argument('--'+k+'-sha',required=True)
    a=ap.parse_args();phase=a.phase;out=EXPERIMENT/phase;out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    if phase=='select':counts={'selected':[0,128],'gates':[0,1]}
    elif phase.startswith('prep_'):counts={'generated':[0,52],'captured':[0,52],'cached':[0,52],'gates':[0,4]}
    elif phase=='eval96hi':counts={'gates':[0,3],'ttt_evaluation':[0,64],'fixed_replay':[0,80],'repeat_checks':[0,40]}
    elif phase=='baseline':counts={'cpu_cases':[0,122],'gates':[0,2],'ttt_evaluation':[0,64],'fixed_replay':[0,80],'repeat_checks':[0,40]}
    else:
        plan=json.loads((REPO/f'FACTORIAL_SCHEDULE_{phase}.json').read_text(encoding='utf-8'))
        counts={'gates':[0,3 if phase=='train24hi' else 1],'updates':[0,len(plan['steps'])],'ttt_evaluation':[0,64],'fixed_replay':[0,80],'repeat_checks':[0,40],'checkpoints':[0,1]}
    state={'state':'running','stage':'integrity','phase':phase,'started_utc':now(),'counts':counts,'trained_input_tokens':0,'token_budget':BUDGET if phase.startswith('train') else 0}
    def status(stage=None):
        if stage:state['stage']=stage
        state['updated_utc']=now();dump(out/'status.json',state)
    def gate(name,details):
        dump(out/('gate_'+name+'.json'),details);assert details['passed'],name
        counts['gates'][0]+=1;status()
    status()
    try:
        assert sha(__file__)==a.runner_sha
        for name,v in [('FACTORIAL_PROTOCOL.md',a.protocol_sha),('factorial_utils.py',a.helper_sha),(f'FACTORIAL_INPUTS_{phase}.json',a.inputs_sha)]:assert sha(REPO/name)==v
        inherited=json.loads((REPO/f'FACTORIAL_INPUTS_{phase}.json').read_text(encoding='utf-8'))
        assert all(sha(ROOT/p)==v for p,v in inherited.items())
        pilot.verify_vendor();torch.manual_seed(20261008);torch.backends.cuda.matmul.allow_tf32=False
        source=json.loads((PILOT/'source_weights_after.json').read_text(encoding='utf-8'));assert all(sha(ROOT/p)==v for p,v in source.items())
        dump(out/'source_weights_before.json',source)
        previous=json.loads((PILOT/'generated.json').read_text(encoding='utf-8'))
        prior_completion=json.loads((PILOT/'completion.json').read_text(encoding='utf-8'))
        env={'phase':phase,'torch':torch.__version__,'precision':'FP16/eager, FP32 master AdamW','seed':20261008,
             'runner_sha256':a.runner_sha,'protocol_sha256':a.protocol_sha,'helper_sha256':a.helper_sha,'inputs_sha256':a.inputs_sha,
             'inherited_inputs':inherited,'target_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','draft_revision':base.DRAFT_SHA,
             'nemo_revision':'3914f200a4c782d44b58ee7a01b4685e4158e19c','eagle_revision':pilot.COMMIT}
        if phase=='select':
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(str(TARGET),local_files_only=True)
            candidates=json.loads((REPO/'data_budget_candidates_20261008.json').read_text(encoding='utf-8'))['records']
            eligible=[]
            for priority,item in enumerate(candidates):
                ids=tok.apply_chat_template([{'role':'user','content':item['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if 16<=len(ids)<=256:eligible.append({**item,'prompt_ids':ids,'priority':priority})
            lookup={r['id']:r for r in eligible};trainold=[r for r in previous if r['pilot_split']=='train'];oldval=[r for r in previous if r['pilot_split']=='validation']
            train=[]
            for r in trainold:
                assert lookup[r['id']]['prompt_ids']==r['prompt_ids'];train.append({**lookup[r['id']],'pilot_split':'train','inherited':True})
            used={r['id'] for r in train};pool=[r for r in eligible if r['source_split']=='train_sft' and r['id'] not in used]
            for repeat in range(3):
                for ref in trainold:
                    chosen=min(pool,key=lambda r:(abs(len(r['prompt_ids'])-len(ref['prompt_ids'])),r['priority']))
                    pool.remove(chosen);used.add(chosen['id']);train.append({**chosen,'pilot_split':'train','inherited':False,'matched_to':ref['id']})
            excluded={r['id'] for r in previous};excluded_hashes={r['prompt_sha256'] for r in previous+train}
            validation=[]
            for r in eligible:
                if r['source_split']=='test_sft' and r['id'] not in excluded and r['prompt_sha256'] not in excluded_hashes:
                    validation.append({**r,'pilot_split':'validation','inherited':False})
                    if len(validation)==32:break
            assert len(train)==96 and len(validation)==32
            selected=train+validation
            gate('split_integrity',{'passed':len({r['id'] for r in selected})==128 and len({r['prompt_sha256'] for r in selected})==128 and not ({r['id'] for r in validation}&excluded),
                                    'train_original_ids':[r['id'] for r in train[:24]],'excluded_old_validation':[r['id'] for r in oldval],
                                    'eligible_counts':{s:sum(r['source_split']==s for r in eligible) for s in ['train_sft','test_sft']}})
            dump(out/'selection.json',{'train':train,'validation':validation,'new_order':train[24:]+validation,'old_development':oldval,
                 'selection_before_model_outputs':True,'candidate_sha256':sha(REPO/'data_budget_candidates_20261008.json')})
            counts['selected'][0]=128;status('selection_frozen')
        elif phase.startswith('prep_'):
            from transformers import Qwen3ForCausalLM,AutoTokenizer
            selection=json.loads((EXPERIMENT/'select/selection.json').read_text(encoding='utf-8'));selected=selection['new_order'][0:52] if phase=='prep_a' else selection['new_order'][52:104]
            tok=AutoTokenizer.from_pretrained(str(TARGET),local_files_only=True)
            status('target_loading')
            with read_only_safetensors():target=Qwen3ForCausalLM.from_pretrained(str(TARGET),torch_dtype=torch.float16,device_map={'':0},attn_implementation='eager',local_files_only=True).eval().requires_grad_(False)
            env['gpu']=torch.cuda.get_device_name();generated=[];generation_begin=time.perf_counter()
            @torch.no_grad()
            def generate(item):
                ids=torch.tensor([item['prompt_ids']],device='cuda');torch.cuda.synchronize();start=time.perf_counter()
                seq=target.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=128,temperature=None,top_p=None,top_k=None,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize();return seq.cpu().tolist()[0],time.perf_counter()-start
            inherited_tokens,_=generate(previous[0]);gate('inherited_generation',{'passed':inherited_tokens==previous[0]['tokens'],'id':previous[0]['id']})
            status('generation')
            for i,item in enumerate(selected):
                tokens,elapsed=generate(item);assert 0<len(tokens)-len(item['prompt_ids'])<=128
                row={**item,'tokens':tokens,'new_tokens':len(tokens)-len(item['prompt_ids']),'generation_seconds':elapsed}
                if i==0:
                    repeat,_=generate(item);gate('new_generation_repeat',{'passed':repeat==tokens,'id':item['id']})
                generated.append(row);dump(out/'generated.json',generated);counts['generated'][0]+=1;status();print('GENERATED',i+1,item['pilot_split'],flush=True)
            generation_wall=time.perf_counter()-generation_begin;status('capture');captured={};docs=[];capture_rows=[]
            def hook(i):
                def record(module,args,result):captured[i]=(result[0] if isinstance(result,tuple) else result).detach()
                return record
            handles=[target.model.layers[i].register_forward_hook(hook(i)) for i in [1,13,24]]
            with torch.no_grad():
                for item in generated:
                    ids=torch.tensor([item['tokens']],device='cuda');start=time.perf_counter();hidden=target.model(input_ids=ids,use_cache=False).last_hidden_state
                    aux=torch.cat([captured[i] for i in [1,13,24]],-1);mask=torch.zeros_like(ids);mask[:,len(item['prompt_ids']):]=1
                    assert torch.isfinite(hidden).all() and torch.isfinite(aux).all()
                    docs.append({'tokens':ids.cpu(),'hidden':hidden.cpu(),'aux':aux.cpu(),'loss_mask':mask.cpu()})
                    torch.cuda.synchronize();capture_rows.append({'id':item['id'],'seconds':time.perf_counter()-start,'sequence_tokens':ids.shape[1]})
                    captured.clear();counts['captured'][0]+=1;status();del ids,hidden,aux,mask
            for h in handles:h.remove()
            head=target.lm_head.weight.detach();embedding=target.get_input_embeddings().weight.detach();del target;gc.collect();torch.cuda.empty_cache()
            draft,ids,membership=base.load_draft(ROOT,embedding,out);del embedding
            assert base.digest(draft)==prior_completion['initial_draft_sha256']
            trainer=base.R.Eagle3TrainerModule(draft,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=7)
            caches=[];cache_rows=[];status('cache')
            for i,(item,doc) in enumerate(zip(generated,docs)):
                start=time.perf_counter();batch,cache=cache_from_doc(doc,head,ids,membership,True)
                if i==0:
                    check=dense_parity(trainer,batch,head,ids,membership);gate('dense_cache',check)
                    checks=qualify_crop(trainer,draft,doc,cache,head,ids,membership,len(item['prompt_ids']))
                    gate('cropped_cache',{'passed':all(x['passed'] for x in checks),'cases':checks})
                minimum=max(len(item['prompt_ids'])+min(8,item['new_tokens']),int(cache['position_mask'].view(-1).nonzero()[0])+2)
                caches.append({'id':item['id'],'split':item['pilot_split'],'cache':cache,'minimum':minimum})
                torch.cuda.synchronize();cache_rows.append({'id':item['id'],'seconds':time.perf_counter()-start,'cache_bytes':sum(v.numel()*v.element_size() for v in cache.values()),'minimum':minimum})
                counts['cached'][0]+=1;status();del batch
            assert base.digest(draft)==prior_completion['initial_draft_sha256']
            torch.save({'caches':caches,'aux_layers':[1,13,24]},out/'teacher_cache.pt');dump(out/'capture_rows.json',capture_rows);dump(out/'cache_rows.json',cache_rows)
            dump(out/'preparation.json',{'generation_wall_seconds_including_repeat_gates':generation_wall,'generation_seconds_sum':sum(x['generation_seconds'] for x in generated),
                 'capture_seconds_sum':sum(x['seconds'] for x in capture_rows),'cache_seconds_sum':sum(x['seconds'] for x in cache_rows),'teacher_cache_sha256':sha(out/'teacher_cache.pt'),'generated_sha256':sha(out/'generated.json')})
        else:
            from eagle.model.configs import EConfig
            from eagle.model.cnets import Model
            from eagle.model.ea_model import evaluate_posterior
            from fixed_path_matcher import cpu_gate
            if phase=='baseline':
                cpu={'optimizer':optimizer_gate(),'oracle':cpu_gate(evaluate_posterior),'scheduler':scheduler_gate()};dump(out/'cpu_gate.json',cpu);counts['cpu_cases'][0]=122;status()
            selection=json.loads((EXPERIMENT/'select/selection.json').read_text(encoding='utf-8'))
            inherited_caches=torch.load(PILOT/'teacher_cache.pt',map_location='cpu',weights_only=True)['caches']
            new_caches=[];new_generated=[]
            for p in ['prep_a','prep_b']:
                assert json.loads((EXPERIMENT/p/'status.json').read_text(encoding='utf-8'))['state']=='complete'
                new_caches+=torch.load(EXPERIMENT/p/'teacher_cache.pt',map_location='cpu',weights_only=True)['caches']
                new_generated+=json.loads((EXPERIMENT/p/'generated.json').read_text(encoding='utf-8'))
            cachemap={x['id']:x for x in inherited_caches+new_caches};genmap={x['id']:x for x in previous+new_generated}
            assert len(cachemap)==len(genmap)==136
            panel=[cachemap[r['id']] for r in selection['train'][:24]]
            fresh=[cachemap[r['id']] for r in selection['validation']];development=[cachemap[r['id']] for r in selection['old_development']]
            index=json.loads((TARGET/'model.safetensors.index.json').read_text())
            with ReadOnlySafeOpen(TARGET/index['weight_map']['model.embed_tokens.weight']) as f:embedding=f.get_tensor('model.embed_tokens.weight').to(device='cuda',dtype=torch.float16)
            draft,ids,membership=base.load_draft(ROOT,embedding,out);initial=base.digest(draft);embedding_sha=tensor_digest([embedding]);del embedding
            gate('initial',{'passed':initial==prior_completion['initial_draft_sha256'],'full_draft_sha256':initial})
            trainer=base.R.Eagle3TrainerModule(draft,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=7)
            env.update(gpu=torch.cuda.get_device_name(),target_loaded=False,initial_draft_sha256=initial)
            if phase.startswith('train'):
                assert sum(x['length'] for x in plan['steps'])==BUDGET and plan['lr'] in [1e-4,1e-5]
                env.update(lr=plan['lr'],train_size=plan['train_size'],schedule_sha256=sha(REPO/f'FACTORIAL_SCHEDULE_{phase}.json'))
                opt=MasterAdamW(draft);opt.optimizer.param_groups[0]['lr']=plan['lr'];updates=[];status('training')
                for i,step in enumerate(plan['steps'],1):
                    entry=cachemap[step['id']];cache=crop_cache(entry['cache'],step['length']);opt.zero_grad();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    kw={k:v.cuda() for k,v in cache.items()};res=trainer(**kw);assert torch.isfinite(res.loss) and res.loss>0 and res.valid_tokens>0
                    (res.loss*LOSS_SCALE).backward();norm=opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    for (_,p),master in zip(opt.named,opt.masters):
                        st=opt.optimizer.state[master];assert all(bool(torch.isfinite(v).all()) for v in (p,master,st['exp_avg'],st['exp_avg_sq']))
                    updates.append({'update':i,'id':step['id'],'input_tokens':step['length'],'cropped':step['cropped'],'update_seconds':elapsed,
                        'supervised_positions_step0':int(kw['position_mask'].sum()),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                        'grad_norm_before_clip':float(norm),'finite':True,**metrics(res)})
                    counts['updates'][0]=i;state['trained_input_tokens']+=step['length'];dump(out/'updates.json',updates);status();opt.zero_grad();del kw,res,norm,cache
                    if phase=='train24hi' and i==24:
                        gate('inherited_epoch1',{'passed':base.digest(draft)==prior_completion['final_draft_sha256'],'full_draft_sha256':base.digest(draft)})
                    if i%24==0:print('UPDATE',i,flush=True)
                assert state['trained_input_tokens']==BUDGET
                params={n:p.detach().cpu().clone() for n,p in draft.named_parameters() if p.requires_grad};full=base.digest(draft)
                if phase=='train24hi':
                    previous_final=json.loads((CURVE/'checkpoints.json').read_text(encoding='utf-8'))[-1]
                    old=torch.load(CURVE/'draft_trainable_384.pt',map_location='cpu',weights_only=True)['trainable_parameters']
                    gate('inherited_final',{'passed':full==previous_final['full_draft_sha256'] and len(updates)==384 and set(params)==set(old) and all(torch.equal(params[n],old[n]) for n in params),
                                          'full_draft_sha256':full,'parameters':len(params)})
                torch.save({'trainable_parameters':params,'full_draft_sha256':full,'lr':plan['lr'],'train_size':plan['train_size'],'input_token_budget':BUDGET,'updates':len(updates)},out/'draft_final.pt')
                counts['checkpoints'][0]=1;del opt,params;gc.collect();torch.cuda.empty_cache()
            if phase=='eval96hi':
                recovered=torch.load(EXPERIMENT/'train96hi/draft_final.pt',map_location='cpu',weights_only=True)
                assert recovered['updates']==381 and recovered['input_token_budget']==BUDGET and recovered['lr']==1e-4 and recovered['train_size']==96
                parameters=recovered['trainable_parameters'];expected={n for n,p in draft.named_parameters() if p.requires_grad}
                assert set(parameters)==expected and len(expected)==13
                fullstate=draft.state_dict();fullstate.update(parameters);draft.load_state_dict(fullstate,strict=True)
                gate('recovered_checkpoint',{'passed':base.digest(draft)==recovered['full_draft_sha256'],'full_draft_sha256':base.digest(draft),'parameters':13,
                     'checkpoint_sha256':sha(EXPERIMENT/'train96hi/draft_final.pt')})
                env.update(lr=1e-4,train_size=96,recovered_evaluation=True)
                del parameters,fullstate
            # Legacy rotary buffers must be constructed normally, not meta/to_empty.
            layer=Model(EConfig.from_pretrained(str(ORIGINAL)),load_emb=False,bias=json.loads((ORIGINAL/'config.json').read_text()).get('bias',True),total_tokens=64,depth=2,top_k=10,threshold=1.0).to(device='cuda',dtype=torch.float16).eval().requires_grad_(False)
            weights=torch.load(ORIGINAL/'pytorch_model.bin',map_location='cpu',weights_only=True);weights['embed_tokens.weight']=draft.model.embed_tokens.weight.detach().cpu()
            for name,p in draft.named_parameters():
                if not p.requires_grad:continue
                name=name.replace('model.layers.0.','midlayer.').replace('model.fc.','fc.').replace('model.norm.','norm.')
                assert name in weights;weights[name]=p.detach().cpu().clone()
            layer.load_state_dict(weights,strict=True);del weights
            summaries=[];status('ttt_evaluation')
            for split,items in [('train_panel',panel),('validation',fresh),('development',development)]:
                sub=out/split;sub.mkdir();summary=evaluate(trainer,draft,items,sub,0);obj=json.loads((sub/'validation_000.json').read_text(encoding='utf-8'))
                dump(out/f'ttt_{split}.json',obj);summaries.append({'split':split,**summary});counts['ttt_evaluation'][0]+=len(items);status()
            dump(out/'ttt_summary.json',summaries);results=[];status('fixed_replay')
            for split,items in [('validation',fresh),('development',development)]:
                for entry in items:
                    item=genmap[entry['id']];first=None
                    for repeat in range(2):
                        row=replay(layer,item,entry['cache']['aux_hidden_states'],evaluate_posterior);row.update(id=item['id'],split=split,condition=phase,repeat=repeat)
                        if repeat==0:first=row
                        else:
                            row['repeat_exact']=all(row[k]==first[k] for k in ['useful_tokens','cycles','tau_fixed','draft_accepted','events']);assert row['repeat_exact'];counts['repeat_checks'][0]+=1
                        results.append(row);counts['fixed_replay'][0]+=1;dump(out/'fixed_replay.json',results);status()
            if phase=='baseline':
                old=json.loads((ROOT/'fixed_path_20261008/fixed_replay.json').read_text(encoding='utf-8'));checks=[]
                for r in old:
                    if r['teacher']!='full_teacher_cache' or r['variant']!='original' or r['repeat']!=0:continue
                    new=next(x for x in results if x['id']==r['id'] and x['repeat']==0)
                    checks.append({'id':r['id'],'passed':all(r[k]==new[k] for k in ['useful_tokens','cycles','tau_fixed','draft_accepted','events'])})
                gate('inherited_replay',{'passed':len(checks)==8 and all(x['passed'] for x in checks),'cases':checks})
            if phase=='eval96hi':
                oldrows=json.loads((EXPERIMENT/'train96hi/fixed_replay.json').read_text(encoding='utf-8'));lookup={(r['id'],r['repeat']):r for r in results}
                exact=all(all(r[k]==lookup[r['id'],r['repeat']][k] for k in ['cycles','events','useful_tokens','tau_fixed','draft_accepted']) for r in oldrows)
                metric_exact=True
                for split in ['train_panel','validation','development']:
                    oldttt=json.loads((EXPERIMENT/'train96hi'/('ttt_'+split+'.json')).read_text(encoding='utf-8'))
                    newttt=json.loads((out/('ttt_'+split+'.json')).read_text(encoding='utf-8'));metric_exact=metric_exact and oldttt['rows']==newttt['rows']
                gate('partial_replay_exact',{'passed':exact and metric_exact and len(oldrows)==52,'previous_replay_cases':len(oldrows),'ttt_rows_exact':metric_exact})
            assert tensor_digest([draft.model.embed_tokens.weight])==tensor_digest([layer.embed_tokens.weight])==embedding_sha
        status('final_integrity');dump(out/'environment.json',env)
        after={p:sha(ROOT/p) for p in source};assert after==source
        assert all(sha(ROOT/p)==v for p,v in inherited.items());assert all(x==n for x,n in counts.values())
        dump(out/'source_weights_after.json',after)
        comp={'source_weights_unchanged':True,'inherited_inputs_unchanged':True,'total_wall_seconds':time.perf_counter()-begin,'finished_utc':now()}
        if phase.startswith('train'):comp.update(checkpoint_sha256=sha(out/'draft_final.pt'),full_draft_sha256=base.digest(draft),frozen_embedding_unchanged=True,
           update_seconds_sum=sum(x['update_seconds'] for x in updates),input_tokens=sum(x['input_tokens'] for x in updates),sum_sequence_length_squared=sum(x['input_tokens']**2 for x in updates),
           supervised_positions_step0=sum(x['supervised_positions_step0'] for x in updates),schedule_sha256=sha(REPO/f'FACTORIAL_SCHEDULE_{phase}.json'))
        dump(out/'completion.json',comp);state['state']='complete';status('done')
    except Exception:
        state.update(state='failed',error=traceback.format_exc());status();print(state['error'],flush=True);raise
if __name__=='__main__':main()
