"""Complete finite memory/stride/capacity branch; gates before dependent measurements."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import argparse,functools,gc,json,time,traceback
from pathlib import Path
import torch
from probe_nemo_fused_b2_training import (sha,dump,digest,restore,load_model,transfer,state_record,records_match,
 full_step,mask_case,metric_values,parameter_grads,compare_grads,thash)
from nemo_fused_runtime import load_fused_runtime,offset_trainer
from nemo_batched_soft_ce import batch_sliced_soft_ce
from fp32_master_adamw import MasterAdamW,LOSS_SCALE,cpu_gate
from phase_separated_prediction import predict
from lossless_input_transport import load_lossless_inputs
ROOT=Path(__file__).resolve().parents[2]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--nemo-root',required=True);ap.add_argument('--inputs',required=True);ap.add_argument('--legacy-inputs',required=True);ap.add_argument('--out',required=True);ap.add_argument('--resume');a=ap.parse_args()
 out=Path(a.out);assert not out.exists();out.mkdir();cfg=json.loads((ROOT/'code/original/completion_protocol_20261007.json').read_text());dump(out/'protocol_before_probe.json',cfg)
 state={'state':'running','stage':'cpu_optimizer','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checks_done':0,'checks_total':76,'runs_done':0,'runs_total':96,'scope':cfg['scope']};dump(out/'status.json',state)
 model=saved=initial=None;gates=[];rows=[];controls=[];refs={}
 def status(stage):state.update(stage=stage);dump(out/'status.json',state)
 def check(row):
  gates.append(row);dump(out/'checks.json',gates);assert row['passed'],row;state['checks_done']=13+len(gates);dump(out/'status.json',state);print('CHECK',len(gates),row.get('kind'),row.get('length'),row.get('mode'),'PASS',flush=True)
 def row(x):
  rows.append(x);dump(out/'rows.json',rows);state['runs_done']=len(rows);dump(out/'status.json',state);print('RUN',len(rows),x['kind'],x['length'],x['mode'],x.get('outcome','OK'),flush=True)
 try:
  torch.set_num_threads(4);dump(out/'cpu_checks.json',cpu_gate());state['checks_done']=13;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261007)
  inp=Path(a.inputs);meta=json.loads((inp/'input_manifest.json').read_text());assert meta['documents_sha256']==cfg['documents_sha256']
  assert meta['capture_repeat']['passed'] and all(x['passed'] for x in meta['teacher_checks']) and len(meta['teacher_checks'])==8
  data=load_lossless_inputs(inp)
  legacy=Path(a.legacy_inputs);oldmeta=json.loads((legacy/'input_manifest.json').read_text());assert sha(legacy/'b2_teacher_inputs.pt')==oldmeta['data_sha256'];olddata=torch.load(legacy/'b2_teacher_inputs.pt',weights_only=True,map_location='cpu')
  runtime,fused=load_fused_runtime(a.nemo_root);Offset=offset_trainer(runtime);original=runtime.masked_soft_cross_entropy
  import stride_soft_cross_entropy as patched
  split=functools.partial(batch_sliced_soft_ce,fused_apply=fused.SoftCrossEntropyFunction.apply)
  def trainer(mode,steps=7):
   runtime.masked_soft_cross_entropy={'dense':original,'split':split,'stride':patched.fused_soft_cross_entropy}[mode]
   return (runtime.Eagle3TrainerModule if mode=='dense' else Offset)(model,selected_token_ids=ids,selected_token_mask=membership,ttt_steps=steps)
  def clean():
   torch.cuda.set_per_process_memory_fraction(1.);restore(model,saved);torch.cuda.synchronize()
  def step(tr,opt,cache,mode,separated=False):
   torch.cuda.reset_peak_memory_stats();kw=transfer(cache,'shift' if mode=='dense' else 'views',7)
   if separated:torch.cuda.synchronize()
   pt=torch.cuda.max_memory_allocated()/2**20
   if separated:torch.cuda.reset_peak_memory_stats()
   res=tr(**kw)
   if separated:torch.cuda.synchronize()
   pf=torch.cuda.max_memory_allocated()/2**20
   if separated:torch.cuda.reset_peak_memory_stats()
   (res.loss*LOSS_SCALE).backward()
   if separated:torch.cuda.synchronize()
   pb=torch.cuda.max_memory_allocated()/2**20
   if separated:torch.cuda.reset_peak_memory_stats()
   norm=opt.step()
   if separated:torch.cuda.synchronize()
   pu=torch.cuda.max_memory_allocated()/2**20
   return kw,res,norm,{'transfer':pt,'forward':pf,'backward':pb,'update':pu}
  status('strict_pretrained_loading');model,ids,membership=load_model(runtime,meta,out);initial=digest(model);assert initial==meta['expected_initial_draft_sha256']==oldmeta['expected_initial_draft_sha256']
  saved={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
  check({'kind':'strict_loading','passed':True,'initial_sha256':initial})
  free,_=torch.cuda.mem_get_info();assert free+torch.cuda.memory_allocated()>11*2**30,'Insufficient actual T4 free VRAM'
  import transformers,triton
  dump(out/'environment.json',{'torch':torch.__version__,'triton':triton.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'actual_triton':True,'precision':'FP16 eager / FP32 master AdamW','loss_scale':LOSS_SCALE,'initial_draft_sha256':initial,'original_capture_data_sha256':meta['data_sha256'],'transport_gzip_sha256':meta['transport_gzip_sha256'],'all_logical_tensors_SHA_verified':True,'documents_sha256':meta['documents_sha256'],'legacy_data_sha256':oldmeta['data_sha256'],'source_sha256':{n:sha(ROOT/'code/original'/n) for n in ['probe_nemo_completion.py','stride_soft_cross_entropy.py','phase_separated_prediction.py','completion_protocol_20261007.json','fp32_master_adamw.py','lossless_input_transport.py']},'pinned_fused_sha256':sha(Path(a.nemo_root)/'nemo_automodel/components/loss/triton/soft_cross_entropy.py'),'scope':cfg['scope']});dump(out/'input_manifest.json',meta)
  if a.resume:
   payload_path=Path(a.resume);payload=json.loads(payload_path.read_text())
   prior={n:json.loads(payload[n]) for n in ['checks.json','rows.json','environment.json','trajectory_fingerprints.json','prediction_before_evaluation.json','evaluation_start.json','protocol_before_probe.json']}
   assert prior['protocol_before_probe.json']==cfg
   prior_manifest=json.loads(payload['original_output_manifest'])
   for n,v in payload.items():
    if n.endswith('.json') and 'completion/'+n in prior_manifest:assert __import__('hashlib').sha256(v.encode()).hexdigest()==prior_manifest['completion/'+n]
   oldenv=prior['environment.json'];newenv=json.loads((out/'environment.json').read_text())
   for key in ['torch','triton','transformers','gpu','precision','loss_scale','initial_draft_sha256','original_capture_data_sha256','transport_gzip_sha256','documents_sha256','legacy_data_sha256','pinned_fused_sha256']:assert oldenv[key]==newenv[key],key
   for key,h in oldenv['source_sha256'].items():
    if key!='probe_nemo_completion.py':assert newenv['source_sha256'][key]==h,key
   gates=prior['checks.json'];assert len(gates)==62 and all(x['passed'] for x in gates)
   priorrows=prior['rows.json'];assert len(priorrows)==33
   rows=[x for x in priorrows if x['kind']!='timing'];assert len(rows)==30
   refs=prior['trajectory_fingerprints.json'];assert set(refs)==set(map(str,sorted(set(cfg['calibration_lengths']+cfg['evaluation_lengths']+cfg['capacity_lengths']+cfg['timing_lengths']))))
   for name in ['checks.json','trajectory_fingerprints.json','prediction_before_evaluation.json','evaluation_start.json']: (out/name).write_text(payload[name])
   dump(out/'rows.json',rows)
   dump(out/'resume_provenance.json',{'prior_kernel_version':1,'prior_finished_by':'subprocess_timeout_1500s','payload_sha256':sha(payload_path),'original_output_sha256':prior_manifest,'source_before_resume':json.loads(payload['original_source_manifest']),'reused_checks':62,'reused_memory_and_control_rows':30,'discarded_prior_partial_timing_rows':3,'all_36_timing_rows_restarted_in_one_session':True,'model_weights_reloaded_from_unchanged_checkpoint':True,'qualification_and_numeric_contract_unchanged':True})
   state.update(checks_done=75,runs_done=30);dump(out/'status.json',state)
  if not a.resume:
   # Pretrained original-vs-stride recurring autograd qualification, no timing.
   status('pretrained_recurrent_gates')
   cases=[('main',d,m) for d in (1,3,7) for m in ('natural','unequal','empty')]+[('packed',7,'natural')]
   for name,depth,pattern in cases:
    cache=mask_case(olddata[name],pattern);values=[];audits=[]
    for mode in ('dense','stride'):
     clean();tr=trainer(mode,depth);kw=transfer(cache,'shift' if mode=='dense' else 'views',depth);teachers=[];teacher_before=kw['target_probs'].detach().cpu().clone()
     def pack(t):
      if t.dtype==torch.float32 and t.ndim in (2,3) and t.shape[-1]==32000 and not t.requires_grad:
       teachers.append({'ptr':t.untyped_storage().data_ptr(),'bytes':t.untyped_storage().nbytes(),'shape':list(t.shape),'offset':t.storage_offset(),'stride':list(t.stride())})
      return t.detach()
     with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):res=tr(**kw)
     res.loss.backward();values.append((metric_values(res),parameter_grads(model)));audits.append(teachers);assert torch.equal(kw['target_probs'].detach().cpu(),teacher_before) and digest(model)==initial
     tr=kw=res=teacher_before=None;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    m0,g0=values[0];m1,g1=values[1];details=compare_grads(g0,g1,cfg)
    passed=bool(torch.allclose(m0['loss'],m1['loss'],atol=cfg['loss_atol'],rtol=cfg['rtol'])) and all(torch.equal(m0[k],m1[k]) for k in m0 if k!='loss') and all(x['finite'] and x['close'] for x in details)
    storageok=len(audits[1])==depth and len({x['ptr'] for x in audits[1]})==1 and len(audits[0])==depth
    check({'kind':'recurrent','case':name,'depth':depth,'mask':pattern,'passed':passed and storageok,'gradients':details,'gradient_hashes_exact':all(x['shift_sha256']==x['views_sha256'] for x in details),'shared_teacher_storage':storageok,'teacher_wrapper_calls':depth,'teacher_audits':audits})
    values=audits=m0=m1=g0=g1=details=None;gc.collect()
   status('late_cache_gate');late=[]
   for mode in ('dense','stride'):
    clean();tr=trainer(mode);lossfn=runtime.masked_soft_cross_entropy;losses=[];cache0=[]
    def loss_hook(*x,**kw):loss=lossfn(*x,**kw);losses.append(loss);return loss
    def attn_hook(module,x,kw,result):
     if not cache0:k=kw['cache_hidden'][0][0];k.retain_grad();cache0.append(k)
    handle=model.model.layers[0].self_attn.register_forward_hook(attn_hook,with_kwargs=True);runtime.masked_soft_cross_entropy=loss_hook
    try:
     kw=transfer(olddata['main'],'shift' if mode=='dense' else 'views',7);res=tr(**kw);losses[-1].backward();g=cache0[0].grad
     assert g is not None and bool(torch.isfinite(g).all()) and float(g.float().norm())>0;late.append(g.detach().cpu().clone())
    finally:
     handle.remove();runtime.masked_soft_cross_entropy=lossfn;tr=kw=res=losses=cache0=g=None;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
   check({'kind':'late_cache','passed':bool(torch.allclose(late[0],late[1],atol=cfg['grad_atol'],rtol=cfg['rtol'])),'dense_sha256':thash(late[0]),'stride_sha256':thash(late[1]),'hash_exact':torch.equal(late[0],late[1])});late=None
  def gate(n):
   status('trajectory_'+str(n));rr=[]
   for mode in ('dense','split','stride'):
    clean();tr=trainer(mode);opt=MasterAdamW(model)
    try:
     for i in range(3):
      kw,res,norm,_=step(tr,opt,data[str(n)],mode);rec=state_record(opt,res,norm)
      if mode=='dense':rr.append(rec)
      else:check({'kind':'trajectory','length':n,'mode':mode,'update':i+1,'passed':records_match(rec,rr[i],cfg),'fingerprints_exact':rec['fingerprint']==rr[i]['fingerprint'],'loss_dense':rr[i]['loss'],'loss_candidate':rec['loss'],'loss_hash_exact':rec['loss_sha256']==rr[i]['loss_sha256']})
      opt.zero_grad();kw=res=norm=rec=None;gc.collect()
    finally:opt.zero_grad();tr=opt=kw=res=norm=None;clean()
   refs[str(n)]=rr;dump(out/'trajectory_fingerprints.json',refs)
  def measure(n,mode,order,kind='memory',isolated=True):
   clean();tr=trainer(mode);opt=MasterAdamW(model)
   try:
    kw,res,norm,_=step(tr,opt,data[str(n)],mode);opt.zero_grad();kw=res=norm=None;gc.collect();torch.cuda.synchronize()
    start=time.perf_counter();kw,res,norm,peaks=step(tr,opt,data[str(n)],mode,isolated);opt.zero_grad();torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
    reserved=torch.cuda.max_memory_reserved()/2**20;rec=state_record(opt,res,norm,with_grads=False);assert records_match(rec,refs[str(n)][1],cfg,ignore_grad=True)
    x={'kind':kind,'length':n,'mode':mode,'order':order,'counter':'isolated' if isolated else 'continuous','phase_peaks_mib':peaks,'full_peak_mib':max(peaks.values()),'peak_reserved_mib':reserved,'ms':ms,'update_matches_reference':True,'teacher_saved_storage_bytes':2*((7*n) if mode=='dense' else n+6)*32000*4}
    row(x);return x
   finally:opt.zero_grad();tr=opt=kw=res=norm=rec=None;clean()
  if not a.resume:
   # Calibration only; evaluation tensors have not been transferred or executed.
   for n in cfg['calibration_lengths']:
    gate(n)
    for order,modes in enumerate((('dense','split'),('split','dense'))):
     for mode in modes:measure(n,mode,order)
   # Qualify separate reset counters against one uninterrupted measurement.
   for mode in ('dense','split'):
    x=measure(256,mode,0,kind='counter_control',isolated=False);y=next(r for r in rows if r['kind']=='memory' and r['length']==256 and r['mode']==mode)
    check({'kind':'counter_control','mode':mode,'difference_mib':abs(x['full_peak_mib']-y['full_peak_mib']),'passed':abs(x['full_peak_mib']-y['full_peak_mib'])<=cfg['counter_control_error_mib']})
   prediction=predict(rows,cfg);prediction['calibration_rows_sha256']=__import__('hashlib').sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest();prediction['frozen_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());dump(out/'prediction_before_evaluation.json',prediction)
   dump(out/'evaluation_start.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'prediction_sha256':sha(out/'prediction_before_evaluation.json'),'calibration_rows':len(rows),'no_evaluation_draft_training_execution_before_freeze':True})
   for n in cfg['evaluation_lengths']:
    gate(n)
    for order,modes in enumerate((('dense','split'),('split','dense'))):
     for mode in modes:measure(n,mode,order)
   for n in sorted(set(cfg['capacity_lengths']+cfg['timing_lengths'])-set(cfg['calibration_lengths']+cfg['evaluation_lengths'])):gate(n)
  status('balanced_full_step_timings')
  for n in cfg['timing_lengths']:
   for order,modes in enumerate(cfg['timing_orders']):
    for mode in modes:measure(n,mode,order,kind='timing',isolated=False)
  status('predeclared_fused_capacity')
  for n in cfg['capacity_lengths']:
   for order,modes in enumerate(cfg['capacity_orders']):
    for mode in modes:
     clean();total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(cfg['allocator_budget_mib']*2**20/total)
     tr=opt=kw=res=norm=None;done=[];oom=None;phase='optimizer';torch.cuda.reset_peak_memory_stats()
     try:
      tr=trainer(mode);opt=MasterAdamW(model)
      for i in range(3):
       phase='update_'+str(i+1);kw,res,norm,peaks=step(tr,opt,data[str(n)],mode);torch.cuda.synchronize();rec=state_record(opt,res,norm);assert records_match(rec,refs[str(n)][i],cfg)
       done.append({'update':i+1,'reference_exact':rec['fingerprint']==refs[str(n)][i]['fingerprint'],'full_peak_mib':max(peaks.values()),'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20});opt.zero_grad();kw=res=norm=rec=None;gc.collect()
     except torch.OutOfMemoryError as e:oom={'type':type(e).__name__,'phase':phase,'message':str(e),'allocated_mib':torch.cuda.memory_allocated()/2**20,'reserved_mib':torch.cuda.memory_reserved()/2**20}
     finally:
      if opt is not None:opt.zero_grad()
      tr=opt=kw=res=norm=rec=None;clean()
     restored=digest(model)==initial;assert restored;row({'kind':'capacity','length':n,'mode':mode,'order':order,'allocator_budget_mib':cfg['allocator_budget_mib'],'outcome':'allocator_oom' if oom else 'complete','updates_done':len(done),'updates':done,'oom':oom,'weights_restored':restored})
  final=digest(model);check({'kind':'final_integrity','passed':final==initial,'initial':initial,'final':final,'checkpoint_written':False});dump(out/'weights_integrity.json',gates[-1])
  # 13CPU +1load +10rec +1late +48trajectory +2counter +1integrity =76; execution count12cal+2control+16eval+36timing+30capacity=96.
  assert len(gates)==63 and len(rows)==96,(len(gates),len(rows));state.update(state='complete',stage='done',checks_total=76,checks_done=76,runs_total=96,runs_done=96,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));dump(out/'status.json',state)
 except Exception:
  if model is not None and saved is not None:
   torch.cuda.set_per_process_memory_fraction(1.);restore(model,saved);dump(out/'weights_on_failure.json',{'restored':digest(model)==initial,'checkpoint_written':False})
  state.update(state='failed',error=traceback.format_exc());dump(out/'status.json',state);raise
if __name__=='__main__':main()
