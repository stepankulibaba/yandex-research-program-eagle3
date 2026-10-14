"""Balanced 2x2 metadata ablation in a fresh process; no AR or profiler timings."""
import argparse
import contextlib
import csv
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import run_pilot as pilot
from metadata import finish_tensor_metadata
from topk_patch import make_topk
from loading import read_only_safetensors

MODES=('upstream','leaf_cpu','tensor_gpu_leaf','tensor_cpu')
DESCRIPTIONS={
 'upstream':'author CPU loops and GPU unique(parent).tolist()',
 'leaf_cpu':'author CPU loops; sorted(set(already copied parent list))',
 'tensor_gpu_leaf':'CPU tensor metadata; retained GPU unique(parent).tolist()',
 'tensor_cpu':'CPU tensor metadata and CPU leaf detection (pilot B1)',
}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--prompts',default=str(pilot.ROOT/'prompts.json'))
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--total-nodes',type=int,default=64)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--context-pad-tokens',type=int,default=0)
    args=p.parse_args()
    modes=args.modes
    assert 'upstream' in modes
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    dump=lambda name,x:pilot.dump(out/name,x)
    dump('status.json',dict(state='loading',completed_runs=0))
    pilot.verify_vendor()
    random.seed(20261005); torch.manual_seed(20261005)
    base,bi=pilot.resolve_checkpoint('Qwen/Qwen3-1.7B',None,[])
    draft,di=pilot.resolve_checkpoint('AngelSlim/Qwen3-1.7B_eagle3',None,[])
    from eagle.model.ea_model import EaModel
    torch.cuda.init()
    with read_only_safetensors():
        model=EaModel.from_pretrained(use_eagle3=True,base_model_path=base,ea_model_path=draft,
            total_token=args.total_nodes,depth=args.depth,top_k=args.top_k,torch_dtype=torch.float16,device_map={'':0}).eval()
    tok=model.get_tokenizer()
    prompts=json.loads(Path(args.prompts).read_text())
    encoded=[]
    for x in prompts:
        text=tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,
                                    add_generation_prompt=True,enable_thinking=False)
        ids=torch.tensor(tok([text]).input_ids,device='cuda')
        if args.context_pad_tokens:
            # Identical extra prefix tokens in every variant, with recorded IDs.
            neutral=tok('Reference notes. The following question is independent of these notes. ',add_special_tokens=False).input_ids
            pad=(neutral*((args.context_pad_tokens+len(neutral)-1)//len(neutral)))[:args.context_pad_tokens]
            ids=torch.cat((torch.tensor([pad],device='cuda'),ids),dim=1)
        encoded.append((x,ids))
    capacity=max(ids.shape[1] for _,ids in encoded)+args.tokens+args.total_nodes+96
    env=dict(python=platform.python_version(),os=platform.platform(),torch=torch.__version__,
        transformers=pilot.transformers.__version__,gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,num_threads=torch.get_num_threads(),
        num_interop_threads=torch.get_num_interop_threads(),eagle_commit=pilot.COMMIT,
        attention='pinned explicit eager',free_total_bytes=torch.cuda.mem_get_info())
    dump('environment.json',env)
    dump('config.json',dict(base=bi,draft=di,modes={m:DESCRIPTIONS[m] for m in modes},
        repeats=args.repeats,tokens=args.tokens,cache_capacity=capacity,seed=20261005,
        total_nodes=args.total_nodes,depth=args.depth,top_k=args.top_k,precision='fp16',batch=1,
        context_pad_tokens=args.context_pad_tokens,
        order=f'Cyclic Latin rotation over {len(modes)} modes; balanced positions when repeats are a multiple of mode count',
        stopping='EOS disabled, upstream strict > cap, actual overshoot recorded',
        timing_scope='wall including prefill and initial draft; CUDA sync only at endpoints',
        source_sha256={name:hashlib.sha256((pilot.ROOT/'code/original'/name).read_bytes()).hexdigest()
         for name in ('run_ablation.py','metadata.py','topk_patch.py','run_pilot.py','loading.py')},
        loading='read-only safetensors mapping; copies/conversions unchanged',
        prompts_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest()))
    dump('prompts.json',prompts)
    torch.save({x['id']:ids.cpu() for x,ids in encoded},out/'encoded_prompts.pt')
    original=model.ea_layer.topK_genrate
    variants={m:(original if m=='upstream' else make_topk(model.ea_layer,original,variant=m)) for m in modes}
    @contextlib.contextmanager
    def no_eos():
        eos=tok.eos_token_id; tok.eos_token_id=-1
        try: yield
        finally: tok.eos_token_id=eos
    def generate(ids,n=args.tokens):
        with torch.no_grad(),no_eos():
            return model.eagenerate(ids,temperature=0,max_new_tokens=n,max_length=capacity,log=True)
    # Exact diagnostic replay on actual captured GPU parents, outside timed runs.
    captured=[]
    def observer(parents,result):
        if len(captured)<24:
            captured.append((parents.detach().cpu().clone(),tuple(x.cpu().clone() for x in result)))
    model.ea_layer.topK_genrate=make_topk(model.ea_layer,original,observer=observer)
    generate(encoded[0][1],32)
    for parents,expected in captured:
        for gpu_leaf in (False,True):
            result=finish_tensor_metadata(expected[0][0].cuda(),parents.cuda(),'cuda',None,gpu_unique=gpu_leaf)
            assert all(torch.equal(a.cpu(),b) for a,b in zip(result,expected))
        assert torch.unique(parents).tolist()==sorted(set(parents.tolist()))
    for mode,fn in variants.items():
        model.ea_layer.topK_genrate=fn
        for x,ids in (encoded[0],encoded[-1]): generate(ids,32)
    print('WARM_AND_REPLAY_COMPLETE',len(captured),flush=True)
    checks=[];rows=[];outputs={};telemetry=[]
    for rep in range(args.repeats):
        # Keep pairs together, rotate prompt order between repetitions.
        prompt_order=list(range(len(encoded)))
        shift=rep%len(encoded); prompt_order=prompt_order[shift:]+prompt_order[:shift]
        for j in prompt_order:
            x,ids=encoded[j]; order=modes[(rep+j)%len(modes):]+modes[:(rep+j)%len(modes)]
            group={}
            for order_index,mode in enumerate(order):
                model.ea_layer.topK_genrate=variants[mode]
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                start=time.perf_counter();result,new,idx=generate(ids)
                torch.cuda.synchronize();sec=time.perf_counter()-start
                tokens=result[0,ids.shape[1]:].tolist();new=int(new);cycles=int(idx)+1
                assert len(tokens)==new
                row=dict(rep=rep,prompt_id=x['id'],group_id=x.get('group_id',x['id'].replace('_en_','_').replace('_ru_','_')),
                    language=x['language'],mode=mode,order_index=order_index,input_tokens=int(ids.shape[1]),
                    output_tokens=new,cycles=cycles,tau=new/cycles,sec=sec,ms_per_token=sec*1000/new,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                rows.append(row);group[mode]=(tokens,cycles);outputs[rep,x['id'],mode]=tokens
                dump('rows.json',rows);dump('status.json',dict(state='running',completed_runs=len(rows),
                    expected_runs=args.repeats*len(encoded)*len(modes)))
                print('RUN',rep,x['id'],mode,round(row['ms_per_token'],3),'ms/token',flush=True)
            reference=group['upstream']
            exact=all(pair==reference for pair in group.values())
            checks.append(dict(rep=rep,prompt_id=x['id'],all_variants_full_exact=exact))
            dump('correctness.json',dict(replay_trees=len(captured),all_metadata_exact=True,generation_checks=checks))
            if not exact:
                raise AssertionError('A metadata variant changed tokens or cycles')
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                telemetry.append(dict(rep=rep,prompt_id=x['id'],metrics_csv=r.stdout.strip(),exit_code=r.returncode))
            except (OSError,subprocess.TimeoutExpired) as e:
                telemetry.append(dict(rep=rep,prompt_id=x['id'],error=type(e).__name__))
    with (out/'rows.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    torch.save(outputs,out/'generated_tokens.pt');dump('telemetry.json',telemetry)
    dump('status.json',dict(state='complete',completed_runs=len(rows)))
    print('ABLATION_COMPLETE',flush=True)


if __name__=='__main__': main()
