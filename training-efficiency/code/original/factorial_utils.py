"""Deterministic exact token-budget schedule and qualified cached-prefix truncation."""
import random

def token_schedule(documents,budget):
    assert documents and budget>0
    result=[];used=0;i=0
    while used+documents[i%len(documents)]['length']<=budget:
        d=documents[i%len(documents)];result.append({'id':d['id'],'length':d['length'],'minimum':d['minimum'],'cropped':False})
        used+=d['length'];i+=1
    if used<budget:
        d=documents[i%len(documents)];remaining=budget-used;need=max(0,d['minimum']-remaining)
        for row in reversed(result):
            move=min(need,row['length']-row['minimum']);row['length']-=move;row['cropped']=move>0 or row['cropped'];need-=move;remaining+=move
            if not need:break
        assert not need,'Insufficient budget for supervised final document'
        result.append({'id':d['id'],'length':remaining,'minimum':d['minimum'],'cropped':remaining<d['length']})
    assert sum(x['length'] for x in result)==budget and all(x['length']>=x['minimum'] for x in result)
    return result

def scheduler_gate():
    rng=random.Random(20261008);rows=[]
    for case in range(32):
        ds=[{'id':str(i),'length':rng.randint(30,384),'minimum':10} for i in range(rng.randint(2,24))]
        budget=sum(d['length'] for d in ds)*rng.randint(2,16)+rng.randint(1,29)
        plan=token_schedule(ds,budget);again=token_schedule(ds,budget)
        lookup={d['id']:d for d in ds}
        assert plan==again and sum(p['length'] for p in plan)==budget
        assert [p['id'] for p in plan]==[ds[i%len(ds)]['id'] for i in range(len(plan))]
        assert all(10<=p['length']<=lookup[p['id']]['length'] for p in plan)
        rows.append({'case':case,'budget':budget,'tokens':sum(p['length'] for p in plan),'passed':True})
    return rows

def crop_cache(cache,n):
    import torch
    full=cache['input_ids'].shape[1];assert 1<n<=full
    if n==full:return cache
    out={}
    for k,v in cache.items():
        if k=='seq_lens':out[k]=torch.tensor([[n]],dtype=v.dtype,device=v.device)
        elif k=='doc_remaining':out[k]=torch.arange(n-1,-1,-1,device=v.device,dtype=v.dtype).view(1,-1)
        else:out[k]=v[:,:n].clone()
    out['input_ids'][:,-1]=0;out['loss_mask'][:,-1]=0
    out['position_mask'][:,-1]=False
    out['target_probs'][:,-1].fill_(1.0/out['target_probs'].shape[-1])
    assert out['position_mask'].any()
    return out

def optimizer_gate():
    import torch
    from fp32_master_adamw import MasterAdamW,LOSS_SCALE,OPTIMIZER_KWARGS
    rows=[]
    for lr in [1e-4,1e-5]:
        for label,values in [('small',[1e-4,1e-7,.01]),('zero',[0.,0.,0.]),('mixed',[.25,0.,1e-6]),('accumulate',[.01,.01,.01])]:
            model=torch.nn.Linear(3,1,bias=False).half()
            with torch.no_grad():model.weight.copy_(torch.tensor([[.1,1.,.001]],dtype=torch.float16))
            opt=MasterAdamW(model);opt.optimizer.param_groups[0]['lr']=lr
            reference=torch.nn.Parameter(model.weight.detach().float().clone());refopt=torch.optim.AdamW([reference],**{**OPTIMIZER_KWARGS,'lr':lr})
            for step in range(3):
                g=torch.tensor([values])*(1 if step%2==0 or label=='accumulate' else -1);incoming=(g*LOSS_SCALE).half()
                model.weight.grad=incoming.clone();reference.grad=incoming.float()/LOSS_SCALE
                torch.nn.utils.clip_grad_norm_([reference],1.,foreach=False,error_if_nonfinite=True)
                refopt.step();opt.step()
                same=torch.equal(reference,opt.masters[0]) and torch.equal(reference.half(),model.weight)
                same=same and all(torch.equal(refopt.state[reference][k],opt.optimizer.state[opt.masters[0]][k]) for k in ('exp_avg','exp_avg_sq','step'))
                assert same;rows.append({'lr':lr,'case':label,'step':step+1,'passed':same});opt.zero_grad();refopt.zero_grad(set_to_none=True)
        model=torch.nn.Linear(3,1,bias=False).half();before=model.weight.detach().clone();opt=MasterAdamW(model);opt.optimizer.param_groups[0]['lr']=lr
        model.weight.grad=torch.full_like(model.weight,float('nan'));rejected=False
        try:opt.step()
        except RuntimeError:rejected=True
        assert rejected and torch.equal(before,model.weight) and not opt.optimizer.state
        rows.append({'lr':lr,'case':'nonfinite_rejected','passed':True})
    assert len(rows)==26;return rows
