"""Explicit FP16 compute / FP32 master AdamW; no implicit dtype rescue."""
import torch

LOSS_SCALE=128.0
OPTIMIZER_KWARGS=dict(lr=1e-4,betas=(0.9,0.95),eps=1e-8,weight_decay=0.0,foreach=False,fused=False)

class MasterAdamW:
    def __init__(self,model):
        self.named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert self.named and all(p.dtype==torch.float16 for _,p in self.named)
        self.masters=[torch.nn.Parameter(p.detach().to(torch.float32)) for _,p in self.named]
        self.optimizer=torch.optim.AdamW(self.masters,**OPTIMIZER_KWARGS)

    def step(self):
        for (_,p),master in zip(self.named,self.masters):
            if p.grad is None:raise RuntimeError('Missing trainable gradient')
            master.grad=p.grad.detach().to(torch.float32).mul_(1.0/LOSS_SCALE)
            p.grad=None
        norm=torch.nn.utils.clip_grad_norm_(self.masters,1.0,foreach=False,error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for (_,p),master in zip(self.named,self.masters):p.copy_(master)
        return norm

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for _,p in self.named:p.grad=None

    def persistent_bytes(self):
        master=sum(p.numel()*p.element_size() for p in self.masters)
        states=sum(v.numel()*v.element_size() for st in self.optimizer.state.values() for v in st.values() if isinstance(v,torch.Tensor))
        return {'master_bytes':master,'optimizer_state_bytes':states,'total':master+states}


def cpu_gate():
    """Same quantized incoming gradients vs independent FP32 torch AdamW."""
    records=[]
    for label,values in [('small',[1e-4,1e-7,.01]),('zero',[0.,0.,0.]),('mixed',[.25,0.,1e-6]),('accumulate',[.01,.01,.01])]:
        model=torch.nn.Linear(3,1,bias=False).half()
        with torch.no_grad():model.weight.copy_(torch.tensor([[.1,1.,.001]],dtype=torch.float16))
        opt=MasterAdamW(model)
        reference=torch.nn.Parameter(model.weight.detach().float().clone());refopt=torch.optim.AdamW([reference],**OPTIMIZER_KWARGS)
        for step in range(3):
            g=torch.tensor([values],dtype=torch.float32)*(1 if step%2==0 or label=='accumulate' else -1)
            incoming=(g*LOSS_SCALE).half();model.weight.grad=incoming.clone()
            reference.grad=incoming.float()/LOSS_SCALE
            torch.nn.utils.clip_grad_norm_([reference],1.,foreach=False,error_if_nonfinite=True)
            refopt.step();opt.step()
            same=torch.equal(reference,opt.masters[0]) and torch.equal(reference.half(),model.weight)
            for key in ('exp_avg','exp_avg_sq','step'):same=same and torch.equal(refopt.state[reference][key],opt.optimizer.state[opt.masters[0]][key])
            records.append({'case':label,'step':step+1,'exact_fp32_reference':same,'finite':bool(torch.isfinite(model.weight).all() and torch.isfinite(reference).all()),'passed':same and bool(torch.isfinite(model.weight).all())})
            opt.zero_grad();refopt.zero_grad(set_to_none=True)
    model=torch.nn.Linear(3,1,bias=False).half();before=model.weight.detach().clone();opt=MasterAdamW(model);model.weight.grad=torch.full_like(model.weight,float('nan'));rejected=False
    try:opt.step()
    except RuntimeError:rejected=True
    records.append({'case':'nonfinite_rejected_before_update','passed':rejected and torch.equal(before,model.weight) and not opt.optimizer.state})
    assert len(records)==13 and all(r['passed'] for r in records)
    return records
