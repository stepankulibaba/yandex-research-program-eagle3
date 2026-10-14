"""Strict real-Triton qualification; no fallback or performance claim."""
import argparse,hashlib,inspect,json,sys,time,traceback
from pathlib import Path
import torch


ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nemo-root',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);assert not out.exists(),'Inspect existing artifacts before rerunning'
    out.mkdir(parents=True);checks=[];state={'state':'starting','scope':'real pinned Triton component qualification; no recurrent model/timing'}
    def dump(name,obj):(out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    dump('status.json',state)
    try:
        assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
        nemo=Path(args.nemo_root).resolve();relative='nemo_automodel/components/loss/triton/soft_cross_entropy.py'
        expected=json.loads((ROOT/'Automodel/source_hashes.json').read_text())[relative]
        assert hashlib.sha256((nemo/relative).read_bytes()).hexdigest()==expected,'NeMo source mismatch'
        sys.path.insert(0,str(nemo))
        import triton
        from nemo_automodel.components.loss.triton import soft_cross_entropy as fused
        import stride_soft_cross_entropy as candidate
        assert candidate.HAVE_TRITON
        assert fused.HAVE_TRITON
        assert Path(inspect.getsourcefile(fused.SoftCrossEntropyFunction)).resolve()==(nemo/relative).resolve()
        torch.manual_seed(20261006);torch.backends.cuda.matmul.allow_tf32=False
        env={'torch':torch.__version__,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(),
            'pinned_source_sha256':expected,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256((ROOT/'code/original/stride_soft_cross_entropy.py').read_bytes()).hexdigest(),
            'precision':'synthetic FP32/FP16 logits; FP32 frozen teacher','actual_triton':True,
            'loss_atol_fp32':2e-5,'loss_atol_fp16':2e-4,'rtol':2e-3,'scope':state['scope']}
        dump('environment.json',env);state.update(state='running',checks_done=0,checks_total=108);dump('status.json',state)
        for B in (1,2,3):
         for dtype in (torch.float32,torch.float16):
          for V in (19,257,32000):
           for depth in (0,6):
            for pattern in ('all','unequal','empty'):
             S=11;storage=torch.zeros(B,S+6,V,device='cuda',dtype=torch.float32)
             storage[:,:S]=torch.softmax(torch.randn(B,S,V,device='cuda'),-1)*torch.linspace(.5,1.5,S,device='cuda')[None,:,None]
             p=storage[:,depth:depth+S];m=torch.ones(B,S,1,device='cuda')
             if pattern=='unequal':
              m[0,1:]=0
              if B>1:m[1].zero_()
              if B>2:m[2,::2]=0
             if pattern=='empty':m.zero_()
             teacher_before=storage.clone()
             original=torch.randn(B,S,V,device='cuda',dtype=dtype)
             x=original.clone().requires_grad_();ref=original.clone().requires_grad_();dense=original.clone().requires_grad_()
             expected_loss=(-(p*torch.log_softmax(ref.float(),-1)).sum(-1)*m.squeeze(-1)).sum()/m.sum().clamp_min(1.)
             baseline=fused.fused_soft_cross_entropy(dense,p.contiguous(),m)
             target_storages=[]
             def save(t):
              if t.untyped_storage().data_ptr()==storage.untyped_storage().data_ptr():target_storages.append(t.untyped_storage().data_ptr())
              return t
             with torch.autograd.graph.saved_tensors_hooks(save,lambda t:t):
              actual=candidate.fused_soft_cross_entropy(x,p,m)
             incoming=torch.tensor(.37,device='cuda');expected_loss.backward(incoming);baseline.backward(incoming);actual.backward(incoming)
             atol=env['loss_atol_fp32'] if dtype==torch.float32 else env['loss_atol_fp16']
             loss_close=bool(torch.allclose(actual,expected_loss,atol=atol,rtol=env['rtol']) and torch.allclose(actual,baseline,atol=atol,rtol=env['rtol']))
             grad_close=bool(torch.allclose(x.grad,ref.grad,atol=atol,rtol=env['rtol']) and torch.allclose(x.grad,dense.grad,atol=atol,rtol=env['rtol']))
             launches=1
             storage_ok=len(target_storages)==launches and len(set(target_storages))==1
             row={'batch':B,'dtype':str(dtype),'vocab':V,'depth':depth,'mask':pattern,
                'loss_abs':float(abs(actual-expected_loss)),'grad_abs':float((x.grad.float()-ref.grad.float()).abs().max()),
                'loss_close':loss_close,'gradient_close':grad_close,'one_shared_teacher_storage':storage_ok,
                'gradient_dense_exact':torch.equal(x.grad,dense.grad),'loss_dense_exact':torch.equal(actual,baseline),'teacher_unchanged':torch.equal(storage,teacher_before),
                'passed':loss_close and grad_close and storage_ok and torch.equal(x.grad,dense.grad) and torch.equal(actual,baseline) and torch.equal(storage,teacher_before)}
             checks.append(row);dump('checks.json',checks);state['checks_done']=len(checks);dump('status.json',state);assert row['passed'],row
             del storage,teacher_before,p,m,original,x,ref,dense,actual,expected_loss,baseline,incoming;torch.cuda.empty_cache()
        state.update(state='complete',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));dump('status.json',state)
    except Exception:
        state.update(state='failed',error=traceback.format_exc());dump('status.json',state);raise

if __name__=='__main__':main()
