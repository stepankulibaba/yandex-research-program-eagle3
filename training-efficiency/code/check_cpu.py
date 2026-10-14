"""Small random-model qualification after source relocation; no CUDA or weights."""
import ast
import copy
import os
from pathlib import Path
import sys

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code/original'))


def main():
    import torch
    from factorial_utils import optimizer_gate, scheduler_gate
    from fixed_path_matcher import cpu_gate
    import probe_nemo_ttt as base
    from nemo_fused_runtime import offset_trainer
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    optimizer = optimizer_gate()
    scheduler = scheduler_gate()
    source = (ROOT / 'EAGLE/eagle/model/utils.py').read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_posterior')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned posterior oracle>', 'exec'), namespace)
    oracle = cpu_gate(namespace['evaluate_posterior'])
    runtime = base.R
    model = runtime.LlamaEagle3DraftModel(base.config()).float()
    model.freeze_embeddings()
    ids = torch.tensor([0, 2, 4, 7, 11])
    membership = torch.zeros(19, dtype=torch.bool)
    membership[ids] = True
    docs = [{'tokens': torch.randint(0, 19, (1, n)), 'hidden': torch.randn(1, n, 16),
             'aux': torch.randn(1, n, 48), 'loss_mask': torch.ones(1, n, dtype=torch.long)} for n in [9, 5]]
    first = base.pack(docs, 14, 'cpu')
    second = base.pack(list(reversed(docs)), 14, 'cpu')
    batch = {key: torch.cat([first[key], second[key]], dim=0) for key in first}
    head = torch.randn(19, 16) * .1
    offset = offset_trainer(runtime)
    cases = 0
    for steps in [1, 3, 7]:
        for mask in ['all', 'unequal', 'empty']:
            current = {key: value.clone() for key, value in batch.items()}
            if mask == 'unequal':
                current['loss_mask'][0, ::2] = 0
                current['loss_mask'][1, 1::2] = 0
            if mask == 'empty':
                current['loss_mask'].zero_()
            with torch.no_grad():
                probs, positions = base.distribution(base.shift(torch.nn.functional.linear(current['hidden'], head)),
                                                     ids, membership, base.shift(current['loss_mask']))
            B, S, V = probs.shape
            storage = torch.zeros(B, S + steps - 1, V)
            storage[:, :S].copy_(probs)
            values = []
            before = copy.deepcopy(model.state_dict())
            for cls, target in [(runtime.Eagle3TrainerModule, probs), (offset, storage[:, :S])]:
                trainer = cls(model, selected_token_ids=ids, selected_token_mask=membership, ttt_steps=steps)
                trainer.zero_grad(set_to_none=True)
                result = base.forward(trainer, current, {'target_probs': target, 'position_mask': positions})
                result.loss.backward()
                values.append((base.metrics(result), base.grads(model)))
                assert all(torch.equal(model.state_dict()[key], tensor) for key, tensor in before.items())
            metrics, grads = values[0]
            other_metrics, other_grads = values[1]
            assert set(grads) == set(other_grads)
            assert all(torch.isfinite(tensor).all() for tensor in grads.values())
            assert all(torch.equal(metrics[key], other_metrics[key]) for key in metrics)
            assert all(torch.equal(grads[key], other_grads[key]) for key in grads)
            assert torch.equal(storage[:, :S], probs)
            cases += 1
    assert cases == 9
    print(f'CPU: optimizer {len(optimizer)}, scheduler {len(scheduler)}, posterior {len(oracle)}, B2 TTT loss/gradient pairs {cases}: PASS')
    print('Random FP32 CPU cases validate relocation and buffer semantics; not the Triton CUDA implementation.')

if __name__ == '__main__':
    main()
