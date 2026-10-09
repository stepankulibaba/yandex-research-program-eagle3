"""Scoped NeMo adapter: canonical inputs/mapping and exact optimizer timing."""
import argparse
import inspect
import os
from pathlib import Path

from runtime import BenchmarkWindow, atomic_json, digest, read_json
from training_data import canonical_nemo_loader, mapping


def main():
    import torch
    import nemo_automodel.recipes.llm.train_eagle3 as recipe
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--warmup', type=int, required=True)
    args = parser.parse_args()
    # Do not silently adapt another API after an upstream change.
    if 'apply_draft_compile' not in inspect.getsource(recipe.TrainEagle3Recipe.setup):
        raise RuntimeError('Unsupported NeMo recipe API')
    original_loop = recipe.TrainEagle3Recipe._train_epochs
    original_forward = recipe.TrainEagle3Recipe._forward_batch

    def loader(self, data_path, split=None):
        cfg = self.cfg.recipe_args
        return canonical_nemo_loader(args.data, packed=bool(cfg.get('packed_sequence_size', 0)),
                                     batch_size=int(cfg.micro_batch_size))
    recipe.TrainEagle3Recipe._build_train_dataloader = loader

    def shared_mapping(dataloader, *, target_vocab_size, **kwargs):
        return mapping(args.mapping, target_vocab_size)
    recipe.load_or_build_eagle3_token_mapping = shared_mapping

    def loop(self, *loop_args, **loop_kwargs):
        cfg = self.cfg.recipe_args
        if self.compute_dtype != torch.bfloat16 or cfg.get('draft_gradient_checkpointing', False):
            raise RuntimeError('Matched baseline requires BF16 and no activation checkpointing')
        config = {'dtype': 'bfloat16', 'target_attention': cfg.get('target_attn_implementation'),
                  'draft_attention': cfg.get('draft_attn_implementation'), 'seq_length': 2048,
                  'micro_batch': int(cfg.micro_batch_size), 'accumulation': self.grad_accumulation_steps,
                  'packing': int(cfg.get('packed_sequence_size', 0)), 'ttt_steps': int(cfg.ttt_steps),
                  'mapping_sha256': digest(read_json(args.mapping)), 'lr': self.peak_lr,
                  'compile_requested': bool(self.cfg.get('compile.enabled', False)),
                  'compile_applied': bool(getattr(self.draft_model, '_compiled_call_impl', None)),
                  'fp8_requested': bool(self.cfg.get('fp8.enabled', False)),
                  'fp8_modules': [n for n, m in self.draft_model.named_modules() if 'Float8' in type(m).__name__]}
        if config['compile_requested'] and not config['compile_applied']:
            raise RuntimeError('Requested torch.compile was not applied')
        if config['fp8_requested'] and not config['fp8_modules']:
            raise RuntimeError('Requested FP8 was not applied')
        self._speed_window = BenchmarkWindow(args.warmup, self.total_optim_steps, 'nemo', config)
        step = self.optimizer.step
        completed = 0

        def measured_step(*a, **kw):
            nonlocal completed
            result = step(*a, **kw)
            completed += 1
            self._speed_window.after_step(completed, torch)
            return result
        self.optimizer.step = measured_step
        try:
            result = original_loop(self, *loop_args, **loop_kwargs)
            atomic_json(args.out, self._speed_window.result(torch))
            return result
        finally:
            self.optimizer.step = step

    def forward(self, batch, target_batch=None):
        window = self._speed_window
        window.before_batch(self.runtime.global_step, batch, torch)
        batch = {k: v for k, v in batch.items() if not k.startswith('_')}
        metrics = original_forward(self, batch, target_batch)
        if not bool(torch.isfinite(metrics.loss.detach())):
            window.finite_loss = False
            raise RuntimeError('Nonfinite training loss')
        return metrics
    recipe.TrainEagle3Recipe._train_epochs = loop
    recipe.TrainEagle3Recipe._forward_batch = forward
    # Speed-only adapters deliberately exclude final checkpoint/eval cost.
    recipe.TrainEagle3Recipe.save_checkpoint = lambda *a, **kw: None
    import sys
    sys.argv = [sys.argv[0], '-c', args.config]
    recipe.main()


if __name__ == '__main__':
    main()
