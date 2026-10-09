"""Training-speed run of NeMo AutoModel's EAGLE-3 recipe on our shared data, timed by SpeedWindow.

    python train_nemo.py --config work/<run>.yaml --data canonical.jsonl --mapping selected_tokens.json
                         --out <run>.speed.json --warmup 20

The recipe itself (nemo_automodel.recipes.llm.train_eagle3, pinned commit) runs unchanged, except for four
replacements made before it starts:
  1. the dataloader -> our shared corpus (training_common.nemo_loader), the same rows the authors' trainer gets;
  2. the draft-vocabulary mapping -> our shared 32K mapping;
  3. the training loop and the forward pass are wrapped to feed the stopwatch and to stop on a non-finite loss;
  4. checkpoint saving is switched off (it is not part of the step time).
"""
import argparse
import inspect

from common import atomic_json, digest, read_json
from training_common import SpeedWindow, mapping, nemo_loader


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

    Recipe = recipe.TrainEagle3Recipe
    # The replacements below rely on the pinned recipe's internals; refuse to run against another version.
    if 'apply_draft_compile' not in inspect.getsource(Recipe.setup):
        raise RuntimeError('Unsupported NeMo recipe API')
    original_train_epochs = Recipe._train_epochs
    original_forward_batch = Recipe._forward_batch

    # 1. Dataloader.
    def shared_dataloader(self, data_path, split=None):
        cfg = self.cfg.recipe_args
        return nemo_loader(args.data, packed=bool(cfg.get('packed_sequence_size', 0)),
                           batch_size=int(cfg.micro_batch_size))

    # 2. Draft vocabulary.
    def shared_mapping(dataloader, *, target_vocab_size, **kwargs):
        return mapping(args.mapping, target_vocab_size)

    # 3a. Training loop: describe the run, check that compile/FP8 were really applied, time the optimizer steps.
    def timed_train_epochs(self, *loop_args, **loop_kwargs):
        cfg = self.cfg.recipe_args
        if self.compute_dtype != torch.bfloat16 or cfg.get('draft_gradient_checkpointing', False):
            raise RuntimeError('Matched baseline requires BF16 and no activation checkpointing')
        fp8_modules = [name for name, m in self.draft_model.named_modules() if 'Float8' in type(m).__name__]
        config = {'dtype': 'bfloat16', 'target_attention': cfg.get('target_attn_implementation'),
                  'draft_attention': cfg.get('draft_attn_implementation'), 'seq_length': 2048,
                  'micro_batch': int(cfg.micro_batch_size), 'accumulation': self.grad_accumulation_steps,
                  'packing': int(cfg.get('packed_sequence_size', 0)), 'ttt_steps': int(cfg.ttt_steps),
                  'mapping_sha256': digest(read_json(args.mapping)), 'lr': self.peak_lr,
                  'compile_requested': bool(self.cfg.get('compile.enabled', False)),
                  'compile_applied': bool(getattr(self.draft_model, '_compiled_call_impl', None)),
                  'fp8_requested': bool(self.cfg.get('fp8.enabled', False)),
                  'fp8_modules': fp8_modules}
        if config['compile_requested'] and not config['compile_applied']:
            raise RuntimeError('Requested torch.compile was not applied')
        if config['fp8_requested'] and not fp8_modules:
            raise RuntimeError('Requested FP8 was not applied')

        self._speed_window = SpeedWindow(args.warmup, self.total_optim_steps, 'nemo', config)
        optimizer_step = self.optimizer.step
        completed = 0

        def timed_step(*a, **kw):
            nonlocal completed
            result = optimizer_step(*a, **kw)
            completed += 1
            self._speed_window.after_step(completed, torch)
            return result

        self.optimizer.step = timed_step
        try:
            result = original_train_epochs(self, *loop_args, **loop_kwargs)
            atomic_json(args.out, self._speed_window.result(torch))
            return result
        finally:
            self.optimizer.step = optimizer_step

    # 3b. Forward pass: count the micro-batch, drop our bookkeeping keys, stop on a non-finite loss.
    def counted_forward_batch(self, batch, target_batch=None):
        self._speed_window.before_batch(self.runtime.global_step, batch, torch)
        batch = {k: v for k, v in batch.items() if not k.startswith('_')}
        metrics = original_forward_batch(self, batch, target_batch)
        if not bool(torch.isfinite(metrics.loss.detach())):
            self._speed_window.finite_loss = False
            raise RuntimeError('Nonfinite training loss')
        return metrics

    Recipe._build_train_dataloader = shared_dataloader
    recipe.load_or_build_eagle3_token_mapping = shared_mapping
    Recipe._train_epochs = timed_train_epochs
    Recipe._forward_batch = counted_forward_batch
    Recipe.save_checkpoint = lambda *a, **kw: None          # 4. no checkpoints

    import sys
    sys.argv = [sys.argv[0], '-c', args.config]
    recipe.main()


if __name__ == '__main__':
    main()
