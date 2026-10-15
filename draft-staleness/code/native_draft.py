"""Pinned-author chain verification and a parity-checked training adapter.

The authoritative path calls the vendored, unmodified EAGLE CNets model with
its real incremental KV cache. It recomputes the prefix at each cycle: this is a
correctness evaluator, not a wall-clock speed benchmark. The training adapter
uses the existing vectorized teacher-forced unroll, with local arithmetic shims
and packed states for ONLY the official three-state Llama EAGLE-3 architecture.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import types

import numpy as np
import torch


def _local_module(suffix: str):
    prefix = f"{__package__}." if __package__ else ""
    return importlib.import_module(prefix + suffix)


def _checkpoint_state(path: Path) -> dict:
    files = sorted(path.glob("*.safetensors"))
    if files:
        from safetensors.torch import load_file
        state = {}
        for file in files:
            shard = load_file(str(file), device="cpu")
            if set(state) & set(shard):
                raise ValueError("duplicate state keys across checkpoint shards")
            state.update(shard)
        return state
    filename = path / "pytorch_model.bin"
    if not filename.exists():
        raise FileNotFoundError(f"No safetensors or pytorch_model.bin in {path}")
    return torch.load(filename, map_location="cpu", weights_only=True)


def _native_rms(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Match CNets: normalize in FP32, cast, THEN multiply the learned weight."""
    dtype = x.dtype
    normalized = x.float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + eps)
    normalized = normalized.to(dtype)
    return normalized if weight is None else weight * normalized


def _features(features: torch.Tensor, hidden_size: int, device, dtype) -> torch.Tensor:
    tensor = torch.as_tensor(features, device=device, dtype=dtype)
    if tensor.ndim == 3 and tuple(tensor.shape[1:]) == (3, hidden_size):
        return tensor.flatten(1)
    if tensor.ndim == 2 and tensor.shape[1] == 3 * hidden_size:
        return tensor
    raise ValueError("features must have shape [T, 3, H] or [T, 3*H]")


class NativeDraft:
    """Frozen official draft with FC replacement and real incremental decoding.

    ``base_embed_cpu`` is always copied from the UNCHANGED base target. No edited
    target embedding may be used, even if a checkpoint happens to contain one.
    Public model configuration and d2t mapping are loaded from checkpoint_dir.
    """

    def __init__(self, checkpoint_dir, base_embed_cpu, device="cuda", *, dtype=torch.bfloat16):
        path = Path(checkpoint_dir)
        cnets = _local_module("vendor.eagle.model.cnets")
        config_module = _local_module("vendor.eagle.model.configs")
        raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config = config_module.EConfig(**raw_config)
        # Recent Transformers canonicalizes an absent/None rope_scaling into a
        # new rope_type dictionary. CNets expects its original None or old-style
        # type/factor fields; preserve the checkpoint's exact original setting.
        config.rope_scaling = raw_config.get("rope_scaling")
        if int(config.pretraining_tp) != 1:
            raise ValueError("pilot supports pretraining_tp=1 only")
        if getattr(config, "target_hidden_size", config.hidden_size) != config.hidden_size:
            raise ValueError("pilot requires target and draft hidden sizes to match")
        self.model = cnets.Model(config, load_emb=False, bias=False)
        state = _checkpoint_state(path)
        # Freeze the baseline embedding explicitly rather than borrowing a live
        # target Parameter. This also prevents LoRA merge/swap aliasing.
        state.pop("embed_tokens.weight", None)
        incompatible = self.model.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != {"embed_tokens.weight"} or incompatible.unexpected_keys:
            raise ValueError(f"Unexpected checkpoint structure: {incompatible}")
        embedding = torch.as_tensor(base_embed_cpu).detach().clone()
        if tuple(embedding.shape) != tuple(self.model.embed_tokens.weight.shape):
            raise ValueError("base embedding shape does not match draft configuration")
        self.model.embed_tokens.weight.data.copy_(embedding)
        self.model.to(device=device, dtype=dtype).eval()
        self.model.requires_grad_(False)
        self.model.reset()
        self.device, self.dtype = torch.device(device), dtype
        self.hidden_size = int(config.hidden_size)
        self.d2t_ids = torch.arange(config.draft_vocab_size, device=self.device) + self.model.d2t.long()
        if self.d2t_ids.unique().numel() != self.d2t_ids.numel():
            raise ValueError("draft d2t mapping must be unique")
        if int(self.d2t_ids.min()) < 0 or int(self.d2t_ids.max()) >= int(config.vocab_size):
            raise ValueError("draft d2t mapping is outside target vocabulary")

    def get_fc(self) -> torch.Tensor:
        return self.model.fc.weight.detach().float().cpu().clone()

    @property
    def fc_numpy(self) -> np.ndarray:
        return self.get_fc().numpy()

    def set_fc(self, tensor) -> None:
        matrix = torch.as_tensor(tensor, device=self.device, dtype=self.dtype)
        if matrix.shape != self.model.fc.weight.shape or not torch.isfinite(matrix).all():
            raise ValueError("FC replacement needs the original shape and finite values")
        with torch.no_grad():
            self.model.fc.weight.copy_(matrix)

    @torch.no_grad()
    def prefix_logits(self, features, ids, root_t: int, max_depth: int = 6) -> list[torch.Tensor]:
        """Teacher-forced incremental logits at a single verification start.

        This deliberately follows supplied tokens at all available depths; only
        the caller's leading correct proposals can count toward acceptance.
        It is also the independent reference used for vectorized-unroll parity.
        """
        features = _features(features, self.hidden_size, self.device, self.dtype)
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        total = ids.numel()
        if ids.ndim != 1 or features.shape[0] != total:
            raise ValueError("features and one-dimensional ids must have equal length")
        if not 0 <= root_t < total - 1 or max_depth < 1:
            raise ValueError("root_t needs an existing root token and max_depth >= 1")
        if total - root_t - 2 == 0:
            return []
        self.model.reset()
        # Features x_0..x_t are paired with shifted inputs x_1..x_(t+1).
        hidden, cache = self.model(
            features[:root_t + 1][None], input_ids=ids[1:root_t + 2][None], use_cache=True,
        )
        previous = hidden[:, -1:]
        logits = []
        for k in range(min(max_depth, total - root_t - 2)):
            if k:
                previous, cache = self.model(
                    previous, input_ids=ids[root_t + k + 1:root_t + k + 2][None],
                    past_key_values=cache, use_cache=True,
                )
            logits.append(self.model.lm_head(self.model.norm(previous))[0, 0])
        return logits

    @torch.no_grad()
    def eval_chain(self, features, ids, prompt_len: int, max_depth: int = 6) -> tuple[np.ndarray, dict]:
        """Return sparse cycle predictions [K,T] and exact metrics.score_chain.

        The caller supplies already trimmed own-greedy tokens, including EOS.
        Token IDs at non-visited starts remain -1 and are never scored. A final
        root-only cycle needs no draft inference. No full-depth tail is dropped.
        """
        features = _features(features, self.hidden_size, self.device, self.dtype)
        ids_gpu = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        ids_cpu = ids_gpu.detach().cpu().numpy()
        if not isinstance(prompt_len, (int, np.integer)) or not 1 <= prompt_len <= ids_gpu.numel():
            raise ValueError("invalid prompt_len")
        if max_depth < 1 or features.shape[0] != ids_gpu.numel():
            raise ValueError("invalid depth or mismatched feature/token lengths")
        predictions = np.full((max_depth, ids_gpu.numel()), -1, dtype=np.int64)
        t = int(prompt_len) - 1
        while t + 1 < ids_gpu.numel():
            accepted = 0
            # Teacher-forcing beyond the first rejection is used only as a
            # logit reference; those proposals are not recorded or accepted.
            for k, logits in enumerate(self.prefix_logits(features, ids_gpu, t, max_depth)):
                token = int(self.d2t_ids[logits.argmax()].item())
                predictions[k, t] = token
                if token != int(ids_cpu[t + k + 2]):
                    break
                accepted += 1
            t += accepted + 1
        metrics = _local_module("metrics")
        return predictions, metrics.score_chain(predictions, ids_cpu, int(prompt_len))


def make_training_draft(checkpoint_dir, base_embed_cpu, device="cuda"):
    """Vectorized packed-three-state adapter; the old zoo module is unmodified.

    FC remains trainable through zoo.trainable(['fc']) with an FP32 master copy.
    Forward weights are BF16, as in the author reference used for evaluation.
    A runtime parity gate is mandatory before any experimental training.
    """
    zoo = _local_module("vendor.zoo")
    # Clone function globals instead of monkeypatching zoo._rms process-wide.
    # Existing unrelated research users of vendor.zoo therefore stay unchanged.
    globals_unroll = {**zoo.Draft.unroll.__globals__, "_rms": _native_rms}
    native_unroll = types.FunctionType(
        zoo.Draft.unroll.__code__, globals_unroll, zoo.Draft.unroll.__name__, zoo.Draft.unroll.__defaults__,
        zoo.Draft.unroll.__closure__,
    )

    class PackedDraft(zoo.Draft):
        unroll = native_unroll

        def fuse(self, features):
            x = _features(features, self.H, self.W["fc"].device, torch.bfloat16)
            return x @ self.W["fc"].to(torch.bfloat16).T

    embedding = torch.as_tensor(base_embed_cpu).detach().clone().to(device=device, dtype=torch.bfloat16)
    draft = PackedDraft(checkpoint_dir, n_target_layers=32, target_embed=embedding,
                        device=device, layer_ids=[2, 16, 29], name="official-packed")
    # Always use base embeddings even when the checkpoint includes an embedding.
    draft.embed = embedding
    if draft.norm_output or draft.norm_before_fc or draft.norm_before_residual or draft.fc_norms is not None:
        raise ValueError("pilot accepts the official unnormalized FC architecture only")
    if draft.W["q_norm"] is not None or draft.W["k_norm"] is not None:
        raise ValueError("pilot does not accept draft Q/K norms")
    return draft


class AuthorTrainingDraft:
    """Differentiable selected-prefix calls to the UNMODIFIED author's model.

    A vectorized legacy training path showed larger BF16 logit differences on
    real weights. The experiment therefore uses the author path for BOTH CE and
    primary evaluation. Functional FC replacement retains an FP32 master and
    native BF16 arithmetic, without mutating the module during optimization.
    """
    def __init__(self, native):
        self.native, self.model = native, native.model
        self.W = {'fc': torch.nn.Parameter(native.get_fc().to(native.device))}
        self.embed = self.model.embed_tokens.weight
        self.in_vocab = torch.zeros(len(self.embed), dtype=torch.bool, device=native.device)
        self.in_vocab[native.d2t_ids] = True

    def trainable(self, groups):
        if groups != ['fc']:
            raise ValueError('The author training adapter only permits fc')
        return [self.W['fc']]

    def unroll_selected(self, features, ids, root_positions, depth=6):
        flat = _features(features, self.native.hidden_size, self.native.device, self.native.dtype)
        ids = torch.as_tensor(ids, device=self.native.device, dtype=torch.long)
        rows = [[] for _ in range(depth)]
        parameters = {'fc.weight': self.W['fc'].to(self.native.dtype)}
        for t in root_positions:
            t = int(t)
            self.model.reset()
            hidden, cache = torch.func.functional_call(
                self.model, parameters, (flat[:t+1][None],),
                {'input_ids':ids[1:t+2][None], 'use_cache':True}, strict=False,
            )
            previous = hidden[:, -1:]
            for k in range(depth):
                if t+k+2 >= len(ids):
                    rows[k].append(torch.zeros(len(self.native.d2t_ids), device=ids.device, dtype=self.native.dtype))
                    continue
                if k:
                    previous, cache = torch.func.functional_call(
                        self.model, parameters, (previous,),
                        {'input_ids':ids[t+k+1:t+k+2][None], 'past_key_values':cache, 'use_cache':True},
                        strict=False,
                    )
                rows[k].append(self.model.lm_head(self.model.norm(previous))[0,0])
        return [torch.stack(values) for values in rows]


@torch.no_grad()
def check_unroll_parity(native: NativeDraft, training_draft, features, ids, prompt_lens=(3, 7),
                        max_depth=3, *, atol=0.02, rtol=0.02) -> dict:
    """Fail closed on arithmetic or argmax disagreement; return diagnostics.

    Tests each available depth at the requested prefix lengths against the real
    author KV cache. Call for original AND a perturbed FC; after adaptation this
    can be called again using the final FC before evaluating experimental rows.
    """
    features = _features(features, native.hidden_size, native.device, native.dtype)
    packed = features.view(features.shape[0], 3, native.hidden_size)
    ids = torch.as_tensor(ids, device=native.device, dtype=torch.long)
    selected_path = hasattr(training_draft, 'unroll_selected')
    roots = [int(p)-1 for p in prompt_lens]
    output = (training_draft.unroll_selected(packed, ids, roots, depth=max_depth)
              if selected_path else training_draft.unroll(packed, ids, depth=max_depth))
    max_abs, max_rel, comparisons, wrong = 0.0, 0.0, 0, 0
    failures = []
    for prompt_len in prompt_lens:
        t = int(prompt_len) - 1
        for k, reference in enumerate(native.prefix_logits(features, ids, t, max_depth)):
            actual = output[k][roots.index(t) if selected_path else t]
            error = (actual.float() - reference.float()).abs()
            max_abs = max(max_abs, float(error.max()))
            max_rel = max(max_rel, float((error / reference.float().abs().clamp_min(1e-6)).max()))
            comparisons += 1
            same_argmax = bool(actual.argmax() == reference.argmax())
            wrong += int(not same_argmax)
            if not torch.allclose(actual.float(), reference.float(), atol=atol, rtol=rtol) or not same_argmax:
                failures.append({"prompt_len": int(prompt_len), "depth": k + 1,
                                 "max_abs": float(error.max()), "same_argmax": same_argmax})
    report = {"passed": not failures, "comparisons": comparisons, "max_abs_logit_error": max_abs,
              "max_rel_logit_error": max_rel, "argmax_disagreements": wrong,
              "atol": atol, "rtol": rtol, "failures": failures,
              "training_path": "author_native_functional" if selected_path else "legacy_vectorized"}
    if comparisons == 0 or failures:
        raise AssertionError("Author/training-path parity failed: " + json.dumps(report))
    return report
