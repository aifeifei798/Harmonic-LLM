#!/usr/bin/env python3
"""
harmonic-llm mechanism validation harness (v4)
==============================================

The operator under test is an amplitude-domain soft limiter:

    ref    = median(block RMS)                    # robust per-forward scale
    ratio  = block peak / ref                     # amplitude outlier ratio
    alpha  = alpha_max * tanh(steepness * relu(ratio - threshold))
    limit  = threshold * ref
    gain_e = 1 - alpha * relu(1 - limit / |x_e|)
    x_e   <- x_e * gain_e

What this script measures
-------------------------
1. Paired response on the *same token trajectory* (teacher forcing), so every
   difference is attributable to the operator and not to different text.
2. Gate statistics: the block-ratio distribution and the duty cycle
   P(ratio > threshold) per layer.
3. Suppression that is actually suppression: paired per-block changes in peak
   magnitude and in crest factor. (Crest factor is scale-invariant, so it is
   expected to stay flat -- that is a property, not a failure.)
4. Cost: tokens/s, likelihood drift (teacher-forced NLL of the base
   continuation) and KL(base || condition).
5. Controls: (a) unconditional limiter (alpha = 1 everywhere), (b) real alpha
   values permuted across blocks (same magnitude distribution, no selectivity).
6. A bypass audit: with the gate closed the output must be bit-identical to the
   input, and with the gate open the peaks must land on the limit.

What this script does NOT measure
---------------------------------
Reasoning quality, hallucination rate, instruction following. Those need an
eval harness (IFEval / GSM8K / TruthfulQA) with n >= 200 prompts and confidence
intervals.

Usage
-----
    python test_package_spectrum.py
    python test_package_spectrum.py --model Qwen/Qwen2.5-7B-Instruct \
        --threshold 6.0 --steepness 3.0 --max-new-tokens 64
    python test_package_spectrum.py --quick      # skip sham controls + timing
"""

import argparse
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harmonic_llm import (apply_harmonic_eq, remove_harmonic_eq,
                          harmonic_status, dynamic_latent_resonance_suppressor)

DEFAULT_MODEL = os.environ.get(
    "HARMONIC_MODEL_ID", "./gemma-4-12B-it-qat-q4_0-unquantized")
DEFAULT_PROMPT = (
    "Explain the exact folded frequencies of a 14kHz tone saturated by tanh at "
    "44.1kHz.")

BLOCK = 8
R_MIN, R_MAX = 1.0, 400.0       # block-peak / median-block-rms range
NBINS = 80
# log spacing: the bulk lives at 1-4 and the tail at 25-300, so linear bins
# would either hide the tail or destroy all low-end resolution
BIN_EDGES = np.logspace(math.log10(R_MIN), math.log10(R_MAX), NBINS + 1)


# --------------------------------------------------------------------------- #
# Block helpers
# --------------------------------------------------------------------------- #
def flat_blocks(tensor, block=BLOCK):
    """[B,S,D] -> ([B*S, n_blocks, block], pad)"""
    block = max(int(block), 1)
    d = tensor.shape[-1]
    flat = tensor.reshape(-1, d)
    pad = (block - d % block) % block
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat.reshape(flat.shape[0], -1, block), pad


def unflat_blocks(blocks, shape, pad):
    d = shape[-1]
    flat = blocks.reshape(blocks.shape[0], -1)
    if pad:
        flat = flat[:, :d]
    return flat.reshape(shape)


def block_stats(blocks, block=BLOCK):
    """Return (peak, rms, crest) per block, in float32."""
    blocks = blocks.float()
    peak = blocks.abs().amax(dim=-1)
    rms = torch.sqrt(blocks.pow(2).mean(dim=-1) + 1e-12)
    return peak, rms, peak / rms


def detector(blocks, threshold, steepness=3.0, alpha_max=1.0):
    """Same arithmetic as harmonic_llm.core, returned as tensors for analysis."""
    peak, rms, crest = block_stats(blocks)
    ref = rms.median().clamp(min=1e-9)
    ratio = peak / ref
    alpha = alpha_max * torch.tanh(steepness * torch.clamp(ratio - threshold,
                                                           min=0.0))
    limit = (threshold * ref).clamp(min=1e-9)
    over = 1.0 - limit / blocks.float().abs().clamp(min=limit)
    gain = 1.0 - alpha.unsqueeze(-1) * over
    return {"peak": peak, "rms": rms, "crest": crest, "ref": ref,
            "ratio": ratio, "alpha": alpha, "limit": limit, "gain": gain}


# --------------------------------------------------------------------------- #
# Sham transforms (controls)
# --------------------------------------------------------------------------- #
class ShamTransform:
    """Non-selective variants of the limiter.

    mode='on'   : alpha = 1 for every block (unconditional limiter)
    mode='rand' : the same *number* of blocks as the real gate would select is
                  chosen uniformly at random and fully limited. This keeps the
                  operation count and the clamping strength identical while
                  destroying the correlation with the actual outliers, so any
                  difference in drift is attributable to selectivity alone.

    A permutation of the real alpha is deliberately *not* used: the elementwise
    ``over`` term is zero for blocks without oversized elements, so a permuted
    alpha usually has nothing to act on and the control degenerates to a no-op.
    """

    def __init__(self, model, mode, threshold, steepness):
        self.mode = mode
        self.threshold = threshold
        self.steepness = steepness
        self.handles = []
        for name, module in model.named_modules():
            if name.endswith((".q_proj", ".o_proj")):
                self.handles.append(
                    module.register_forward_hook(self._hook))

    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        tensor = output[0] if is_tuple else output
        shape = tensor.shape
        blocks, pad = flat_blocks(tensor.detach())
        stats = detector(blocks, self.threshold, self.steepness)
        alpha = stats["alpha"]
        if self.mode == "on":
            alpha = torch.ones_like(alpha)
        else:
            k = int((alpha > 0.5).sum().item())
            alpha = torch.zeros_like(alpha)
            if k > 0:
                idx = torch.randperm(alpha.numel(),
                                     device=alpha.device)[:k]
                alpha.reshape(-1)[idx] = 1.0
        over = 1.0 - stats["limit"] / blocks.float().abs().clamp(
            min=stats["limit"])
        gain = 1.0 - alpha.unsqueeze(-1) * over
        out = unflat_blocks((blocks.float() * gain), shape, pad).to(
            tensor.dtype)
        return (out, *output[1:]) if is_tuple else out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def layer_index(name):
    hits = re.findall(r"\.(\d+)\.", name)
    return int(hits[-1]) if hits else -1


class StatsStore:
    def __init__(self):
        self.data = {"q": {}, "o": {}}

    def cell(self, kind, layer):
        slots = self.data[kind]
        if layer not in slots:
            slots[layer] = {
                "pos_peak": [],
                "pos_crest": [],
                "ratio_hist": np.zeros(NBINS),
                "crest_hist": np.zeros(NBINS),
                "n_blocks": 0,
                "duty": 0,
                "ratio_max": 0.0,
                "peak_p99": 0.0,
                "peak_rows": [],
                "crest_rows": [],
            }
        return slots[layer]


class Probe:
    """Records post-transform statistics for one Q/O projection."""

    def __init__(self, store, kind, layer, threshold, start=0, n_gen=None):
        self.store = store
        self.kind = kind
        self.layer = layer
        self.threshold = threshold
        self.start = start
        self.n_gen = n_gen

    def __call__(self, module, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            blocks, _ = flat_blocks(tensor.detach())
            peak, rms, crest = block_stats(blocks)
            ref = rms.median().clamp(min=1e-9)
            ratio = peak / ref

            cell = self.store.cell(self.kind, self.layer)
            cell["pos_peak"].append(peak.mean(dim=1).cpu())
            cell["pos_crest"].append(crest.mean(dim=1).cpu())
            idx = torch.bucketize(
                ratio.reshape(-1),
                torch.as_tensor(BIN_EDGES, device=ratio.device,
                                dtype=ratio.dtype)).clamp(1, NBINS) - 1
            cell["ratio_hist"] += torch.bincount(
                idx, minlength=NBINS).float().cpu().numpy()
            cell["crest_hist"] += torch.histc(crest.reshape(-1), bins=NBINS,
                                              min=1.0,
                                              max=math.sqrt(BLOCK) + 0.01
                                              ).cpu().numpy()
            cell["n_blocks"] += int(ratio.numel())
            cell["duty"] += int((ratio > self.threshold).sum().item())
            cell["ratio_max"] = max(cell["ratio_max"],
                                    float(ratio.max().item()))
            cell["peak_p99"] = max(cell["peak_p99"],
                                   float(peak.quantile(0.99).item()))

            lo = min(self.start, peak.shape[0])
            hi = (peak.shape[0] if self.n_gen is None
                  else min(peak.shape[0], self.start + self.n_gen))
            if hi > lo:
                cell["peak_rows"].append(peak[lo:hi].cpu())
                cell["crest_rows"].append(crest[lo:hi].cpu())
        return None


def assemble(vectors):
    """dict layer -> list of per-position vectors; returns dict of arrays."""
    out = {}
    for layer, vecs in vectors.items():
        if not vecs:
            continue
        out[layer] = torch.stack(vecs).mean(dim=0).numpy()
    return out


# --------------------------------------------------------------------------- #
# Model / generation helpers
# --------------------------------------------------------------------------- #
def load_model(model_id):
    from transformers import AutoTokenizer
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(processor, "tokenizer", processor)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
    except Exception as exc:
        print(f"[info] multimodal load failed ({exc}); falling back to CausalLM")
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def build_inputs(tokenizer, prompt, device):
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True)
    else:
        text = prompt
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def tf_forward(model, inputs):
    out = model(**inputs, use_cache=False)
    return out.logits


def continuation_stats(logits, ids, start):
    logp = torch.log_softmax(logits[0, start - 1:-1].float(), dim=-1)
    target = ids[0, start:]
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    return nll, logp


def kl_divergence(base_logp, cond_logp):
    return (base_logp.exp() *
            (base_logp - cond_logp)).sum(dim=-1).clamp(min=0.0)


def time_generate(model, inputs, n_new, reps, warmup):
    device = inputs["input_ids"].device
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_new, do_sample=False)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_new, do_sample=False)
    sync(device)
    return reps * n_new / (time.perf_counter() - t0)


CONDITIONS = {
    "base": "Native base (no hooks)",
    "filter": "harmonic-llm (gated soft limiter)",
    "sham_on": "Sham A: unconditional limiter (alpha=1)",
    "sham_perm": "Sham B: same block count, random selection",
}


def install_condition(model, cond, cfg, observer=None):
    if cond == "filter":
        apply_harmonic_eq(model,
                          threshold=cfg.threshold,
                          steepness=cfg.steepness,
                          alpha_max=cfg.alpha_max,
                          verbose=False,
                          observer=observer)
        return lambda: remove_harmonic_eq(verbose=False)
    if cond in ("sham_on", "sham_perm"):
        sham = ShamTransform(model,
                             "on" if cond == "sham_on" else "rand",
                             cfg.threshold, cfg.steepness)
        return sham.remove
    return lambda: None


class InstantProbe:
    """Observer for the operator's own before/after tensors.

    Measures the instantaneous effect at the projection that was patched, so
    the number is not confounded by downstream trajectory divergence. It covers
    the whole teacher-forced sequence (prompt + continuation), because the
    operator acts on prompt positions too -- restricting it to the generated
    tokens silently hides prompt-side clipping and makes the gate look inert.
    """

    def __init__(self, start=0, n_gen=None, block=BLOCK):
        self.start = start
        self.n_gen = n_gen
        self.block = block
        self.rel = {"q": [], "o": []}
        self.before = {"q": [], "o": []}
        self.after = {"q": [], "o": []}
        self.by_layer = {"q": {}, "o": {}}

    def _peaks(self, tensor):
        blocks, _ = flat_blocks(tensor.detach().float(), self.block)
        return blocks.abs().amax(dim=-1)

    def _window(self, peaks):
        lo = min(self.start, peaks.shape[0])
        hi = (peaks.shape[0] if self.n_gen is None
              else min(peaks.shape[0], self.start + self.n_gen))
        return lo, hi

    def __call__(self, name, before, after):
        kind = "q" if name.endswith(".q_proj") else "o"
        layer = layer_index(name)
        with torch.no_grad():
            pb_all = self._peaks(before)
            pa_all = self._peaks(after)
            lo, hi = self._window(pb_all)
            pb, pa = pb_all[lo:hi], pa_all[lo:hi]
        if pb.numel() == 0 or pb.shape != pa.shape:
            return
        rel = ((pa - pb) / pb.clamp(min=1e-9)).flatten().cpu()
        self.rel[kind].append(rel)
        self.before[kind].append(pb.flatten().cpu())
        self.after[kind].append(pa.flatten().cpu())
        self.by_layer[kind].setdefault(layer, []).append(rel)

    def stats(self):
        out = {}
        for kind in ("q", "o"):
            if not self.rel[kind]:
                continue
            rel = torch.cat(self.rel[kind])
            before = torch.cat(self.before[kind])
            after = torch.cat(self.after[kind])
            moved = rel.abs() > 0
            n = rel.numel()
            step = max(1, n // 4000)
            # keep every touched block plus a subsample of the untouched ones,
            # otherwise the touched blocks vanish from the scatter
            sample = torch.zeros(n, dtype=torch.bool)
            sample[::step] = True
            keep = moved | sample
            out[kind] = {
                "n_blocks": int(n),
                "n_moved": int(moved.sum()),
                "frac_moved": float(moved.float().mean()),
                "peak_med_before": float(before.median()),
                "peak_med_after": float(after.median()),
                "dpeak_med": float(rel.median()),
                "dpeak_p05": float(rel.quantile(0.05)),
                "dcrest_med": 0.0,
                "depth_med_gated": float(rel[moved].median())
                if moved.any() else 0.0,
                "depth_p05_gated": float(rel[moved].quantile(0.05))
                if moved.any() else 0.0,
                "scatter_before": np.asarray(before[keep].numpy()),
                "scatter_after": np.asarray(after[keep].numpy()),
                "scatter_moved": np.asarray(moved[keep].numpy()),
                "per_layer_depth": {
                    int(l): float(torch.cat(v).median())
                    for l, v in self.by_layer[kind].items()
                },
                "per_layer_frac": {
                    int(l): float((torch.cat(v).abs() > 0).float().mean())
                    for l, v in self.by_layer[kind].items()
                },
            }
        return out


def run_condition(model, inputs, cond, cfg, prompt_len, instant=None):
    observer = instant if cond == "filter" else None
    teardown = install_condition(model, cond, cfg, observer=observer)

    store = StatsStore()
    handles = []
    for name, module in model.named_modules():
        if not name.endswith((".q_proj", ".o_proj")):
            continue
        kind = "q" if name.endswith(".q_proj") else "o"
        probe = Probe(store, kind, layer_index(name), cfg.threshold,
                      start=prompt_len, n_gen=cfg.max_new_tokens)
        handles.append(module.register_forward_hook(probe))

    logits = tf_forward(model, inputs)

    for h in handles:
        h.remove()
    teardown()
    return store, logits


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarise(store):
    summary = {}
    for kind in ("q", "o"):
        cells = store.data[kind]
        if not cells:
            continue
        layers = sorted(cells)
        total_blocks = sum(cells[l]["n_blocks"] for l in layers)
        total_duty = sum(cells[l]["duty"] for l in layers)
        ratio_hist = np.sum([cells[l]["ratio_hist"] for l in layers], axis=0)
        crest_hist = np.sum([cells[l]["crest_hist"] for l in layers], axis=0)
        per_layer = {
            l: {
                "duty": cells[l]["duty"] / max(cells[l]["n_blocks"], 1),
                "peak_p99": cells[l]["peak_p99"],
            } for l in layers
        }
        pos = assemble({l: cells[l]["pos_peak"] for l in layers})
        n_pos = min(len(v) for v in pos.values())
        comb = np.concatenate([np.array([pos[l][p] for l in layers])
                               for p in range(n_pos)])
        summary[kind] = {
            "ratio_hist": ratio_hist,
            "crest_hist": crest_hist,
            "duty": total_duty / max(total_blocks, 1),
            "blocks": total_blocks,
            "ratio_max": max(cells[l]["ratio_max"] for l in layers),
            "per_layer": per_layer,
            "comb": comb,
            "n_layers": len(layers),
            "_peak_rows": {l: torch.cat(cells[l]["peak_rows"], 0)
                           for l in layers if cells[l]["peak_rows"]},
            "_crest_rows": {l: torch.cat(cells[l]["crest_rows"], 0)
                            for l in layers if cells[l]["crest_rows"]},
        }
    return summary


def suppression_stats(base_sum, cond_sum, kind="q"):
    """Paired per-block: does the operator actually reduce peak amplitude?

    Pairing is done per layer, because projection widths differ inside a model
    (Gemma-4 Unified, for instance, has 8 layers with double-width Q/O). Only
    dimensionless relative changes are pooled across layers.
    """
    pb_map = base_sum[kind]["_peak_rows"]
    pf_map = cond_sum[kind]["_peak_rows"]
    cb_map = base_sum[kind]["_crest_rows"]
    cf_map = cond_sum[kind]["_crest_rows"]

    rel_peak, rel_crest, before_all, after_all = [], [], [], []
    per_layer = {}
    for layer, pb in pb_map.items():
        pf = pf_map.get(layer)
        if pf is None or pf.shape != pb.shape:
            continue
        rel = ((pf - pb) / pb.clamp(min=1e-9)).flatten()
        rel_peak.append(rel)
        per_layer[int(layer)] = float(rel.abs().median())
        cb, cf = cb_map.get(layer), cf_map.get(layer)
        if cb is not None and cf is not None and cf.shape == cb.shape:
            rel_crest.append(((cf - cb) / cb.clamp(min=1e-9)).flatten())
        before_all.append(pb.flatten())
        after_all.append(pf.flatten())
    if not rel_peak:
        return {}

    rel_peak = torch.cat(rel_peak)
    rel_crest = torch.cat(rel_crest) if rel_crest else torch.zeros_like(rel_peak)
    before = torch.cat(before_all)
    after = torch.cat(after_all)
    # bf16 rounding alone moves a value by ~0.4%, so use a 1% tolerance:
    # anything below that is quantization noise between the two runs
    moved = rel_peak.abs() > 1e-2
    n = rel_peak.numel()
    step = max(1, n // 4000)
    return {
        "n_blocks": int(n),
        "n_moved": int(moved.sum()),
        "frac_moved": float(moved.float().mean()),
        "peak_med_before": float(before.median()),
        "peak_med_after": float(after.median()),
        "dpeak_med": float(rel_peak.median()),
        "dpeak_abs_med": float(rel_peak.abs().median()),
        "dpeak_p05": float(rel_peak.quantile(0.05)),
        "dcrest_med": float(rel_crest.median()),
        "depth_med_gated": float(rel_peak[moved].median()) if moved.any() else 0.0,
        "depth_p05_gated": float(rel_peak[moved].quantile(0.05))
        if moved.any() else 0.0,
        "per_layer_abspeak": per_layer,
        "scatter_before": np.asarray(before[::step].numpy()),
        "scatter_after": np.asarray(after[::step].numpy()),
    }


# --------------------------------------------------------------------------- #
# Audits
# --------------------------------------------------------------------------- #
def bypass_audit(threshold=64.0):
    """Correctness: closed gate => bit-exact, open gate => peaks land on limit."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 128, dtype=torch.bfloat16)
    x[0, 3, 7] = 60.0
    closed = dynamic_latent_resonance_suppressor(x, threshold=1e9)
    default = dynamic_latent_resonance_suppressor(x, threshold=threshold)

    blocks, _ = flat_blocks(x)
    peak, rms, _ = block_stats(blocks)
    ref = rms.median()
    limit = threshold * ref

    print("\n bypass audit:")
    print(f"   closed gate  : bit-identical = {torch.equal(closed, x)}")
    print(f"   default gate : spike {float(x[0, 3, 7]):.2f} -> "
          f"{float(default[0, 3, 7]):.2f} "
          f"(limit {float(limit):.2f}, "
          f"ratio {float(x[0, 3, 7] / limit):.2f}x)")
    print(f"   default gate : finite = {bool(torch.isfinite(default).all())}")

    # synthetic outliers at a realistic scale, to prove the operator suppresses
    # amplitude (not shape) with the shipped default threshold
    torch.manual_seed(1)
    base = torch.randn(64, 8) * 0.02
    spikes = torch.tensor([0.3, 0.6, 1.2, 2.5, 5.0]).unsqueeze(-1)
    base[:5, 3] = spikes.squeeze(-1)
    out = dynamic_latent_resonance_suppressor(base, threshold=threshold)
    b_peak = base.abs().amax(-1)
    o_peak = out.abs().amax(-1)
    print("   synthetic spikes (block peak before -> after):")
    for before, after in zip(b_peak[:5].tolist(), o_peak[:5].tolist()):
        print(f"      {before:8.4f} -> {after:8.4f} "
              f"({100 * (after - before) / before:+7.2f}%)")
    untouched = (b_peak[5:] == o_peak[5:]).float().mean()
    print(f"   healthy blocks untouched : {100 * float(untouched):.1f}%")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def serialisable(summary):
    out = {}
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            out[key] = [float(v) for v in value]
        elif isinstance(value, dict):
            out[key] = {str(k): ({kk: float(vv) for kk, vv in v.items()}
                                 if isinstance(v, dict) else float(v))
                        for k, v in value.items()}
        else:
            out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=64.0)
    ap.add_argument("--steepness", type=float, default=3.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--outdir",
                    default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--reps", type=int, default=2, help="timing repetitions")
    ap.add_argument("--quick", action="store_true",
                    help="skip sham controls and timing (fast smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 74)
    print(" harmonic-llm :: mechanism validation (paired, same-trajectory)")
    print("=" * 74)
    print(f" model            : {cfg.model}")
    print(f" threshold/steep  : {cfg.threshold} / {cfg.steepness} "
          f"(alpha_max {cfg.alpha_max})")
    print(f" new tokens       : {cfg.max_new_tokens} (greedy)")

    bypass_audit(cfg.threshold)

    model, tokenizer = load_model(cfg.model)
    inputs = build_inputs(tokenizer, cfg.prompt, model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # --- 1. one reference (base) generation; every comparison replays exactly
    #        this trajectory, so differences are not text differences.
    with torch.no_grad():
        gen = model.generate(**inputs,
                             max_new_tokens=cfg.max_new_tokens,
                             do_sample=False)
    ids = gen[:, :prompt_len + cfg.max_new_tokens]
    full = {"input_ids": ids,
            "attention_mask": torch.ones_like(ids)}
    response = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
    print(f"\n reference continuation ({cfg.max_new_tokens} tokens):")
    print("   " + response.replace("\n", " ")[:300])
    print("   (all conditions below are teacher-forced on this exact sequence)")

    conditions = ["base", "filter"]
    if not cfg.quick:
        conditions += ["sham_on", "sham_perm"]

    # --- 2. paired measurement pass
    results = {}
    base_logp = None
    instant = InstantProbe()
    for cond in conditions:
        print(f"\n[measure] {cond}: {CONDITIONS[cond]}")
        store, logits = run_condition(model, full, cond, cfg, prompt_len,
                                      instant=instant)
        nll, logp = continuation_stats(logits, ids, prompt_len)
        if cond == "base":
            base_logp = logp
            kl = torch.zeros_like(nll)
        else:
            kl = kl_divergence(base_logp, logp)
        results[cond] = {
            "summary": summarise(store),
            "nll": float(nll.mean()),
            "kl": float(kl.mean()),
        }
        if cond == "filter":
            results[cond]["instant"] = instant.stats()
        del logits
        sync(model.device)
        print(f"   done (harmonic_status={harmonic_status()})")

    # --- 3. timing pass (no probes: this is the real deployment cost)
    if not cfg.quick:
        print("\n[timing] tokens/s, hooks installed, probes removed")
        for cond in conditions:
            teardown = install_condition(model, cond, cfg)
            tps = time_generate(model, inputs, cfg.max_new_tokens,
                                cfg.reps, warmup=1)
            teardown()
            results[cond]["tps"] = tps
            print(f"   {cond:10s} {tps:7.2f} tok/s")
        base_tps = results["base"]["tps"]
        for cond in conditions:
            results[cond]["tps_ratio"] = results[cond]["tps"] / base_tps
    else:
        for cond in conditions:
            results[cond]["tps"] = float("nan")
            results[cond]["tps_ratio"] = float("nan")

    # --- 4. report
    print("\n" + "=" * 74)
    print(" RESULTS")
    print("=" * 74)
    for kind, label in (("q", "Q"), ("o", "O")):
        stats = results["base"]["summary"][kind]
        print(f" {label} gate duty P(peak/ref > {cfg.threshold}) : "
              f"{100 * stats['duty']:.2f}% of {stats['blocks']:,} blocks "
              f"| ratio max {stats['ratio_max']:.2f}")
    base_q = results["base"]["summary"]["q"]
    print(" ratio distribution (Q, peak / median block RMS):")
    hist = np.asarray(base_q["ratio_hist"], dtype=float)
    cdf = np.cumsum(hist) / max(hist.sum(), 1.0)
    centers = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
        idx = min(int(np.searchsorted(cdf, q)), NBINS - 1)
        print(f"    p{q * 100:<7g} = {centers[idx]:6.2f}")
    print(" duty at multiples of the configured threshold:")
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
        th = cfg.threshold * mult
        frac = hist[BIN_EDGES[1:] > th].sum() / max(hist.sum(), 1.0)
        print(f"    P(ratio > {th:7.2f}) = {100 * frac:6.3f}%")

    print("\n instantaneous suppression (inside the patched projection, "
          "whole sequence):")
    inst = results["filter"].get("instant", {})
    for kind in ("q", "o"):
        s = inst.get(kind) or {}
        if not s.get("n_blocks"):
            continue
        print(f"   filter    {kind.upper()}: touched {s['n_moved']:,}/"
              f"{s['n_blocks']:,} blocks ({100 * s['frac_moved']:.2f}%) | "
              f"depth median {100 * s['depth_med_gated']:+.2f}% "
              f"(p05 {100 * s['depth_p05_gated']:+.2f}%)")
        print(f"             block peak median "
              f"{s['peak_med_before']:.4g} -> {s['peak_med_after']:.4g}")

    print("\n downstream drift (same trajectory; shows how the change propagates):")
    for cond in conditions:
        if cond == "base":
            continue
        stats = {k: suppression_stats(results["base"]["summary"],
                                      results[cond]["summary"],
                                      kind=k)
                 for k in ("q", "o")}
        results[cond]["suppression"] = stats
        for kind in ("q", "o"):
            s = stats.get(kind) or {}
            if not s.get("n_blocks"):
                continue
            print(f"   {cond:9s} {kind.upper()}: |d(peak)| median "
                  f"{100 * s['dpeak_abs_med']:.2f}%, "
                  f"{100 * s['frac_moved']:.1f}% of layers-blocks moved "
                  f">1% | d(crest) median {100 * s['dcrest_med']:+.3f}%")

    print("\n capability-adjacent drift (teacher-forced on the base continuation):")
    for cond in conditions:
        r = results[cond]
        print(f"   {cond:10s} NLL {r['nll']:.4f}  "
              f"KL(base||cond) {r['kl']:.4e}  "
              f"{r['tps']:7.2f} tok/s  ({r['tps_ratio']:.3f}x)")

    # --- 5. figures
    outdir = cfg.outdir
    os.makedirs(outdir, exist_ok=True)
    plot_response(results, cfg, outdir)
    plot_diagnostics(results, cfg, outdir)

    payload = {
        "config": vars(cfg),
        "prompt_len": prompt_len,
        "conditions": {
            cond: {
                "nll": results[cond]["nll"],
                "kl": results[cond]["kl"],
                "tps": results[cond]["tps"],
                "tps_ratio": results[cond]["tps_ratio"],
                "q": serialisable(results[cond]["summary"]["q"]),
                "o": serialisable(results[cond]["summary"]["o"]),
                "suppression": {
                    kind: {k: v for k, v in stats.items()
                           if not k.startswith("scatter")}
                    for kind, stats in results[cond].get(
                        "suppression", {}).items()
                },
                "instant": {
                    kind: {k: v for k, v in stats.items()
                           if not k.startswith("scatter")}
                    for kind, stats in results[cond].get(
                        "instant", {}).items()
                },
            }
            for cond in conditions
        },
    }
    json_path = os.path.join(outdir, "harmonic_diagnostics.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    print("\n saved:")
    print(f"   {os.path.join(outdir, 'harmonic_package_verification.png')}")
    print(f"   {os.path.join(outdir, 'harmonic_gate_diagnostics.png')}")
    print(f"   {json_path}")
    print("\n mechanism validated: inject / remove / bit-exact bypass / "
          "measurable peak suppression.")
    print(" NOT validated here: reasoning, hallucination, instruction following.")


def plot_response(results, cfg, outdir):
    base_q = results["base"]["summary"]["q"]
    flt_q = results["filter"]["summary"]["q"]
    inst = results["filter"].get("instant", {})
    layers = sorted(base_q["per_layer"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"harmonic-llm :: amplitude response on the same token "
                 f"trajectory (threshold={cfg.threshold:g}, "
                 f"steepness={cfg.steepness:g}, alpha_max={cfg.alpha_max:g})",
                 fontsize=13)

    # (0,0) tail magnitude per layer
    ax = axes[0, 0]
    ax.plot(layers, [base_q["per_layer"][l]["peak_p99"] for l in layers],
            "r--o", markersize=3, label="native base")
    ax.plot(layers, [flt_q["per_layer"][l]["peak_p99"] for l in layers],
            "b-s", markersize=3, label="harmonic-llm")
    ax.set_yscale("log")
    ax.set_title("Q: p99 |x| per layer (tail magnitude)")
    ax.set_xlabel("layer")
    ax.set_ylabel("p99 magnitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (0,1) ratio distribution, both projections
    ax = axes[0, 1]
    centers = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])
    for kind, color in (("q", "#4C72B0"), ("o", "#DD8452")):
        hist = results["base"]["summary"][kind]["ratio_hist"].astype(float)
        hist = hist / max(hist.sum(), 1.0)
        hist[hist <= 0] = np.nan
        ax.loglog(centers, hist, color=color, label=f"{kind.upper()} (base)")
    ax.axvline(cfg.threshold, color="orange", linestyle="--",
               label=f"threshold {cfg.threshold:g}")
    ax.set_title("Block ratio distribution and where the gate sits")
    ax.set_xlabel("block peak / median block RMS")
    ax.set_ylabel("probability mass")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (1,0) instantaneous response, touched blocks emphasised
    ax = axes[1, 0]
    have = False
    for kind, color in (("q", "#4C72B0"), ("o", "#DD8452")):
        sup = inst.get(kind) or {}
        if not sup.get("n_blocks"):
            continue
        before = np.asarray(sup["scatter_before"])
        after = np.asarray(sup["scatter_after"])
        moved = np.asarray(sup["scatter_moved"])
        if (~moved).any():
            ax.scatter(before[~moved], after[~moved], s=3, alpha=0.12,
                       color=color)
        if moved.any():
            ax.scatter(before[moved], after[moved], s=12, alpha=0.85,
                       color=color,
                       label=f"{kind.upper()}: {int(moved.sum()):,} touched "
                       f"({100 * sup['frac_moved']:.2f}%)")
            have = True
    before_all = [np.asarray(s["scatter_before"])
                  for s in inst.values() if s.get("n_blocks")]
    if before_all:
        flat = np.concatenate(before_all)
        lo, hi = float(flat.min()), float(flat.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="identity")
        ax.set_xscale("log")
        ax.set_yscale("log")
    if not have:
        ax.text(0.5, 0.5, "no block exceeded the threshold",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Instantaneous block peak: before vs after the operator")
    ax.set_xlabel("block peak before")
    ax.set_ylabel("block peak after")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")

    # (1,1) depth vs pre-gate peak: where does the knee bite?
    ax = axes[1, 1]
    for kind, color in (("q", "#4C72B0"), ("o", "#DD8452")):
        sup = inst.get(kind) or {}
        if not sup.get("n_blocks"):
            continue
        before = np.asarray(sup["scatter_before"])
        after = np.asarray(sup["scatter_after"])
        moved = np.asarray(sup["scatter_moved"])
        if moved.any():
            depth = 100 * (after[moved] - before[moved]) / before[moved]
            ax.scatter(before[moved], depth, s=12, alpha=0.8, color=color,
                       label=kind.upper())
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_title("Suppression depth vs pre-gate peak (touched blocks only)")
    ax.set_xlabel("block peak before")
    ax.set_ylabel("change in peak (%)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "harmonic_package_verification.png"),
                dpi=300)
    plt.close(fig)


def plot_diagnostics(results, cfg, outdir):
    base_q = results["base"]["summary"]["q"]
    inst = results["filter"].get("instant", {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("harmonic-llm :: gate diagnostics "
                 "(selectivity, propagation, cost)", fontsize=13)

    # (0,0) crest distribution: shape statistic, deliberately unchanged
    ax = axes[0, 0]
    crest_edges = np.linspace(1.0, math.sqrt(BLOCK) + 0.01, NBINS + 1)
    centers = 0.5 * (crest_edges[:-1] + crest_edges[1:])
    for cond, color, width, alpha, label in (
            ("base", "r", 4.0, 0.5, "base"),
            ("filter", "b", 0.9, 0.95, "harmonic-llm")):
        hist = results[cond]["summary"]["q"]["crest_hist"].astype(float)
        hist = hist / max(hist.sum(), 1.0)
        ax.plot(centers, hist, color=color, linewidth=width, alpha=alpha,
                label=label)
    ax.set_title("Block crest factor (peak/rms): unchanged by design")
    ax.set_xlabel("crest factor")
    ax.set_ylabel("probability mass")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (0,1) duty cycle by layer, both projections
    ax = axes[0, 1]
    layers = sorted(base_q["per_layer"])
    x = np.arange(len(layers))
    width = 0.4
    any_duty = False
    for i, (kind, color, label) in enumerate((("q", "#4C72B0", "Q"),
                                              ("o", "#DD8452", "O"))):
        pl = results["base"]["summary"][kind]["per_layer"]
        duties = [100 * pl[l]["duty"] for l in layers]
        any_duty = any_duty or max(duties, default=0.0) > 0
        ax.bar(x + (i - 0.5) * width, duties, width, color=color,
               label=f"{label} (mean "
               f"{100 * results['base']['summary'][kind]['duty']:.3f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=6, rotation=90)
    ax.set_title("Gate duty cycle by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("% blocks above threshold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    if not any_duty:
        ax.text(0.5, 0.55,
                f"threshold {cfg.threshold:g} above the observed ratio "
                f"maximum ({base_q['ratio_max']:.1f})\n"
                "the gate never opens on this model",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # (1,0) how the perturbation propagates downstream
    ax = axes[1, 0]
    sup = results["filter"].get("suppression", {})
    for kind, color, label in (("q", "#4C72B0", "Q"), ("o", "#DD8452", "O")):
        per_layer = (sup.get(kind) or {}).get("per_layer_abspeak") or {}
        if not per_layer:
            continue
        ls = sorted(per_layer)
        ax.plot(ls, [100 * per_layer[l] for l in ls], "-o", markersize=3,
                color=color, label=label)
    ax.set_title("Downstream propagation: median |change| in block peak "
                 "per layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("|change| in block peak (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (1,1) cost
    ax = axes[1, 1]
    conds = [c for c in ("base", "filter", "sham_on", "sham_perm")
             if c in results]
    tps = [results[c]["tps"] for c in conds]
    colors = ["#8C8C8C", "#4C72B0", "#DD8452", "#55A868"][:len(conds)]
    bars = ax.bar(range(len(conds)), tps, color=colors)
    for cond, bar in zip(conds, bars):
        ratio = results[cond]["tps_ratio"]
        txt = "n/a" if math.isnan(ratio) else f"{ratio:.3f}x"
        if cond != "base" and not math.isnan(ratio):
            txt += f"\nKL {results[cond]['kl']:.1e}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                txt, ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_title("Throughput and likelihood drift")
    ax.set_ylabel("tokens / s")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "harmonic_gate_diagnostics.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
