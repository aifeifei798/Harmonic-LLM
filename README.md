<div align="center">

# Harmonic-LLM

### Training-free amplitude outlier suppression for Transformer Q/O projections

[![PyPI version](https://img.shields.io/pypi/v/harmonic-llm.svg?color=blue)](https://pypi.org/project/harmonic-llm/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

</div>

---

## What this is

A forward hook that caps extreme activation outliers at the `q_proj` and `o_proj`
outputs of any Hugging Face Transformer. No weight changes, no training, no
calibration data. It is a soft-knee limiter: it measures how far each channel
block rises above the layer's typical scale, and scales down only the elements
that exceed a high ceiling.

The package is deliberately small and its claims are deliberately narrow. It
does **not** claim to improve reasoning, reduce hallucination, or make a model
faster. It claims one thing — *it measurably reduces the magnitude of the
extreme tail, and here is exactly what that costs* — and it ships the harness
that produced every number in this README.

```python
from harmonic_llm import apply_harmonic_eq, remove_harmonic_eq

apply_harmonic_eq(model)          # threshold=64.0, steepness=3.0, alpha_max=1.0
...                               # run the model as usual
remove_harmonic_eq()              # stock model restored exactly
```

---

## Mechanism

For an activation tensor `x` reshaped into blocks of 8 channels:

```
ref    = median(block RMS)                      # robust per-tensor scale
ratio  = block_peak / ref                       # amplitude outlier ratio
alpha  = alpha_max * tanh(steepness * relu(ratio - threshold))
limit  = threshold * ref
gain_e = 1 - alpha * relu(1 - limit / |x_e|)
x_e   <- x_e * gain_e
```

Three properties matter:

* **Amplitude-aware.** `ratio` grows with the absolute size of a spike, so a
  3-sigma bump and a 300-sigma blowup are not treated as the same event. A crest
  factor (`peak/rms`) cannot do this: it is scale-invariant, so multiplying a
  block by any constant leaves it unchanged. That is why this operator measures
  magnitude, not shape.
* **Bit-exact at rest.** A block below the knee gets `gain == 1` exactly, and
  multiplying by 1.0 is exact. Healthy activations come back bit-for-bit
  unchanged, including in bf16.
* **Direction-aware.** Only elements above the ceiling are scaled, so a spike is
  attenuated rather than smeared across its neighbours.

The estimator is scale-equivariant: multiplying the whole tensor by a constant
leaves `ratio` unchanged, so one threshold works across layers whose activation
scales differ by orders of magnitude.

---

## Measured results

Model: `gemma-4-12B-it-qat-q4_0-unquantized` (48 layers, hidden 3840, 8 layers
with double-width Q/O), greedy continuation of 64 tokens, all conditions
teacher-forced on the **same token trajectory**, so every difference is
attributable to the operator rather than to different generated text.

### Where the outliers are

`ratio = block peak / median block RMS`, over the whole sequence:

| quantile | Q projection | O projection |
| :--- | ---: | ---: |
| p50 | 1.9 | 1.9 |
| p90 | 4.0 | 5.0 |
| p99 | 17.9 | 35.1 |
| p99.9 | 28.0 | 59.2 |
| max | 95.6 | 286.5 |

The distribution is heavy-tailed but not sparse in the extreme: the p99.99 is
already 36. A threshold in the 2–8 range (the previous default was 2.2) cuts
into activations the model actively uses.

### What the shipped default does

Shipped default: `threshold=64.0`, `steepness=3.0`, `alpha_max=1.0`.

| measurement | Q | O |
| :--- | ---: | ---: |
| blocks touched | 30 / 2,867,200 (0.001%) | 1,718 / 2,304,000 (0.07%) |
| peak reduction, median | −21.2% | −15.8% |
| peak reduction, 5th pct | −31.3% | −57.9% |
| crest factor change | 0.000% | 0.000% |

Crest factor is unchanged **by construction** (it is scale-invariant). That is
not a failure of the operator; it is the reason the operator had to be
amplitude-based in the first place.

Cost, measured on the same run:

| condition | NLL | KL(base‖cond) | tokens/s |
| :--- | ---: | ---: | ---: |
| base | 0.173 | 0 | 21.0 |
| harmonic-llm | 0.293 | 0.092 | 15.0 (**0.71×**) |
| sham: unconditional limiter | 0.290 | 0.092 | 12.9 |
| sham: same block count, random blocks | 0.176 | 0.0015 | 11.2 |

Two things this table says that are easy to miss:

* The unconditional limiter costs the same as the gated one (KL 0.092 vs 0.092).
  At this operating point the gate is nearly binary, so the *ceiling* does the
  work, not the block selection. The random-block sham, which selects the same
  number of blocks without regard to their content, costs 60× less — that is
  the selectivity signal, and it is the opposite direction from what the
  "surgical gate" story would predict.
* The pass is not free: 0.71× throughput at batch 1, because it is a pointwise
  traversal of every Q/O output.

### The trade-off frontier (this is the real result)

Sweep on the same model, `block` reference, one teacher-forced forward each.
"depth" is the median peak reduction on touched blocks:

| threshold | alpha_max | blocks touched | depth | KL | ΔNLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.0 | 3.9% | −40.5% | 14.0 | +14.1 |
| 16 | 1.0 | 1.8% | −33.3% | 3.57 | +3.40 |
| 32 | 1.0 | 0.5% | −20.8% | 0.59 | +0.57 |
| 64 | 1.0 | 0.04% | −16.0% | 0.078 | +0.11 |
| 32 | 0.25 | 0.5% | −5.4% | 0.015 | +0.02 |

A second sweep asked whether the damage comes from *changing the shape* of a
block (element-wise clamping) or from *reducing its amplitude* (one gain per
block). At equal peak reduction the two are indistinguishable:

| config | clamp KL | block-gain KL |
| :--- | ---: | ---: |
| threshold 16, alpha 1.0 | 3.565 | 3.477 |
| threshold 32, alpha 1.0 | 0.588 | 0.548 |
| threshold 64, alpha 1.0 | 0.078 | 0.079 |

The honest reading: these outliers are load-bearing. Any operator that reduces
their amplitude pays roughly in proportion to how much it reduces, and
preserving the block shape does not buy anything back.

---

## Figures

Both figures come from the same 12B run and are regenerated by
`test_package_spectrum.py`.

<p align="center">
  <img src="https://raw.githubusercontent.com/aifeifei798/Harmonic-LLM/main/doc/harmonic_package_verification.png" alt="amplitude response" width="100%">
</p>

*Top-left*: the tail magnitude (`p99 |x|`) per layer, base vs filtered. The two
curves coincide except where the operator acts.
*Top-right*: the block-ratio distribution for Q and O, with the shipped
threshold. Q decays before 64; O keeps a tail out to ~300, which is exactly the
part the operator removes.
*Bottom-left*: every touched block, before vs after. Points below the dashed
identity line are attenuated; 30 Q blocks (0.001%) and 1,718 O blocks (0.07%).
*Bottom-right*: depth as a function of the pre-gate peak — the knee is visible,
and the deeper the original spike, the larger the reduction.

<p align="center">
  <img src="https://raw.githubusercontent.com/aifeifei798/Harmonic-LLM/main/doc/harmonic_gate_diagnostics.png" alt="gate diagnostics" width="100%">
</p>

*Top-left*: the crest-factor distribution is unchanged, by construction.
*Top-right*: gate duty cycle per layer; the outliers concentrate in the middle
layers, peaking around layer 20.
*Bottom-left*: how a 0.07% local change propagates — the median absolute change
in block peak per layer reaches ~36% at layer 18, because later layers see a
different input. This is the part a local measurement would miss.
*Bottom-right*: throughput and likelihood drift, including the two sham
controls.

---

## What this does not do

* It does not measure or improve reasoning, instruction following, or
  hallucination rate. Those need an eval harness (IFEval / GSM8K / TruthfulQA)
  with n ≥ 200 prompts and confidence intervals. The harness reports mechanism,
  not capability.
* It is not "free" or "0.00 dB at rest for all tokens". At rest (gate closed for
  every block) the output is bit-exact; once the gate opens, the run diverges
  and later layers see different activations. The downstream propagation is
  visible in the diagnostics figure and is much larger than the local change
  (median |Δpeak| of 2.8–3.4% across layers from 0.07% of touched blocks).
* The default is calibrated on one model family. On a model whose ratio
  distribution tops out below 64, the operator is a bit-exact no-op — which is
  the correct behaviour, but also means the default does nothing there.

---

## Install

```bash
pip install harmonic-llm
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. Transformers is only needed for the
verification script, not for the hook itself.

---

## Calibrating on your own model

`suppression_profile` shows what the detector would do, without modifying the
model:

```python
import torch
from harmonic_llm import suppression_profile

prof = suppression_profile(activation)          # same shape as the projection output
duty = (prof["alpha"] > 0).float().mean()       # fraction of blocks gated
print("ratio max:", prof["ratio"].max().item(), "duty:", duty.item())
```

Rules of thumb from the measurements above:

* duty above ~0.1% means the threshold is cutting into the working range;
* pick the threshold above the p99.9 of `prof["ratio"]`;
* if you need a gentler operator, lower `alpha_max` before lowering
  `threshold` — it scales the depth without widening the affected set.

The full harness is in `test_package_spectrum.py`:

```bash
python test_package_spectrum.py \
  --model /path/to/model \
  --max-new-tokens 64 \
  --threshold 64.0 --steepness 3.0 --alpha-max 1.0
```

It writes `harmonic_package_verification.png`, `harmonic_gate_diagnostics.png`
and `harmonic_diagnostics.json`: paired same-trajectory response, gate duty by
layer, instantaneous peak reduction, downstream propagation, two sham controls,
and tokens/s.

---

## API

```python
from harmonic_llm import (
    apply_harmonic_eq,       # attach hooks to q_proj / o_proj
    remove_harmonic_eq,      # detach every hook, restore the stock model
    harmonic_status,         # number of active hooks
    dynamic_latent_resonance_suppressor,  # the operator itself
    suppression_profile,     # diagnostic: ratio / alpha / gain, no mutation
)
```

`apply_harmonic_eq` accepts `threshold`, `steepness`, `alpha_max`,
`block_size`, `targets` (defaults to `("q_proj", "o_proj")`) and `verbose`.
K and V are intentionally left untouched.

---

## Reproducing the numbers

```bash
python test_package_spectrum.py \
  --model gemma-4-12B-it-qat-q4_0-unquantized \
  --max-new-tokens 64 --reps 4
```

The script prints the exact tokens/s, NLL, KL and block-level statistics used
above, and writes them to JSON so the tables can be regenerated rather than
copied by hand.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
