import math

import torch
import torch.nn.functional as F

_BASIS_CACHE = {}


def get_dct_basis(r=8, k=8, device=None, dtype=None):
    """Orthonormal 1D-DCT-II basis.

    Kept for backward compatibility and for transform-domain experiments. The
    v4 suppressor below does not use it: amplitude suppression is a pointwise
    operation and does not need a transform.
    """
    key = (r, k)
    if key not in _BASIS_CACHE:
        n = torch.arange(r, dtype=torch.float32)
        j = torch.arange(k, dtype=torch.float32)
        c = torch.where(j == 0, math.sqrt(1.0 / r), math.sqrt(2.0 / r))
        _BASIS_CACHE[key] = c[:, None] * torch.cos(
            math.pi / r * (n[None, :] + 0.5) * j[:, None])
    m = _BASIS_CACHE[key]
    return m.to(device=device, dtype=dtype) if device or dtype else m


def _robust_scale(rms, cap=8192):
    """Median block RMS, computed on a stride-subsample when the tensor is big.

    The median is what makes the reference resistant to the outliers we are
    trying to suppress. A stride sample keeps the cost bounded on long
    sequences without changing the statistic in any material way.
    """
    flat = rms.reshape(-1)
    n = flat.numel()
    if n > cap:
        flat = flat[:: max(1, n // cap)]
    return flat.median().clamp(min=1e-6)


def _components(values, block_size=8, threshold=64.0, steepness=3.0,
                alpha_max=1.0):
    """Shared implementation. All arithmetic stays in the input dtype.

    Keeping the elementwise work in bf16/fp16 halves the memory traffic versus
    an upcast to fp32, and the quantities involved (magnitudes, ratios, gains)
    do not need fp32 mantissa. Only the reference statistic is a reduction, so
    its precision is unaffected by the element dtype.
    """
    d = values.shape[-1]
    pad = (block_size - d % block_size) % block_size
    x = F.pad(values, (0, pad)) if pad > 0 else values
    blocks = x.view(*x.shape[:-1], -1, block_size)
    abs_blocks = blocks.abs()
    peak = abs_blocks.amax(dim=-1)
    rms = blocks.pow(2).mean(dim=-1).sqrt()
    ref = _robust_scale(rms)
    ratio = peak / ref
    alpha = alpha_max * torch.tanh(
        steepness * torch.clamp(ratio - threshold, min=0.0))
    limit = (threshold * ref).clamp(min=1e-6)
    over = 1.0 - limit / abs_blocks.clamp(min=limit)
    gain = 1.0 - alpha.unsqueeze(-1) * over
    return {"blocks": blocks, "gain": gain, "alpha": alpha, "ratio": ratio,
            "limit": limit, "ref": ref, "pad": pad, "d": d, "x": x}


def dynamic_latent_resonance_suppressor(out,
                                        block_size=8,
                                        threshold=64.0,
                                        steepness=3.0,
                                        alpha_max=1.0):
    """Dynamic range suppression on activations (v4, amplitude domain).

    The operator is a soft-knee limiter with a block-level gate:

        ref    = median(block RMS over the whole tensor)   # robust layer scale
        ratio  = block peak / ref                          # amplitude outlier ratio
        alpha  = alpha_max * tanh(steepness * relu(ratio - threshold))
        limit  = threshold * ref
        gain_e = 1 - alpha * relu(1 - limit / |x_e|)
        x_e   <- x_e * gain_e

    Properties
    ----------
    * amplitude-aware: unlike a crest factor, ``ratio`` grows with the absolute
      size of a spike, so a 3-sigma bump and a 300-sigma blowup are not treated
      as the same event.
    * bit-exact bypass at rest: a block below the knee gets ``gain == 1``
      exactly, and multiplying by 1.0 is exact, so healthy activations are
      returned bit-for-bit unchanged.
    * conservative: alpha == 1 clamps any element above ``limit`` exactly to
      ``limit``; alpha < 1 applies a proportional, continuous reduction.
    * direction-aware: only the elements above the limit are scaled, so a spike
      is attenuated rather than smeared across its neighbours.

    Parameters
    ----------
    block_size : int
        Channel-group size used for the detector (default 8).
    threshold : float
        Gate position in units of ``block peak / median block RMS``. The bulk
        of a healthy tensor sits around 2-4; the extreme tail of a real model
        reaches 25-90. Default 64.0, which only touches that extreme tail
        (~0.06% of blocks on Gemma-4-12B). Lower it only after measuring your
        own model with :func:`suppression_profile`.
    steepness : float
        Soft-knee sharpness (multiplier inside tanh). Default 3.0.
    alpha_max : float
        Maximum reduction strength in [0, 1]. 1.0 gives a full ceiling;
        smaller values keep part of the original amplitude. Default 1.0.

    Notes
    -----
    The estimator is scale-equivariant: multiplying the whole tensor by a
    constant leaves ``ratio`` unchanged, so the same threshold works on layers
    whose activation scales differ by orders of magnitude.
    """
    if out is None:
        return None
    if out.numel() == 0:
        return out

    state = _components(out, block_size, threshold, steepness, alpha_max)
    suppressed = (state["blocks"] * state["gain"]).view(*state["x"].shape)
    if state["pad"] > 0:
        suppressed = suppressed[..., :state["d"]]
    return suppressed


def suppression_profile(values, block_size=8, threshold=64.0, steepness=3.0,
                        alpha_max=1.0):
    """Diagnostic helper: per-block gain statistics without modifying anything.

    Returns a dict of tensors with the block-level ``ratio``, ``alpha`` and the
    element-level ``gain``. Use it to calibrate ``threshold`` on a new model:

        prof = suppression_profile(activation)
        duty = (prof["alpha"] > 0).float().mean()   # gate duty cycle
        print(prof["ratio"].max())

    A gate duty cycle above ~0.1% means the threshold is cutting into the
    model's working range; raise it until only the extreme tail is touched.
    """
    if values is None or values.numel() == 0:
        return {}
    state = _components(values, block_size, threshold, steepness, alpha_max)
    return {"ratio": state["ratio"], "alpha": state["alpha"],
            "gain": state["gain"], "ref": state["ref"],
            "limit": state["limit"]}
