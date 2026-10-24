import torch.nn as nn

from .core import dynamic_latent_resonance_suppressor

# global handle registry so the whole patch can be removed in one call
_ACTIVE_HOOK_HANDLES = []


def make_hook(block_size=8, threshold=64.0, steepness=3.0, alpha_max=1.0,
              observer=None, name=None):
    """Build a forward hook that applies the suppressor.

    ``observer``, when given, is called as ``observer(name, before, after)``
    with the pre- and post-transform tensors. It is meant for diagnostics
    (measuring the instantaneous effect of the operator) and must not modify
    anything.
    """

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        tensor = output[0] if is_tuple else output
        filtered = dynamic_latent_resonance_suppressor(
            tensor,
            block_size=block_size,
            threshold=threshold,
            steepness=steepness,
            alpha_max=alpha_max)
        if observer is not None:
            observer(name, tensor, filtered)
        if is_tuple:
            return (filtered, *output[1:])
        return filtered

    return hook


def apply_harmonic_eq(model: nn.Module,
                      threshold: float = 64.0,
                      steepness: float = 3.0,
                      block_size: int = 8,
                      alpha_max: float = 1.0,
                      targets: tuple = ("q_proj", "o_proj"),
                      verbose: bool = True,
                      observer=None):
    """Attach dynamic outlier suppression to the Q/O projection outputs.

    Works on any Hugging Face style Transformer (LLaMA, Qwen, Gemma, Mistral,
    DeepSeek) without changing weights. Call it again to replace the current
    configuration; call :func:`remove_harmonic_eq` to restore the stock model.

    Parameters
    ----------
    threshold : float
        Gate position in ``block peak / median block RMS`` units (default 64).
        This is deliberately far in the tail: on Gemma-4-12B the ratio
        distribution has p99.9 ~= 25 and max ~= 85, so a threshold of 64 only
        touches the extreme tail (~0.04% of blocks). Measure your own model
        with :func:`harmonic_llm.suppression_profile` before lowering it.
    steepness : float
        Soft-knee sharpness (default 3.0).
    block_size : int
        Channel-group size for the detector (default 8).
    alpha_max : float
        Maximum reduction strength in [0, 1] (default 1.0 = full ceiling).
    targets : tuple[str, ...]
        Module-name suffixes to patch. Defaults to query and output
        projections; K/V are intentionally left alone.
    verbose : bool
        Print a one-line summary (default True).
    observer : callable or None
        Optional ``observer(name, before, after)`` diagnostic hook.
    """
    global _ACTIVE_HOOK_HANDLES
    if _ACTIVE_HOOK_HANDLES:
        remove_harmonic_eq(verbose=False)

    injected = 0
    for name, module in model.named_modules():
        if not any(name.endswith(f".{t}") for t in targets):
            continue
        hook = make_hook(block_size=block_size,
                         threshold=threshold,
                         steepness=steepness,
                         alpha_max=alpha_max,
                         observer=observer,
                         name=name)
        _ACTIVE_HOOK_HANDLES.append(module.register_forward_hook(hook))
        injected += 1

    if verbose:
        print(f"[harmonic-llm] patched {injected} Q/O projections "
              f"(threshold={threshold}, steepness={steepness}, "
              f"alpha_max={alpha_max})")
    return model


def remove_harmonic_eq(model: nn.Module = None, verbose: bool = True):
    """Detach every hook and restore the stock model exactly."""
    global _ACTIVE_HOOK_HANDLES
    count = len(_ACTIVE_HOOK_HANDLES)
    for handle in _ACTIVE_HOOK_HANDLES:
        handle.remove()
    _ACTIVE_HOOK_HANDLES.clear()
    if verbose:
        print(f"[harmonic-llm] removed {count} suppression hooks")
    return model


def harmonic_status():
    """Number of currently active suppression hooks."""
    return len(_ACTIVE_HOOK_HANDLES)
