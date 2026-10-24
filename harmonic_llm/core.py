import math
import torch
import torch.nn.functional as F

_BASIS_CACHE = {}


def get_dct_basis(r=8, k=8, device=None, dtype=None):
    key = (r, k)
    if key not in _BASIS_CACHE:
        n = torch.arange(r, dtype=torch.float32)
        j = torch.arange(k, dtype=torch.float32)
        c = torch.where(j == 0, math.sqrt(1.0 / r), math.sqrt(2.0 / r))
        _BASIS_CACHE[key] = c[:, None] * torch.cos(
            math.pi / r * (n[None, :] + 0.5) * j[:, None])
    m = _BASIS_CACHE[key]
    return m.to(device=device, dtype=dtype) if device or dtype else m


def dynamic_latent_resonance_suppressor(out,
                                        block_size=8,
                                        threshold=2.2,
                                        steepness=3.0):
    """
    v3 动态自适应谐振抑制算子 (Dynamic Latent Resonance Suppressor)
    - 侧链检波: 波峰因数 (Peak / RMS)
    - 动态门控: <= 2.2 时 100% 物理透明直通 (0.00 dB)
    - 动态下压: > 2.2 时动态拉出负增益，启动 DCT 削峰
    """
    if out is None:
        return None

    D = out.shape[-1]
    pad = (block_size - (D % block_size)) % block_size
    out_pad = F.pad(out, (0, pad)) if pad > 0 else out
    n_blocks = out_pad.shape[-1] // block_size

    # 切 8 通道宏块
    blocks = out_pad.view(*out_pad.shape[:-1], n_blocks, block_size)

    # 侧链波峰因数检波
    peak = blocks.abs().amax(dim=-1, keepdim=True)
    rms = torch.sqrt(torch.mean(blocks**2, dim=-1, keepdim=True) + 1e-8)
    crest_factor = peak / (rms + 1e-8)  # (..., n_blocks, 1)

    # 动态下压阻尼系数
    overshoot = F.relu(crest_factor - threshold)
    alpha_dyn = 0.90 * torch.tanh(steepness * overshoot)

    # 正交 DCT 变换与重构
    basis = get_dct_basis(block_size,
                          block_size,
                          device=out.device,
                          dtype=out.dtype)
    coef = torch.matmul(blocks, basis.t())

    # 动态量化削峰 (STE)
    amax = coef.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    coef_q = (torch.round(coef.float() / scale.float()).clamp(-127, 127) *
              scale.float()).to(out.dtype)
    rec_blocks = torch.matmul(coef_q, basis)

    # 宏块原生自动广播插值
    blended_blocks = (1.0 - alpha_dyn) * blocks + alpha_dyn * rec_blocks
    blended = blended_blocks.view(*out_pad.shape)

    return blended[..., :D] if pad > 0 else blended
