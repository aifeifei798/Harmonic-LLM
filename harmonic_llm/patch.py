import torch.nn as nn
from .core import dynamic_latent_resonance_suppressor

# 全局 Hook 句柄容器，方便一键拔出
_ACTIVE_HOOK_HANDLES = []


def make_hook(block_size=8, threshold=2.2):

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (dynamic_latent_resonance_suppressor(output[0],
                                                        block_size=block_size,
                                                        threshold=threshold),
                    *output[1:])
        return dynamic_latent_resonance_suppressor(output,
                                                   block_size=block_size,
                                                   threshold=threshold)

    return hook


def apply_harmonic_eq(model: nn.Module,
                      threshold: float = 2.2,
                      block_size: int = 8,
                      targets: tuple = ("q_proj", "o_proj")):
    """
    一键为模型注意力机制注入 v3 动态声学均衡器
    自动适配 LLaMA, Gemma, Qwen, Mistral, DeepSeek 等主流架构
    """
    global _ACTIVE_HOOK_HANDLES
    if len(_ACTIVE_HOOK_HANDLES) > 0:
        print("⚠️ 警告: 模型已被注入过 Harmonic EQ，正在重置现有通道...")
        remove_harmonic_eq(model)

    hook_fn = make_hook(block_size=block_size, threshold=threshold)
    injected_count = 0

    for name, module in model.named_modules():
        # 严格限定只对注意力机制的 Q 寻址和 O 输出通道挂载
        if any(name.endswith(f".{t}") for t in targets):
            handle = module.register_forward_hook(hook_fn)
            _ACTIVE_HOOK_HANDLES.append(handle)
            injected_count += 1

    print(
        f"✨ [Harmonic-LLM] 成功为 {injected_count} 个注意力 Q/O 通道注入动态均衡器！(门限 Crest={threshold})"
    )
    return model


def remove_harmonic_eq(model: nn.Module = None):
    """一键拔除所有 Hook，100% 恢复原生模型状态"""
    global _ACTIVE_HOOK_HANDLES
    count = len(_ACTIVE_HOOK_HANDLES)
    for handle in _ACTIVE_HOOK_HANDLES:
        handle.remove()
    _ACTIVE_HOOK_HANDLES.clear()
    print(f"🧹 [Harmonic-LLM] 已安全移除所有 {count} 个动态均衡通道，模型已还原！")


def harmonic_status():
    """查看当前活跃的均衡器通道数量"""
    global _ACTIVE_HOOK_HANDLES
    return len(_ACTIVE_HOOK_HANDLES)


def make_hook(block_size=8, threshold=2.2, steepness=3.0):

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (dynamic_latent_resonance_suppressor(output[0],
                                                        block_size=block_size,
                                                        threshold=threshold,
                                                        steepness=steepness),
                    *output[1:])
        return dynamic_latent_resonance_suppressor(output,
                                                   block_size=block_size,
                                                   threshold=threshold,
                                                   steepness=steepness)

    return hook


def apply_harmonic_eq(model: nn.Module,
                      threshold: float = 2.2,
                      steepness: float = 3.0,
                      block_size: int = 8,
                      targets: tuple = ("q_proj", "o_proj")):
    global _ACTIVE_HOOK_HANDLES
    if len(_ACTIVE_HOOK_HANDLES) > 0:
        remove_harmonic_eq(model)

    hook_fn = make_hook(block_size=block_size,
                        threshold=threshold,
                        steepness=steepness)
    injected_count = 0

    for name, module in model.named_modules():
        if any(name.endswith(f".{t}") for t in targets):
            handle = module.register_forward_hook(hook_fn)
            _ACTIVE_HOOK_HANDLES.append(handle)
            injected_count += 1

    print(
        f"✨ [Harmonic-LLM] 成功为 {injected_count} 个注意力 Q/O 通道注入动态均衡器！(门限 Crest={threshold}, 斜率={steepness})"
    )
    return model
