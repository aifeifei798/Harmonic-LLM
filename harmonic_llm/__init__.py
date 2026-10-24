import json

from .core import dynamic_latent_resonance_suppressor, suppression_profile
from .patch import apply_harmonic_eq, remove_harmonic_eq, harmonic_status

__version__ = "0.4.0"
__all__ = [
    "apply_harmonic_eq",
    "remove_harmonic_eq",
    "harmonic_status",
    "dynamic_latent_resonance_suppressor",
    "suppression_profile",
]
