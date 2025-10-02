"""
Audio conditioning nodes for Stable Audio in ComfyUI.

This subpackage may have optional dependencies (e.g., torchaudio, librosa).
If those are not installed, we gracefully disable audio nodes rather than
failing the entire custom node pack.
"""

from typing import Dict

try:
    from .audio_latent_nodes import (  # type: ignore
        AudioVAEEncode,
        AudioVAEDecode,
        AudioLatentInterpolate,
        AudioLatentBlend,
        AudioFeatureExtractor,
        AudioLatentManipulator,
    )

    from .audio_style_transfer import (  # type: ignore
        AudioStyleTransfer,
        AudioLatentGuidance,
        AudioReferenceEncoder,
    )

    NODE_CLASS_MAPPINGS: Dict[str, object] = {
        "AudioVAEEncode": AudioVAEEncode,
        "AudioVAEDecode": AudioVAEDecode,
        "AudioLatentInterpolate": AudioLatentInterpolate,
        "AudioLatentBlend": AudioLatentBlend,
        "AudioFeatureExtractor": AudioFeatureExtractor,
        "AudioLatentManipulator": AudioLatentManipulator,
        "AudioStyleTransfer": AudioStyleTransfer,
        "AudioLatentGuidance": AudioLatentGuidance,
        "AudioReferenceEncoder": AudioReferenceEncoder,
    }

    NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
        "AudioVAEEncode": "Audio VAE Encode",
        "AudioVAEDecode": "Audio VAE Decode",
        "AudioLatentInterpolate": "Audio Latent Interpolate",
        "AudioLatentBlend": "Audio Latent Blend",
        "AudioFeatureExtractor": "Audio Feature Extractor",
        "AudioLatentManipulator": "Audio Latent Manipulator",
        "AudioStyleTransfer": "Audio Style Transfer",
        "AudioLatentGuidance": "Audio Latent Guidance",
        "AudioReferenceEncoder": "Audio Reference Encoder",
    }

except Exception as _audio_import_error:  # pragma: no cover
    # Dependencies for audio nodes are missing; disable audio nodes gracefully
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]