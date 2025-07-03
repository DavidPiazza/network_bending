"""
Audio conditioning nodes for Stable Audio in ComfyUI
"""

from .audio_latent_nodes import (
    AudioVAEEncode,
    AudioVAEDecode,
    AudioLatentInterpolate,
    AudioLatentBlend,
    AudioFeatureExtractor,
    AudioLatentManipulator,
)

from .audio_style_transfer import (
    AudioStyleTransfer,
    AudioLatentGuidance,
    AudioReferenceEncoder
)

NODE_CLASS_MAPPINGS = {
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

NODE_DISPLAY_NAME_MAPPINGS = {
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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']