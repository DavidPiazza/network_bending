"""
Shared audio utility helpers for the network_bending audio nodes.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _to_scalar_int(value: Any) -> int:
    """
    Convert the provided value into a Python int.

    Supports plain numbers as well as torch or numpy scalars.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() != 1:
            raise ValueError("Sample rate tensor must contain exactly one value")
        value = value.item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Sample rate array must contain exactly one value")
        value = value.item()
    return int(value)


def unpack_audio_input(audio: Any, default_sample_rate: int = 44100) -> Tuple[torch.Tensor, int]:
    """
    Normalize different AUDIO payload formats into a (waveform, sample_rate) tuple.

    ComfyUI audio payloads are commonly either:
      * A tuple/list: (waveform, sample_rate)
      * A mapping: {"waveform": waveform, "sample_rate": sample_rate, ...}

    This helper accepts both forms (and nests thereof) and always returns a torch.Tensor
    waveform along with an integer sample rate. A sensible default sample rate is used
    if one is not provided in the payload.
    """
    if isinstance(audio, dict):
        waveform = audio.get("waveform")
        if waveform is None and "samples" in audio:
            waveform = audio["samples"]
        if waveform is None and "audio" in audio:
            waveform = audio["audio"]

        sample_rate = (
            audio.get("sample_rate")
            or audio.get("samplerate")
            or audio.get("sr")
            or default_sample_rate
        )
    elif isinstance(audio, (list, tuple)):
        if len(audio) == 0:
            raise ValueError("Audio input is empty")
        if len(audio) == 1:
            return unpack_audio_input(audio[0], default_sample_rate=default_sample_rate)

        waveform = audio[0]
        sample_rate = audio[1] if len(audio) > 1 else default_sample_rate
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    if waveform is None:
        raise ValueError("Audio input is missing waveform data")

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    sample_rate = _to_scalar_int(sample_rate)
    return waveform, sample_rate


def unpack_latent(latent: Any) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Return the tensor inside a latent payload and any accompanying metadata.

    The audio nodes historically passed raw tensors around. To interoperate with
    other ComfyUI latent nodes (which expect a dict containing a ``"samples"``
    tensor), we normalize inputs here. If the latent is already a dict we return
    the contained tensor along with the original dict so callers can preserve
    metadata when re-wrapping.
    """

    if isinstance(latent, dict) and "samples" in latent:
        return latent["samples"], latent
    return torch.as_tensor(latent), None


def wrap_latent(samples: torch.Tensor, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wrap a latent tensor in a Comfy-style dict, preserving existing fields."""

    latent_dict: Dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    latent_dict["samples"] = samples
    return latent_dict


def get_vae_device(vae: Any) -> torch.device:
    """Best-effort device resolver for ComfyUI VAEs and torch modules."""

    # ComfyUI VAE wrapper exposes .device
    device = getattr(vae, "device", None)
    if device is not None:
        return torch.device(device)

    # Underlying torch module is usually on the correct device
    first_stage = getattr(vae, "first_stage_model", None)
    if hasattr(first_stage, "parameters"):
        try:
            return next(first_stage.parameters()).device
        except StopIteration:
            pass

    if hasattr(vae, "parameters"):
        try:
            return next(vae.parameters()).device
        except StopIteration:
            pass

    return torch.device("cpu")
