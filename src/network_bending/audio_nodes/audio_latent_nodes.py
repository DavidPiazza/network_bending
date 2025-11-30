"""
Audio Latent Space Manipulation Nodes for Stable Audio
These nodes enable working directly with audio latents for style transfer and hybrid generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torchaudio
import librosa

from ._audio_utils import get_vae_device, unpack_audio_input, unpack_latent, wrap_latent

# Import the additional nodes from the style transfer module
from .audio_style_transfer import (
    AudioStyleTransfer,
    AudioLatentGuidance,
    AudioReferenceEncoder
)

class AudioVAEEncode:
    """
    Encode audio waveforms into Stable Audio latent space
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio waveform to encode"}),
                "vae": ("VAE", {"tooltip": "Stable Audio VAE model"}),
            },
            "optional": {
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize audio before encoding"
                }),
                "target_sample_rate": ("INT", {
                    "default": 44100,
                    "min": 16000,
                    "max": 48000,
                    "tooltip": "Target sample rate for encoding"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT_INFO")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "encode_audio"
    CATEGORY = "audio/latent"
    
    def encode_audio(self, audio, vae, normalize=True, target_sample_rate=44100):
        # Extract audio tensor and sample rate (supports dict and tuple formats)
        waveform, sample_rate = unpack_audio_input(audio, default_sample_rate=target_sample_rate)
        waveform = waveform.to(torch.float32)
        
        # Resample if necessary
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        
        # Normalize audio
        if normalize:
            max_abs = torch.max(torch.abs(waveform))
            if float(max_abs) > 1e-12:
                waveform = waveform / max_abs
        
        # Ensure correct shape for VAE (batch, channels, samples)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        # Encode with VAE
        with torch.no_grad():
            # Move to same device as VAE
            device = get_vae_device(vae)
            waveform = waveform.to(device)
            
            # Encode
            latent = vae.encode(waveform)
            latent_tensor, latent_meta = unpack_latent(latent)
            
            # Get latent statistics
            latent_mean = latent_tensor.mean()
            latent_std = latent_tensor.std()
            
        # Create info dictionary
        info = {
            "original_shape": waveform.shape,
            "latent_shape": latent_tensor.shape,
            "sample_rate": target_sample_rate,
            "latent_mean": latent_mean.item(),
            "latent_std": latent_std.item(),
        }
        
        meta = dict(latent_meta) if latent_meta else {}
        meta.setdefault("sample_rate", target_sample_rate)
        meta.setdefault("audio_shape", waveform.shape)

        return (wrap_latent(latent_tensor, meta), info)

class AudioVAEDecode:
    """
    Decode audio latents back to waveforms
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Audio latent to decode"}),
                "vae": ("VAE", {"tooltip": "Stable Audio VAE model"}),
            },
            "optional": {
                "denormalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Denormalize output audio"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode_audio"
    CATEGORY = "audio/latent"
    
    def decode_audio(self, latent, vae, denormalize=True):
        # Decode with VAE
        with torch.no_grad():
            # Ensure latent is on correct device
            device = get_vae_device(vae)
            latent_tensor, latent_meta = unpack_latent(latent)
            latent_tensor = latent_tensor.to(device)
            
            # Decode
            waveform = vae.decode(latent_tensor)
            
            # Move to CPU for further processing
            waveform = waveform.cpu()
        
        # Denormalize if needed (clamp to valid range)
        if denormalize:
            waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # Extract sample rate (default to 44100 if not stored)
        sample_rate = 44100
        if isinstance(latent_meta, dict):
            sr = latent_meta.get("sample_rate") or latent_meta.get("audio_sample_rate")
            if sr is not None:
                if isinstance(sr, torch.Tensor):
                    sr = sr.detach().cpu().item()
                sample_rate = int(sr)
        
        # Return as (waveform, sample_rate) tuple
        return ((waveform, sample_rate),)

class AudioLatentInterpolate:
    """
    Interpolate between two audio latents for smooth transitions and hybrid sounds
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT", {"tooltip": "First audio latent"}),
                "latent_b": ("LATENT", {"tooltip": "Second audio latent"}),
                "interpolation_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Interpolation factor (0=A, 1=B)"
                }),
                "interpolation_mode": ([
                    "linear",
                    "spherical",
                    "cubic",
                    "sine"
                ], {
                    "default": "spherical",
                    "tooltip": "Interpolation method"
                }),
            },
            "optional": {
                "curve_power": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Power curve for non-linear interpolation"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "interpolate_latents"
    CATEGORY = "audio/latent"
    
    def interpolate_latents(self, latent_a, latent_b, interpolation_factor, 
                          interpolation_mode, curve_power=1.0):
        latent_a_tensor, latent_a_meta = unpack_latent(latent_a)
        latent_b_tensor, _ = unpack_latent(latent_b)

        # Ensure latents are on the same device
        device = latent_a_tensor.device
        latent_b_tensor = latent_b_tensor.to(device)
        
        # Apply curve to interpolation factor if needed
        if curve_power != 1.0:
            interpolation_factor = interpolation_factor ** curve_power
        
        if interpolation_mode == "linear":
            # Simple linear interpolation
            result = (1 - interpolation_factor) * latent_a_tensor + interpolation_factor * latent_b_tensor
            
        elif interpolation_mode == "spherical":
            # Spherical linear interpolation (SLERP)
            latent_a_flat = latent_a_tensor.flatten(1)
            latent_b_flat = latent_b_tensor.flatten(1)

            # Normalize latents per-batch element
            latent_a_unit = F.normalize(latent_a_flat, dim=1, eps=1e-6)
            latent_b_unit = F.normalize(latent_b_flat, dim=1, eps=1e-6)

            dot_product = torch.sum(latent_a_unit * latent_b_unit, dim=1, keepdim=True)
            dot_product = torch.clamp(dot_product, -0.999999, 0.999999)
            angles = torch.acos(dot_product)
            sin_angles = torch.sin(angles)

            denom = torch.where(sin_angles.abs() < 1e-6, torch.ones_like(sin_angles), sin_angles)
            coef_a = torch.sin((1 - interpolation_factor) * angles) / denom
            coef_b = torch.sin(interpolation_factor * angles) / denom

            result_flat = coef_a * latent_a_flat + coef_b * latent_b_flat

            # For nearly identical latents fall back to linear interpolation
            near_linear = sin_angles.abs() < 1e-6
            if near_linear.any():
                linear = (1 - interpolation_factor) * latent_a_flat + interpolation_factor * latent_b_flat
                result_flat = torch.where(
                    near_linear.expand_as(result_flat),
                    linear,
                    result_flat
                )

            result = result_flat.view_as(latent_a_tensor)
                        
        elif interpolation_mode == "cubic":
            # Cubic interpolation (smooth start and end)
            t = interpolation_factor
            t2 = t * t
            t3 = t2 * t
            factor = 3 * t2 - 2 * t3
            result = (1 - factor) * latent_a_tensor + factor * latent_b_tensor
            
        elif interpolation_mode == "sine":
            # Sine-based interpolation
            factor = 0.5 * (1 + math.sin(math.pi * (interpolation_factor - 0.5)))
            result = (1 - factor) * latent_a_tensor + factor * latent_b_tensor

        return (wrap_latent(result, latent_a_meta),)

class AudioLatentBlend:
    """
    Blend multiple audio latents with customizable weights
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT", {"tooltip": "First audio latent"}),
                "latent_b": ("LATENT", {"tooltip": "Second audio latent"}),
                "blend_mode": ([
                    "add",
                    "multiply", 
                    "screen",
                    "overlay",
                    "soft_light",
                    "difference",
                    "exclusion"
                ], {
                    "default": "add",
                    "tooltip": "Blending mode"
                }),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Blend strength"
                }),
            },
            "optional": {
                "latent_c": ("LATENT", {"tooltip": "Optional third audio latent"}),
                "latent_d": ("LATENT", {"tooltip": "Optional fourth audio latent"}),
                "weight_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_c": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_d": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "blend_latents"
    CATEGORY = "audio/latent"
    
    def blend_latents(self, latent_a, latent_b, blend_mode, blend_factor,
                     latent_c=None, latent_d=None,
                     weight_a=1.0, weight_b=1.0, weight_c=1.0, weight_d=1.0,
                     normalize=True):
        latent_a_tensor, latent_a_meta = unpack_latent(latent_a)
        latent_b_tensor, _ = unpack_latent(latent_b)

        device = latent_a_tensor.device
        
        # Collect all available latents and weights
        latents = [latent_a_tensor.to(device), latent_b_tensor.to(device)]
        weights = [weight_a, weight_b]
        
        if latent_c is not None:
            latent_c_tensor, _ = unpack_latent(latent_c)
            latents.append(latent_c_tensor.to(device))
            weights.append(weight_c)
        if latent_d is not None:
            latent_d_tensor, _ = unpack_latent(latent_d)
            latents.append(latent_d_tensor.to(device))
            weights.append(weight_d)
        
        # Normalize weights if requested by the user
        if normalize:
            total_weight = sum(weights)
            if abs(total_weight) > 1e-12:
                weights = [w / total_weight for w in weights]
            else:
                # Fallback to uniform weights when total is zero
                count = len(weights)
                weights = [1.0 / count for _ in weights]
        
        # Apply blend mode
        if blend_mode == "add":
            # Weighted addition
            result = sum(latent * weight for latent, weight in zip(latents, weights))
            result = result * blend_factor + latents[0] * (1 - blend_factor)
            
        elif blend_mode == "multiply":
            # Multiplicative blending
            result = latents[0]
            for i in range(1, len(latents)):
                blend = latents[i] * weights[i]
                result = result * (1 + blend_factor * (blend - 1))
                
        elif blend_mode == "screen":
            # Screen blending (inverse multiply)
            result = latents[0]
            for i in range(1, len(latents)):
                blend = 1 - (1 - result) * (1 - latents[i] * weights[i])
                result = result * (1 - blend_factor) + blend * blend_factor
                
        elif blend_mode == "overlay":
            # Overlay blending
            result = latents[0]
            for i in range(1, len(latents)):
                mask = (result < 0.5).float()
                blend = mask * (2 * result * latents[i] * weights[i]) + \
                       (1 - mask) * (1 - 2 * (1 - result) * (1 - latents[i] * weights[i]))
                result = result * (1 - blend_factor) + blend * blend_factor
                
        elif blend_mode == "soft_light":
            # Soft light blending
            result = latents[0]
            for i in range(1, len(latents)):
                blend = latents[i] * weights[i]
                soft = (1 - 2 * blend) * result * result + 2 * blend * result
                result = result * (1 - blend_factor) + soft * blend_factor
                
        elif blend_mode == "difference":
            # Difference blending
            result = latents[0]
            for i in range(1, len(latents)):
                diff = torch.abs(result - latents[i] * weights[i])
                result = result * (1 - blend_factor) + diff * blend_factor
                
        elif blend_mode == "exclusion":
            # Exclusion blending
            result = latents[0]
            for i in range(1, len(latents)):
                excl = result + latents[i] * weights[i] - 2 * result * latents[i] * weights[i]
                result = result * (1 - blend_factor) + excl * blend_factor

        return (wrap_latent(result, latent_a_meta),)

class AudioFeatureExtractor:
    """
    Extract various audio features for latent manipulation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to analyze"}),
                "feature_type": ([
                    "spectral_centroid",
                    "spectral_rolloff",
                    "zero_crossing_rate",
                    "mfcc",
                    "spectral_bandwidth",
                    "rms_energy",
                    "tempo",
                    "onset_strength"
                ], {
                    "default": "spectral_centroid",
                    "tooltip": "Type of audio feature to extract"
                }),
            },
            "optional": {
                "n_mfcc": ("INT", {
                    "default": 13,
                    "min": 1,
                    "max": 40,
                    "tooltip": "Number of MFCC coefficients"
                }),
                "hop_length": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "tooltip": "Hop length for feature extraction"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO_FEATURES",)
    RETURN_NAMES = ("features",)
    FUNCTION = "extract_features"
    CATEGORY = "audio/analysis"
    
    def extract_features(self, audio, feature_type, n_mfcc=13, hop_length=512):
        waveform, sample_rate = unpack_audio_input(audio)
        waveform = waveform.to(torch.float32)
        
        # Convert to numpy for librosa
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.detach().cpu().numpy()
        else:
            waveform_np = np.asarray(waveform)

        # Librosa expects numpy arrays with float dtype
        waveform_np = np.squeeze(waveform_np)
        if waveform_np.ndim == 0:
            waveform_np = waveform_np.reshape(1)
        if not isinstance(waveform_np, np.ndarray):
            waveform_np = np.asarray(waveform_np)
        if not np.issubdtype(waveform_np.dtype, np.floating):
            waveform_np = waveform_np.astype(np.float32, copy=False)
        
        # Extract features based on type
        if feature_type == "spectral_centroid":
            features = librosa.feature.spectral_centroid(
                y=waveform_np, sr=sample_rate, hop_length=hop_length
            )
        elif feature_type == "spectral_rolloff":
            features = librosa.feature.spectral_rolloff(
                y=waveform_np, sr=sample_rate, hop_length=hop_length
            )
        elif feature_type == "zero_crossing_rate":
            features = librosa.feature.zero_crossing_rate(
                waveform_np, hop_length=hop_length
            )
        elif feature_type == "mfcc":
            features = librosa.feature.mfcc(
                y=waveform_np, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length
            )
        elif feature_type == "spectral_bandwidth":
            features = librosa.feature.spectral_bandwidth(
                y=waveform_np, sr=sample_rate, hop_length=hop_length
            )
        elif feature_type == "rms_energy":
            features = librosa.feature.rms(
                y=waveform_np, hop_length=hop_length
            )
        elif feature_type == "tempo":
            tempo, beats = librosa.beat.beat_track(
                y=waveform_np, sr=sample_rate, hop_length=hop_length
            )
            features = np.array([[tempo]])
        elif feature_type == "onset_strength":
            features = librosa.onset.onset_strength(
                y=waveform_np, sr=sample_rate, hop_length=hop_length
            ).reshape(1, -1)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()
        
        # Create feature dictionary
        feature_dict = {
            "type": feature_type,
            "data": features_tensor,
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "shape": features_tensor.shape
        }
        
        return (feature_dict,)

class AudioLatentManipulator:
    """
    Manipulate audio latents based on extracted features or manual controls
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Audio latent to manipulate"}),
                "manipulation_type": ([
                    "frequency_shift",
                    "temporal_stretch",
                    "harmonic_emphasis",
                    "noise_injection",
                    "dynamic_range",
                    "spatial_transform",
                    "resonance_filter"
                ], {
                    "default": "frequency_shift",
                    "tooltip": "Type of manipulation to apply"
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Manipulation strength"
                }),
            },
            "optional": {
                "features": ("AUDIO_FEATURES", {"tooltip": "Optional features to guide manipulation"}),
                "frequency": ("FLOAT", {
                    "default": 440.0,
                    "min": 20.0,
                    "max": 20000.0,
                    "tooltip": "Target frequency for frequency-based operations"
                }),
                "time_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 4.0,
                    "tooltip": "Time stretch factor"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "manipulate_latent"
    CATEGORY = "audio/latent"
    
    def manipulate_latent(self, latent, manipulation_type, strength, 
                         features=None, frequency=440.0, time_factor=1.0):
        latent_tensor, latent_meta = unpack_latent(latent)
        device = latent_tensor.device
        result = latent_tensor.clone()
        
        # Get latent dimensions
        batch, channels, *spatial_dims = result.shape
        
        if manipulation_type == "frequency_shift":
            # Shift frequency content in latent space
            # Apply FFT-like transformation in latent space
            if len(spatial_dims) == 2:
                # 2D latent (height, width)
                height, width = spatial_dims
                fft_latent = torch.fft.fft2(result)
                shift_time = int(round(strength * width * frequency / 22050))
                shift_freq = int(round(strength * height * 0.25))
                if strength > 0:
                    if shift_time == 0 and width > 1:
                        shift_time = 1
                    if shift_freq == 0 and height > 1:
                        shift_freq = 1
                fft_latent = torch.roll(fft_latent, shifts=(shift_freq, shift_time), dims=(-2, -1))
                result = torch.fft.ifft2(fft_latent).real
            else:
                # 1D latent
                fft_latent = torch.fft.fft(result)
                shift_amount = int(round(strength * spatial_dims[0] * frequency / 22050))
                if strength > 0 and shift_amount == 0 and spatial_dims[0] > 1:
                    shift_amount = 1
                fft_latent = torch.roll(fft_latent, shifts=shift_amount, dims=-1)
                result = torch.fft.ifft(fft_latent).real
                
        elif manipulation_type == "temporal_stretch":
            # Stretch or compress temporal dimension
            if len(spatial_dims) >= 1:
                time_dim = spatial_dims[-1]
                new_time_dim = int(time_dim * time_factor)
                
                # Interpolate along time dimension
                if len(spatial_dims) == 2:
                    result = F.interpolate(result, size=(spatial_dims[0], new_time_dim), 
                                         mode='bilinear', align_corners=False)
                else:
                    result = F.interpolate(result, size=new_time_dim, 
                                         mode='linear', align_corners=False)
                    
                # Adjust back to original size with padding or cropping
                if new_time_dim > time_dim:
                    result = result[..., :time_dim]
                elif new_time_dim < time_dim:
                    pad_size = time_dim - new_time_dim
                    result = F.pad(result, (0, pad_size))
                    
        elif manipulation_type == "harmonic_emphasis":
            # Emphasize harmonic content
            # Create harmonic mask in frequency domain
            if len(spatial_dims) >= 1:
                fft_latent = torch.fft.fft(result, dim=-1)
                magnitudes = torch.abs(fft_latent)
                
                # Find peaks (harmonics)
                kernel_size = 5
                maxpool = nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size//2)
                if len(spatial_dims) == 2:
                    height, width = spatial_dims
                    mag_flat = magnitudes.reshape(batch * channels * height, 1, width)
                    pooled = maxpool(mag_flat)
                    pooled = pooled.reshape(batch, channels, height, width)
                    peaks = magnitudes == pooled
                else:
                    pooled = maxpool(magnitudes)
                    peaks = magnitudes == pooled
                
                # Emphasize peaks
                fft_latent = fft_latent * (1 + strength * peaks.float())
                result = torch.fft.ifft(fft_latent, dim=-1).real
                
        elif manipulation_type == "noise_injection":
            # Inject controlled noise
            noise = torch.randn_like(result) * strength * 0.1
            if features is not None and "data" in features:
                # Modulate noise by features
                feature_data = features["data"].to(device)
                if feature_data.numel() > 0:
                    feature_scalar = feature_data.mean().item()
                    noise = noise * (1 + feature_scalar)
            result = result + noise
            
        elif manipulation_type == "dynamic_range":
            # Compress or expand dynamic range
            # Apply soft clipping/expansion
            threshold = 0.5
            if strength < 1.0:
                # Compression
                mask = torch.abs(result) > threshold
                result = torch.where(mask, 
                                   torch.sign(result) * (threshold + (torch.abs(result) - threshold) * strength),
                                   result)
            else:
                # Expansion
                mask = torch.abs(result) < threshold
                result = torch.where(mask,
                                   result * strength,
                                   result)
                                   
        elif manipulation_type == "spatial_transform":
            # Transform spatial relationships in latent
            if len(spatial_dims) == 1:
                length = spatial_dims[0]
                if length > 1:
                    base = result.clone()
                    coords = torch.linspace(-1, 1, length, device=device, dtype=result.dtype)
                    displacement = torch.sin(coords * math.pi * 2) * (float(strength) * 0.3)
                    warped_coords = (coords + displacement).clamp(-1.0, 1.0)
                    grid = torch.zeros((batch, 1, length, 2), device=device, dtype=result.dtype)
                    grid[..., 0] = warped_coords
                    grid[..., 1] = 0.0
                    warped = torch.nn.functional.grid_sample(
                        result.unsqueeze(2),
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=False,
                    ).squeeze(2)
                    mix = max(0.0, min(float(strength), 1.0))
                    result = torch.lerp(base, warped, mix)
            elif len(spatial_dims) == 2:
                # Rotate latent space
                angle = float(strength) * math.pi
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                
                # Create rotation matrix
                rot_matrix = torch.tensor(
                    [[cos_a, -sin_a], [sin_a, cos_a]],
                    dtype=result.dtype,
                    device=device,
                )
                
                # Apply rotation (simplified - works on channel pairs)
                if channels >= 2:
                    base = result.clone()
                    mix = max(0.0, min(float(strength), 1.0))
                    for i in range(0, channels-1, 2):
                        pair = result[:, i:i+2].reshape(batch, 2, -1)
                        rotated = torch.matmul(rot_matrix, pair)
                        blended = torch.lerp(
                            base[:, i:i+2].reshape(batch, 2, -1),
                            rotated,
                            mix,
                        ).reshape(batch, 2, *spatial_dims)
                        result[:, i:i+2] = blended
                        
        elif manipulation_type == "resonance_filter":
            # Apply resonant filtering in latent space
            if len(spatial_dims) >= 1:
                # Create resonance kernel
                center_freq = frequency / 22050  # Normalize frequency
                bandwidth = max(0.1 * float(strength), 1e-3)
                
                freqs = torch.linspace(0, 1, spatial_dims[-1], device=device)
                resonance = torch.exp(-((freqs - center_freq) ** 2) / (2 * bandwidth ** 2))
                
                # Apply in frequency domain
                fft_latent = torch.fft.fft(result, dim=-1)
                fft_latent = fft_latent * resonance
                result = torch.fft.ifft(fft_latent, dim=-1).real

        return (wrap_latent(result, latent_meta),)
