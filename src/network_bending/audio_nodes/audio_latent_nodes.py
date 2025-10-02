"""
Audio Latent Space Manipulation Nodes for Stable Audio
These nodes enable working directly with audio latents for style transfer and hybrid generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torchaudio
import librosa

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
    
    RETURN_TYPES = ("AUDIO_LATENT", "LATENT_INFO")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "encode_audio"
    CATEGORY = "audio/latent"
    
    def encode_audio(self, audio, vae, normalize=True, target_sample_rate=44100):
        # Extract audio tensor and sample rate
        waveform, sample_rate = audio
        
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
            device = next(vae.parameters()).device
            waveform = waveform.to(device)
            
            # Encode
            latent = vae.encode(waveform)
            
            # Get latent statistics
            latent_mean = latent.mean()
            latent_std = latent.std()
            
        # Create info dictionary
        info = {
            "original_shape": waveform.shape,
            "latent_shape": latent.shape,
            "sample_rate": target_sample_rate,
            "latent_mean": latent_mean.item(),
            "latent_std": latent_std.item(),
        }
        
        return (latent, info)

class AudioVAEDecode:
    """
    Decode audio latents back to waveforms
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("AUDIO_LATENT", {"tooltip": "Audio latent to decode"}),
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
            device = next(vae.parameters()).device
            latent = latent.to(device)
            
            # Decode
            waveform = vae.decode(latent)
            
            # Move to CPU for further processing
            waveform = waveform.cpu()
        
        # Denormalize if needed (clamp to valid range)
        if denormalize:
            waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # Extract sample rate (default to 44100 if not stored)
        sample_rate = 44100
        
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
                "latent_a": ("AUDIO_LATENT", {"tooltip": "First audio latent"}),
                "latent_b": ("AUDIO_LATENT", {"tooltip": "Second audio latent"}),
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
    
    RETURN_TYPES = ("AUDIO_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "interpolate_latents"
    CATEGORY = "audio/latent"
    
    def interpolate_latents(self, latent_a, latent_b, interpolation_factor, 
                          interpolation_mode, curve_power=1.0):
        # Ensure latents are on the same device
        device = latent_a.device
        latent_b = latent_b.to(device)
        
        # Apply curve to interpolation factor if needed
        if curve_power != 1.0:
            interpolation_factor = interpolation_factor ** curve_power
        
        if interpolation_mode == "linear":
            # Simple linear interpolation
            result = (1 - interpolation_factor) * latent_a + interpolation_factor * latent_b
            
        elif interpolation_mode == "spherical":
            # Spherical linear interpolation (SLERP)
            # Normalize latents
            latent_a_norm = F.normalize(latent_a.flatten(1), dim=1, eps=1e-6).reshape(latent_a.shape)
            latent_b_norm = F.normalize(latent_b.flatten(1), dim=1, eps=1e-6).reshape(latent_b.shape)
            
            # Compute angle between latents
            dot_product = (latent_a_norm * latent_b_norm).sum()
            angle = torch.acos(torch.clamp(dot_product, -1, 1))
            
            # SLERP formula
            if angle < 1e-6:
                result = (1 - interpolation_factor) * latent_a + interpolation_factor * latent_b
            else:
                sin_angle = torch.sin(angle)
                result = (torch.sin((1 - interpolation_factor) * angle) / sin_angle) * latent_a + \
                        (torch.sin(interpolation_factor * angle) / sin_angle) * latent_b
                        
        elif interpolation_mode == "cubic":
            # Cubic interpolation (smooth start and end)
            t = interpolation_factor
            t2 = t * t
            t3 = t2 * t
            factor = 3 * t2 - 2 * t3
            result = (1 - factor) * latent_a + factor * latent_b
            
        elif interpolation_mode == "sine":
            # Sine-based interpolation
            factor = 0.5 * (1 + torch.sin(torch.tensor(np.pi * (interpolation_factor - 0.5))))
            result = (1 - factor) * latent_a + factor * latent_b
        
        return (result,)

class AudioLatentBlend:
    """
    Blend multiple audio latents with customizable weights
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("AUDIO_LATENT", {"tooltip": "First audio latent"}),
                "latent_b": ("AUDIO_LATENT", {"tooltip": "Second audio latent"}),
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
                "latent_c": ("AUDIO_LATENT", {"tooltip": "Optional third audio latent"}),
                "latent_d": ("AUDIO_LATENT", {"tooltip": "Optional fourth audio latent"}),
                "weight_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_c": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_d": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "blend_latents"
    CATEGORY = "audio/latent"
    
    def blend_latents(self, latent_a, latent_b, blend_mode, blend_factor,
                     latent_c=None, latent_d=None,
                     weight_a=1.0, weight_b=1.0, weight_c=1.0, weight_d=1.0,
                     normalize=True):
        
        device = latent_a.device
        
        # Collect all available latents and weights
        latents = [latent_a, latent_b]
        weights = [weight_a, weight_b]
        
        if latent_c is not None:
            latents.append(latent_c.to(device))
            weights.append(weight_c)
        if latent_d is not None:
            latents.append(latent_d.to(device))
            weights.append(weight_d)
        
        # Normalize weights if requested by the user
        if normalize:
            total_weight = sum(weights)
            if abs(total_weight) > 1e-12:
                weights = [w / total_weight for w in weights]
        
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
        
        return (result,)

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
        waveform, sample_rate = audio
        
        # Convert to numpy for librosa
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze().cpu().numpy()
        else:
            waveform_np = waveform
        
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
                "latent": ("AUDIO_LATENT", {"tooltip": "Audio latent to manipulate"}),
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
    
    RETURN_TYPES = ("AUDIO_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "manipulate_latent"
    CATEGORY = "audio/latent"
    
    def manipulate_latent(self, latent, manipulation_type, strength, 
                         features=None, frequency=440.0, time_factor=1.0):
        device = latent.device
        result = latent.clone()
        
        # Get latent dimensions
        batch, channels, *spatial_dims = result.shape
        
        if manipulation_type == "frequency_shift":
            # Shift frequency content in latent space
            # Apply FFT-like transformation in latent space
            if len(spatial_dims) == 2:
                # 2D latent (height, width)
                fft_latent = torch.fft.fft2(result)
                # Shift frequencies
                shift_amount = int(strength * spatial_dims[1] * frequency / 22050)
                fft_latent = torch.roll(fft_latent, shifts=shift_amount, dims=-1)
                result = torch.fft.ifft2(fft_latent).real
            else:
                # 1D latent
                fft_latent = torch.fft.fft(result)
                shift_amount = int(strength * spatial_dims[0] * frequency / 22050)
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
                    mag_flat = magnitudes.reshape(batch * channels * spatial_dims[0], -1)
                    peaks = (magnitudes == maxpool(mag_flat).reshape(magnitudes.shape))
                else:
                    peaks = (magnitudes == maxpool(magnitudes))
                
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
            if len(spatial_dims) == 2:
                # Rotate latent space
                angle = strength * np.pi
                cos_a = torch.cos(torch.tensor(angle))
                sin_a = torch.sin(torch.tensor(angle))
                
                # Create rotation matrix
                rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
                
                # Apply rotation (simplified - works on channel pairs)
                if channels >= 2:
                    for i in range(0, channels-1, 2):
                        pair = result[:, i:i+2].reshape(batch, 2, -1)
                        rotated = torch.matmul(rot_matrix, pair)
                        result[:, i:i+2] = rotated.reshape(batch, 2, *spatial_dims)
                        
        elif manipulation_type == "resonance_filter":
            # Apply resonant filtering in latent space
            if len(spatial_dims) >= 1:
                # Create resonance kernel
                center_freq = frequency / 22050  # Normalize frequency
                bandwidth = 0.1 * strength
                
                freqs = torch.linspace(0, 1, spatial_dims[-1], device=device)
                resonance = torch.exp(-((freqs - center_freq) ** 2) / (2 * bandwidth ** 2))
                
                # Apply in frequency domain
                fft_latent = torch.fft.fft(result, dim=-1)
                fft_latent = fft_latent * resonance
                result = torch.fft.ifft(fft_latent, dim=-1).real
        
        return (result,)