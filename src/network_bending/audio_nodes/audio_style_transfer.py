"""
Audio Style Transfer and Advanced Latent Manipulation Nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class AudioStyleTransfer:
    """
    Transfer style characteristics from one audio to another in latent space
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_latent": ("AUDIO_LATENT", {"tooltip": "Content audio latent"}),
                "style_latent": ("AUDIO_LATENT", {"tooltip": "Style audio latent"}),
                "transfer_mode": ([
                    "global",
                    "frequency_bands",
                    "temporal_segments",
                    "adaptive",
                    "neural_style"
                ], {
                    "default": "adaptive",
                    "tooltip": "Style transfer mode"
                }),
                "style_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How much style to transfer"
                }),
            },
            "optional": {
                "preserve_content": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "tooltip": "How much original content to preserve"
                }),
                "frequency_bands": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 16,
                    "tooltip": "Number of frequency bands for band-wise transfer"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "transfer_style"
    CATEGORY = "audio/style"
    
    def transfer_style(self, content_latent, style_latent, transfer_mode, 
                      style_strength, preserve_content=0.3, frequency_bands=4):
        device = content_latent.device
        style_latent = style_latent.to(device)
        
        if transfer_mode == "global":
            # Global style transfer using statistics matching
            # Compute statistics
            content_mean = content_latent.mean(dim=(2, 3), keepdim=True) if content_latent.dim() == 4 else content_latent.mean(dim=2, keepdim=True)
            content_std = content_latent.std(dim=(2, 3), keepdim=True) if content_latent.dim() == 4 else content_latent.std(dim=2, keepdim=True)
            
            style_mean = style_latent.mean(dim=(2, 3), keepdim=True) if style_latent.dim() == 4 else style_latent.mean(dim=2, keepdim=True)
            style_std = style_latent.std(dim=(2, 3), keepdim=True) if style_latent.dim() == 4 else style_latent.std(dim=2, keepdim=True)
            
            # Normalize content and apply style statistics
            normalized = (content_latent - content_mean) / (content_std + 1e-6)
            stylized = normalized * style_std + style_mean
            
            # Blend with original
            result = stylized * style_strength + content_latent * preserve_content
            
        elif transfer_mode == "frequency_bands":
            # Transfer style in different frequency bands
            result = content_latent.clone()
            
            # Apply FFT
            fft_content = torch.fft.fft(content_latent, dim=-1)
            fft_style = torch.fft.fft(style_latent, dim=-1)
            
            # Split into frequency bands
            freq_size = fft_content.shape[-1]
            band_size = freq_size // frequency_bands
            
            for i in range(frequency_bands):
                start_idx = i * band_size
                end_idx = (i + 1) * band_size if i < frequency_bands - 1 else freq_size
                
                # Transfer magnitude in this band
                content_mag = torch.abs(fft_content[..., start_idx:end_idx])
                content_phase = torch.angle(fft_content[..., start_idx:end_idx])
                style_mag = torch.abs(fft_style[..., start_idx:end_idx])
                
                # Blend magnitudes
                new_mag = style_mag * style_strength + content_mag * preserve_content
                
                # Reconstruct with new magnitude
                fft_content[..., start_idx:end_idx] = new_mag * torch.exp(1j * content_phase)
            
            # Inverse FFT
            result = torch.fft.ifft(fft_content, dim=-1).real
            
        elif transfer_mode == "temporal_segments":
            # Transfer style in temporal segments
            time_dim = content_latent.shape[-1]
            segment_size = time_dim // 4  # 4 segments
            
            result = content_latent.clone()
            
            for i in range(4):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < 3 else time_dim
                
                # Get segments
                content_seg = content_latent[..., start_idx:end_idx]
                style_seg = style_latent[..., start_idx:end_idx]
                
                # Transfer style in segment
                seg_mean_c = content_seg.mean()
                seg_std_c = content_seg.std()
                seg_mean_s = style_seg.mean()
                seg_std_s = style_seg.std()
                
                normalized_seg = (content_seg - seg_mean_c) / (seg_std_c + 1e-6)
                stylized_seg = normalized_seg * seg_std_s + seg_mean_s
                
                result[..., start_idx:end_idx] = stylized_seg * style_strength + content_seg * preserve_content
                
        elif transfer_mode == "adaptive":
            # Adaptive style transfer using correlation
            # Flatten spatial dimensions
            content_flat = content_latent.reshape(content_latent.shape[0], content_latent.shape[1], -1)
            style_flat = style_latent.reshape(style_latent.shape[0], style_latent.shape[1], -1)
            
            # Compute correlation matrix
            content_cov = torch.matmul(content_flat, content_flat.transpose(1, 2))
            style_cov = torch.matmul(style_flat, style_flat.transpose(1, 2))
            
            # Eigen decomposition for whitening and coloring
            content_eig_val, content_eig_vec = torch.linalg.eigh(content_cov)
            style_eig_val, style_eig_vec = torch.linalg.eigh(style_cov)
            
            # Whitening transform
            content_eig_val = torch.clamp(content_eig_val, min=1e-6)
            whitening = torch.matmul(content_eig_vec, torch.diag_embed(1.0 / torch.sqrt(content_eig_val)))
            
            # Coloring transform
            style_eig_val = torch.clamp(style_eig_val, min=1e-6)
            coloring = torch.matmul(style_eig_vec, torch.diag_embed(torch.sqrt(style_eig_val)))
            
            # Apply transforms
            whitened = torch.matmul(whitening.transpose(1, 2), content_flat)
            stylized = torch.matmul(coloring, whitened)
            
            # Reshape back
            stylized = stylized.reshape(content_latent.shape)
            result = stylized * style_strength + content_latent * preserve_content
            
        elif transfer_mode == "neural_style":
            # Neural style transfer inspired approach
            # Use convolutional layers to extract features
            conv1 = nn.Conv1d(content_latent.shape[1], 64, 3, padding=1).to(device)
            conv2 = nn.Conv1d(64, 32, 3, padding=1).to(device)
            
            # Extract features
            content_feat1 = F.relu(conv1(content_latent.squeeze(2) if content_latent.dim() == 4 else content_latent))
            content_feat2 = F.relu(conv2(content_feat1))
            
            style_feat1 = F.relu(conv1(style_latent.squeeze(2) if style_latent.dim() == 4 else style_latent))
            style_feat2 = F.relu(conv2(style_feat1))
            
            # Compute Gram matrices
            def gram_matrix(features):
                b, c, w = features.size()
                features = features.view(b, c, w)
                gram = torch.bmm(features, features.transpose(1, 2))
                return gram / (c * w)
            
            content_gram = gram_matrix(content_feat2)
            style_gram = gram_matrix(style_feat2)
            
            # Match Gram matrices
            target_gram = style_gram * style_strength + content_gram * preserve_content
            
            # Approximate reconstruction (simplified)
            result = content_latent + 0.1 * (target_gram.mean() - content_gram.mean())
        
        return (result,)

class AudioLatentGuidance:
    """
    Guide audio generation with reference latents during the diffusion process
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Stable Audio diffusion model"}),
                "reference_latent": ("AUDIO_LATENT", {"tooltip": "Reference audio latent for guidance"}),
                "guidance_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Strength of latent guidance"
                }),
                "guidance_mode": ([
                    "additive",
                    "multiplicative",
                    "attention",
                    "gradient"
                ], {
                    "default": "attention",
                    "tooltip": "How to apply guidance"
                }),
            },
            "optional": {
                "start_at_step": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Step to start applying guidance"
                }),
                "end_at_step": ("INT", {
                    "default": 1000,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Step to stop applying guidance"
                }),
                "features": ("AUDIO_FEATURES", {"tooltip": "Optional features to modulate guidance"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_guidance"
    CATEGORY = "audio/conditioning"
    
    def apply_guidance(self, model, reference_latent, guidance_strength, guidance_mode,
                      start_at_step=0, end_at_step=1000, features=None):
        # Clone the model to avoid modifying the original
        guided_model = model.clone()
        
        # Store guidance parameters in the model
        guided_model.audio_latent_guidance = {
            "reference_latent": reference_latent,
            "strength": guidance_strength,
            "mode": guidance_mode,
            "start_step": start_at_step,
            "end_step": end_at_step,
            "features": features
        }
        
        # Patch the model's forward method to include guidance
        original_forward = guided_model.model.forward
        
        def guided_forward(x, timestep, context=None, **kwargs):
            # Get current step from timestep
            current_step = timestep[0].item() if isinstance(timestep, torch.Tensor) else timestep
            
            # Check if we should apply guidance
            guidance = guided_model.audio_latent_guidance
            if guidance["start_step"] <= current_step <= guidance["end_step"]:
                ref_latent = guidance["reference_latent"].to(x.device)
                strength = guidance["strength"]
                
                if guidance["mode"] == "additive":
                    # Add reference latent influence
                    x = x + strength * 0.1 * ref_latent
                    
                elif guidance["mode"] == "multiplicative":
                    # Multiplicative guidance
                    ref_norm = F.normalize(ref_latent.flatten(1), dim=1).reshape(ref_latent.shape)
                    x = x * (1 + strength * ref_norm)
                    
                elif guidance["mode"] == "attention":
                    # Attention-based guidance (modify context)
                    if context is not None:
                        # Add reference latent information to context
                        ref_context = ref_latent.mean(dim=(2, 3), keepdim=True) if ref_latent.dim() == 4 else ref_latent.mean(dim=2, keepdim=True)
                        ref_context = ref_context.expand_as(context)
                        context = context + strength * 0.1 * ref_context
                        
                elif guidance["mode"] == "gradient":
                    # Gradient-based guidance
                    # Compute similarity between x and reference
                    similarity = F.cosine_similarity(x.flatten(1), ref_latent.flatten(1), dim=1)
                    similarity = similarity.mean()
                    
                    # Modify x to increase similarity
                    grad_direction = ref_latent - x
                    x = x + strength * 0.05 * grad_direction * (1 - similarity)
            
            # Call original forward
            return original_forward(x, timestep, context, **kwargs)
        
        # Replace forward method
        guided_model.model.forward = guided_forward
        
        return (guided_model,)

class AudioReferenceEncoder:
    """
    Encode multiple audio references into a combined latent representation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "Stable Audio VAE model"}),
                "audio_a": ("AUDIO", {"tooltip": "First reference audio"}),
                "combination_mode": ([
                    "average",
                    "weighted",
                    "pca",
                    "attention"
                ], {
                    "default": "attention",
                    "tooltip": "How to combine multiple references"
                }),
            },
            "optional": {
                "audio_b": ("AUDIO", {"tooltip": "Second reference audio"}),
                "audio_c": ("AUDIO", {"tooltip": "Third reference audio"}),
                "audio_d": ("AUDIO", {"tooltip": "Fourth reference audio"}),
                "weight_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_c": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "weight_d": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "extract_segments": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extract most characteristic segments"
                }),
                "segment_duration": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "tooltip": "Duration of segments to extract (seconds)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO_LATENT", "LATENT_INFO")
    RETURN_NAMES = ("combined_latent", "info")
    FUNCTION = "encode_references"
    CATEGORY = "audio/reference"
    
    def encode_references(self, vae, audio_a, combination_mode,
                         audio_b=None, audio_c=None, audio_d=None,
                         weight_a=1.0, weight_b=1.0, weight_c=1.0, weight_d=1.0,
                         extract_segments=False, segment_duration=2.0):
        
        device = next(vae.parameters()).device
        
        # Collect all available audio inputs
        audios = [audio_a]
        weights = [weight_a]
        
        if audio_b is not None:
            audios.append(audio_b)
            weights.append(weight_b)
        if audio_c is not None:
            audios.append(audio_c)
            weights.append(weight_c)
        if audio_d is not None:
            audios.append(audio_d)
            weights.append(weight_d)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Encode all audio inputs
        encoded_latents = []
        
        for audio, weight in zip(audios, weights):
            waveform, sample_rate = audio
            
            # Extract characteristic segment if requested
            if extract_segments:
                segment_samples = int(segment_duration * sample_rate)
                
                # Find segment with highest energy
                if waveform.shape[-1] > segment_samples:
                    # Compute energy in sliding windows
                    energy = []
                    for i in range(0, waveform.shape[-1] - segment_samples, segment_samples // 4):
                        segment = waveform[..., i:i+segment_samples]
                        energy.append(torch.sum(segment ** 2).item())
                    
                    # Get the highest energy segment
                    best_idx = np.argmax(energy) * (segment_samples // 4)
                    waveform = waveform[..., best_idx:best_idx+segment_samples]
            
            # Normalize audio
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Ensure correct shape
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            # Encode with VAE
            with torch.no_grad():
                waveform = waveform.to(device)
                latent = vae.encode(waveform)
                encoded_latents.append((latent, weight))
        
        # Combine latents based on mode
        if combination_mode == "average":
            # Simple weighted average
            combined = sum(latent * weight for latent, weight in encoded_latents)
            
        elif combination_mode == "weighted":
            # Weighted combination with normalization
            combined = torch.zeros_like(encoded_latents[0][0])
            for latent, weight in encoded_latents:
                combined += latent * weight
            combined = F.normalize(combined, dim=1)
            
        elif combination_mode == "pca":
            # PCA-based combination
            # Stack all latents
            all_latents = torch.stack([lat for lat, _ in encoded_latents])
            
            # Flatten spatial dimensions
            b, c, *spatial = all_latents.shape
            all_latents_flat = all_latents.reshape(len(encoded_latents), -1)
            
            # Compute PCA
            mean = all_latents_flat.mean(0, keepdim=True)
            centered = all_latents_flat - mean
            
            # Compute covariance
            cov = torch.matmul(centered.T, centered) / (len(encoded_latents) - 1)
            
            # Get first principal component
            eigvals, eigvecs = torch.linalg.eigh(cov)
            pc1 = eigvecs[:, -1]  # Largest eigenvalue
            
            # Project and combine
            projections = torch.matmul(centered, pc1)
            combined_flat = mean + projections.mean() * pc1
            combined = combined_flat.reshape(1, c, *spatial)
            
        elif combination_mode == "attention":
            # Attention-based combination
            # Use latents as queries and keys
            all_latents = torch.stack([lat for lat, _ in encoded_latents])
            
            # Compute attention scores
            queries = all_latents.mean(dim=(2, 3)) if all_latents.dim() == 4 else all_latents.mean(dim=2)
            keys = queries
            
            # Scaled dot-product attention
            scores = torch.matmul(queries, keys.transpose(0, 1)) / np.sqrt(queries.shape[-1])
            attention_weights = F.softmax(scores, dim=-1)
            
            # Apply attention weights
            combined = torch.zeros_like(encoded_latents[0][0])
            for i, (latent, base_weight) in enumerate(encoded_latents):
                # Combine base weight with attention weight
                final_weight = base_weight * attention_weights[i].mean()
                combined += latent * final_weight
        
        # Create info dictionary
        info = {
            "num_references": len(audios),
            "combination_mode": combination_mode,
            "weights": weights,
            "latent_shape": combined.shape,
            "extracted_segments": extract_segments
        }
        
        return (combined, info)