from server import PromptServer
import torch
import torch.nn as nn
import random
import numpy as np
from typing import Dict, List, Tuple, Any

class NetworkBending:
    """
    Network Bending Node - Performs various modifications on neural network models
    
    This node allows for creative manipulation of loaded model checkpoints including:
    - Weight noise injection
    - Layer freezing/unfreezing
    - Weight scaling
    - Activation function replacement
    - Layer pruning
    - Weight mixing between models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model checkpoint to modify"}),
                "operation": ([
                    "add_noise",
                    "scale_weights", 
                    "prune_weights",
                    "randomize_weights",
                    "smooth_weights",
                    "quantize_weights"
                ], {
                    "default": "add_noise",
                    "tooltip": "The type of network bending operation to perform"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Strength of the operation (0-1)"
                }),
                "target_layers": ("STRING", {
                    "default": "all",
                    "multiline": False,
                    "tooltip": "Comma-separated layer names or patterns to target (e.g., 'conv', 'attention')"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible results (-1 for random)"
                }),
            },
            "optional": {
                "model_b": ("MODEL", {
                    "tooltip": "Second model for mixing operations"
                }),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mix ratio when blending two models (0=model_a, 1=model_b)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "bend_network"
    CATEGORY = "network_bending"
    OUTPUT_TOOLTIPS = ("Modified model with network bending applied",)

    def bend_network(self, model, operation, intensity, target_layers, seed, model_b=None, mix_ratio=0.5):
        # Clone the model to avoid modifying the original
        model_clone = model.clone()
        
        # Set random seed if specified
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Get the actual model from the ComfyUI model wrapper
        if hasattr(model_clone, 'model'):
            sd_model = model_clone.model
        else:
            sd_model = model_clone
            
        # Parse target layers
        target_patterns = [pattern.strip() for pattern in target_layers.split(',')]
        if 'all' in target_patterns:
            target_patterns = None
            
        # Track modified layers for reporting
        modified_layers = []
        
        # Apply the selected operation
        if operation == "add_noise":
            modified_layers = self._add_noise(sd_model, intensity, target_patterns)
        elif operation == "scale_weights":
            modified_layers = self._scale_weights(sd_model, intensity, target_patterns)
        elif operation == "prune_weights":
            modified_layers = self._prune_weights(sd_model, intensity, target_patterns)
        elif operation == "randomize_weights":
            modified_layers = self._randomize_weights(sd_model, intensity, target_patterns)
        elif operation == "smooth_weights":
            modified_layers = self._smooth_weights(sd_model, intensity, target_patterns)
        elif operation == "quantize_weights":
            modified_layers = self._quantize_weights(sd_model, intensity, target_patterns)
            
        # Send feedback to UI
        message = f"Applied {operation} to {len(modified_layers)} layers with intensity {intensity}"
        PromptServer.instance.send_sync("network_bending.feedback", {
            "message": message,
            "operation": operation,
            "modified_layers": modified_layers[:10],  # Limit to first 10 for UI
            "total_layers": len(modified_layers)
        })
        
        return (model_clone,)
    
    def _should_modify_layer(self, layer_name: str, patterns: List[str] = None) -> bool:
        """Check if a layer should be modified based on target patterns"""
        if patterns is None:
            return True
        return any(pattern.lower() in layer_name.lower() for pattern in patterns)
    
    def _add_noise(self, model: nn.Module, intensity: float, patterns: List[str] = None) -> List[str]:
        """Add Gaussian noise to model weights"""
        modified = []
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad:
                noise = torch.randn_like(param.data) * intensity * param.data.std()
                param.data.add_(noise)
                modified.append(name)
        return modified
    
    def _scale_weights(self, model: nn.Module, intensity: float, patterns: List[str] = None) -> List[str]:
        """Scale model weights by a factor"""
        modified = []
        scale_factor = 1.0 + (intensity - 0.5) * 2  # Maps 0-1 to 0-2
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad:
                param.data.mul_(scale_factor)
                modified.append(name)
        return modified
    
    def _prune_weights(self, model: nn.Module, intensity: float, patterns: List[str] = None) -> List[str]:
        """Set small weights to zero (pruning)"""
        modified = []
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad:
                threshold = torch.quantile(torch.abs(param.data), intensity)
                mask = torch.abs(param.data) > threshold
                param.data.mul_(mask.float())
                modified.append(name)
        return modified
    
    def _randomize_weights(self, model: nn.Module, intensity: float, patterns: List[str] = None) -> List[str]:
        """Randomize a portion of weights"""
        modified = []
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad:
                mask = torch.rand_like(param.data) < intensity
                random_weights = torch.randn_like(param.data) * param.data.std()
                param.data = torch.where(mask, random_weights, param.data)
                modified.append(name)
        return modified
    
    def _smooth_weights(self, model: nn.Module, intensity: float, patterns: List[str] = None) -> List[str]:
        """Apply smoothing/blurring to weight matrices"""
        modified = []
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad and param.dim() >= 2:
                # Simple averaging with neighbors for 2D+ tensors
                smoothed = param.data.clone()
                if param.dim() == 2:
                    # For 2D weights, average with neighbors
                    kernel = torch.ones(1, 1, 3, 3).to(param.device) / 9.0
                    smoothed = torch.nn.functional.conv2d(
                        param.data.unsqueeze(0).unsqueeze(0),
                        kernel,
                        padding=1
                    ).squeeze()
                param.data = param.data * (1 - intensity) + smoothed * intensity
                modified.append(name)
        return modified
    
    def _quantize_weights(self, model: nn.Module, intensity: float, patterns: List[str] = None) -> List[str]:
        """Quantize weights to discrete levels"""
        modified = []
        num_levels = int(2 + (1 - intensity) * 254)  # 2 to 256 levels
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad:
                # Normalize to 0-1, quantize, then rescale
                min_val = param.data.min()
                max_val = param.data.max()
                normalized = (param.data - min_val) / (max_val - min_val + 1e-8)
                quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
                param.data = quantized * (max_val - min_val) + min_val
                modified.append(name)
        return modified


# Advanced Network Bending Node with more options
class NetworkBendingAdvanced:
    """
    Advanced Network Bending Node - Extended operations for model manipulation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model checkpoint to modify"}),
                "operation": ([
                    "layer_swap",
                    "activation_replace",
                    "weight_transpose",
                    "channel_shuffle",
                    "frequency_filter",
                    "weight_clustering"
                ], {
                    "default": "layer_swap",
                    "tooltip": "Advanced network bending operations"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "preserve_functionality": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Try to preserve model functionality while bending"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "bend_network_advanced"
    CATEGORY = "network_bending"
    
    def bend_network_advanced(self, model, operation, intensity, preserve_functionality):
        # Implementation for advanced operations
        model_clone = model.clone()
        
        # Placeholder for advanced operations
        # These would be implemented based on specific requirements
        
        return (model_clone,)


# Model Mixing Node
class ModelMixer:
    """
    Mix two models together with various blending modes
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL", {"tooltip": "First model"}),
                "model_b": ("MODEL", {"tooltip": "Second model"}),
                "mix_mode": ([
                    "linear_interpolation",
                    "weighted_sum",
                    "layer_wise_mix",
                    "frequency_blend",
                    "random_mix"
                ], {
                    "default": "linear_interpolation"
                }),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "0 = 100% model A, 1 = 100% model B"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mix_models"
    CATEGORY = "network_bending"
    
    def mix_models(self, model_a, model_b, mix_mode, mix_ratio):
        # Clone model A as base
        result = model_a.clone()
        
        # Get the actual models
        sd_a = model_a.model if hasattr(model_a, 'model') else model_a
        sd_b = model_b.model if hasattr(model_b, 'model') else model_b
        sd_result = result.model if hasattr(result, 'model') else result
        
        if mix_mode == "linear_interpolation":
            # Simple linear interpolation between weights
            for (name_a, param_a), (name_b, param_b) in zip(
                sd_a.named_parameters(), 
                sd_b.named_parameters()
            ):
                if name_a == name_b and param_a.shape == param_b.shape:
                    # Find corresponding parameter in result model
                    for name_r, param_r in sd_result.named_parameters():
                        if name_r == name_a:
                            param_r.data = (1 - mix_ratio) * param_a.data + mix_ratio * param_b.data
                            break
        
        return (result,)


# Register all nodes
NODE_CLASS_MAPPINGS = {
    "NetworkBending": NetworkBending,
    "NetworkBendingAdvanced": NetworkBendingAdvanced,
    "ModelMixer": ModelMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NetworkBending": "Network Bending",
    "NetworkBendingAdvanced": "Network Bending (Advanced)",
    "ModelMixer": "Model Mixer",
}

# Export web directory for UI components
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']