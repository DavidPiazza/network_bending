try:
    from server import PromptServer
except Exception:  # pragma: no cover
    class _DummyPromptServer:
        instance = None

        def __init__(self):
            class _Inst:
                def send_sync(self, *args, **kwargs):
                    return None

            self.instance = _Inst()

    PromptServer = _DummyPromptServer()
import torch
import torch.nn as nn
import random
import numpy as np
from typing import Dict, List, Tuple, Any

# Import audio nodes
from .audio_nodes import NODE_CLASS_MAPPINGS as AUDIO_NODE_CLASS_MAPPINGS
from .audio_nodes import NODE_DISPLAY_NAME_MAPPINGS as AUDIO_NODE_DISPLAY_NAME_MAPPINGS

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
        try:
            PromptServer.instance.send_sync("network_bending.feedback", {
                "message": message,
                "operation": operation,
                "modified_layers": modified_layers[:10],  # Limit to first 10 for UI
                "total_layers": len(modified_layers)
            })
        except Exception:
            pass
        
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
                abs_param = torch.abs(param.data)
                if abs_param.numel() == 0:
                    continue
                q = float(max(0.0, min(1.0, intensity)))
                threshold = torch.quantile(abs_param, q)
                mask = torch.abs(param.data) > threshold
                # Convert mask to the same dtype as the parameter
                param.data.mul_(mask.to(dtype=param.dtype))
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
                    # Create kernel with same dtype and device as the parameter
                    kernel = torch.ones(1, 1, 3, 3, dtype=param.dtype, device=param.device) / 9.0
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
                denom = (max_val - min_val)
                if float(denom.abs().item()) < 1e-12:
                    modified.append(name)
                    continue
                normalized = (param.data - min_val) / (denom + 1e-8)
                quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
                param.data = quantized * denom + min_val
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


class InvertedPruning:
    """
    Inverted Model Pruning - Selectively removes critical weights for artistic degradation
    
    Instead of preserving important weights (standard pruning), this node removes them,
    creating unique artistic effects. Based on the inverted Lottery Ticket Hypothesis.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply inverted pruning to"}),
                "pruning_mode": ([
                    "magnitude_inverted",
                    "structured_inverted",
                    "attention_head_removal",
                    "channel_pruning_inverted",
                    "gradient_based_inverted"
                ], {
                    "default": "magnitude_inverted",
                    "tooltip": "Type of inverted pruning to apply"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "Percentage of weights to remove (0.1 = remove top 10% most important)"
                }),
                "target_layers": ("STRING", {
                    "default": "all",
                    "multiline": False,
                    "tooltip": "Comma-separated layer patterns (e.g., 'attention', 'conv', 'mlp')"
                }),
                "preserve_functionality": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How much to preserve base functionality (0=pure degradation, 1=mild effect)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible pruning (-1 for random)"
                }),
            },
            "optional": {
                "gradient_accumulation_steps": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of gradient accumulation steps for more stable importance estimation"
                }),
                "use_actual_gradients": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use actual gradient computation (slower but more accurate) or simplified method"
                }),
                "gradient_loss_type": ([
                    "reconstruction",
                    "magnitude",
                    "perceptual",
                    "variance"
                ], {
                    "default": "reconstruction",
                    "tooltip": "Loss function for gradient computation"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_inverted_pruning"
    CATEGORY = "network_bending"
    OUTPUT_TOOLTIPS = ("Model with inverted pruning applied",)
    
    def apply_inverted_pruning(self, model, pruning_mode, threshold, target_layers, preserve_functionality, seed, 
                             gradient_accumulation_steps=1, use_actual_gradients=True, gradient_loss_type="reconstruction"):
        # Clone the model
        model_clone = model.clone()
        
        # Set random seed
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Get the actual model
        sd_model = model_clone.model if hasattr(model_clone, 'model') else model_clone
        
        # Parse target layers
        target_patterns = [pattern.strip() for pattern in target_layers.split(',')]
        if 'all' in target_patterns:
            target_patterns = None
        
        # Track modified layers
        modified_layers = []
        
        # Apply the selected pruning mode
        if pruning_mode == "magnitude_inverted":
            modified_layers = self._magnitude_inverted_pruning(sd_model, threshold, preserve_functionality, target_patterns)
        elif pruning_mode == "structured_inverted":
            modified_layers = self._structured_inverted_pruning(sd_model, threshold, preserve_functionality, target_patterns)
        elif pruning_mode == "attention_head_removal":
            modified_layers = self._attention_head_removal(sd_model, threshold, preserve_functionality, target_patterns)
        elif pruning_mode == "channel_pruning_inverted":
            modified_layers = self._channel_pruning_inverted(sd_model, threshold, preserve_functionality, target_patterns)
        elif pruning_mode == "gradient_based_inverted":
            modified_layers = self._gradient_based_inverted(sd_model, threshold, preserve_functionality, target_patterns,
                                                           use_actual_gradients, gradient_accumulation_steps, gradient_loss_type)
        
        # Send feedback
        try:
            PromptServer.instance.send_sync("network_bending.feedback", {
                "message": f"Applied {pruning_mode} to {len(modified_layers)} layers",
                "operation": pruning_mode,
                "modified_layers": modified_layers[:10],
                "total_layers": len(modified_layers),
                "threshold": threshold
            })
        except Exception:
            pass
        
        return (model_clone,)
    
    def _should_modify_layer(self, layer_name: str, patterns: List[str] = None) -> bool:
        """Check if a layer should be modified based on target patterns"""
        if patterns is None:
            return True
        return any(pattern.lower() in layer_name.lower() for pattern in patterns)
    
    def _magnitude_inverted_pruning(self, model: nn.Module, threshold: float, preserve: float, patterns: List[str] = None) -> List[str]:
        """Remove weights with highest magnitude (opposite of standard magnitude pruning)"""
        modified = []
        
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad:
                # Calculate magnitude threshold - we want to remove the TOP magnitude weights
                abs_weights = torch.abs(param.data)
                k = int(threshold * param.data.numel())
                
                if k > 0:
                    # Find threshold value - weights above this will be removed
                    threshold_val = torch.topk(abs_weights.flatten(), k).values[-1]
                    
                    # Create mask - True where we want to KEEP weights (low magnitude)
                    mask = abs_weights <= threshold_val
                    
                    # Apply preservation factor
                    if preserve > 0:
                        # Randomly preserve some high-magnitude weights
                        preserve_mask = torch.rand_like(param.data) < preserve
                        mask = mask | preserve_mask
                    
                    # Apply mask
                    param.data.mul_(mask.to(dtype=param.dtype))
                    modified.append(name)
        
        return modified
    
    def _structured_inverted_pruning(self, model: nn.Module, threshold: float, preserve: float, patterns: List[str] = None) -> List[str]:
        """Remove entire structures (channels/filters) with highest importance"""
        modified = []
        
        for name, module in model.named_modules():
            if not self._should_modify_layer(name, patterns):
                continue
            
            # Handle Conv2d layers
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                # Calculate importance per output channel (L2 norm)
                importance = torch.norm(weight, p=2, dim=(1, 2, 3))
                
                # Remove channels with HIGHEST importance
                k = int(threshold * len(importance))
                if k > 0:
                    _, indices_to_remove = torch.topk(importance, k)
                    
                    # Apply preservation
                    if preserve > 0:
                        num_preserve = int(k * preserve)
                        indices_to_remove = indices_to_remove[num_preserve:]
                    
                    # Zero out high-importance channels
                    weight[indices_to_remove] = 0
                    modified.append(f"{name}.weight")
            
            # Handle Linear layers
            elif isinstance(module, nn.Linear):
                weight = module.weight.data
                # Calculate importance per output neuron
                importance = torch.norm(weight, p=2, dim=1)
                
                # Remove neurons with HIGHEST importance
                k = int(threshold * len(importance))
                if k > 0:
                    _, indices_to_remove = torch.topk(importance, k)
                    
                    # Apply preservation
                    if preserve > 0:
                        num_preserve = int(k * preserve)
                        indices_to_remove = indices_to_remove[num_preserve:]
                    
                    # Zero out high-importance neurons
                    weight[indices_to_remove] = 0
                    modified.append(f"{name}.weight")
        
        return modified
    
    def _attention_head_removal(self, model: nn.Module, threshold: float, preserve: float, patterns: List[str] = None) -> List[str]:
        """Remove most important attention heads in transformer models"""
        modified = []
        
        for name, module in model.named_modules():
            # Look for multi-head attention modules
            if ('attention' in name.lower() or 'attn' in name.lower()) and self._should_modify_layer(name, patterns):
                # Check for Q, K, V projections
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']:
                    if hasattr(module, proj_name):
                        proj = getattr(module, proj_name)
                        if isinstance(proj, nn.Linear):
                            weight = proj.weight.data
                            
                            # Assume head dimension is last dimension / num_heads
                            # This is a simplified approach - real implementation would need model-specific logic
                            if weight.shape[0] % 8 == 0:  # Assume 8 heads for simplicity
                                num_heads = 8
                                head_dim = weight.shape[0] // num_heads
                                
                                # Calculate importance per head
                                weight_reshaped = weight.view(num_heads, head_dim, -1)
                                head_importance = torch.norm(weight_reshaped, p=2, dim=(1, 2))
                                
                                # Remove heads with HIGHEST importance
                                k = max(1, int(threshold * num_heads))
                                _, heads_to_remove = torch.topk(head_importance, k)
                                
                                # Apply preservation
                                if preserve > 0:
                                    num_preserve = int(k * preserve)
                                    heads_to_remove = heads_to_remove[num_preserve:]
                                
                                # Zero out high-importance heads
                                for head_idx in heads_to_remove:
                                    start_idx = head_idx * head_dim
                                    end_idx = (head_idx + 1) * head_dim
                                    weight[start_idx:end_idx] = 0
                                
                                modified.append(f"{name}.{proj_name}")
        
        return modified
    
    def _channel_pruning_inverted(self, model: nn.Module, threshold: float, preserve: float, patterns: List[str] = None) -> List[str]:
        """Remove most important channels in convolutional layers"""
        modified = []
        
        # First pass: calculate channel importance across the network
        channel_importance = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and self._should_modify_layer(name, patterns):
                weight = module.weight.data
                
                # Calculate importance for input channels
                in_importance = torch.norm(weight, p=2, dim=(0, 2, 3))
                # Calculate importance for output channels  
                out_importance = torch.norm(weight, p=2, dim=(1, 2, 3))
                
                channel_importance[name] = {
                    'in': in_importance,
                    'out': out_importance,
                    'module': module
                }
        
        # Second pass: prune channels
        for name, info in channel_importance.items():
            module = info['module']
            
            # Prune output channels
            out_importance = info['out']
            k = int(threshold * len(out_importance))
            if k > 0:
                _, indices_to_remove = torch.topk(out_importance, k)
                
                # Apply preservation
                if preserve > 0:
                    num_preserve = int(k * preserve)
                    indices_to_remove = indices_to_remove[num_preserve:]
                
                # Zero out channels
                module.weight.data[indices_to_remove] = 0
                if module.bias is not None:
                    module.bias.data[indices_to_remove] = 0
                
                modified.append(f"{name}.weight")
        
        return modified
    
    def _gradient_based_inverted(self, model: nn.Module, threshold: float, preserve: float, patterns: List[str] = None,
                               use_actual_gradients: bool = True, accumulation_steps: int = 1, loss_type: str = "reconstruction") -> List[str]:
        """Remove weights with highest gradient magnitude (most important for loss)"""
        modified = []
        
        try:
            # Check if we should use actual gradients
            if not use_actual_gradients:
                return self._gradient_based_inverted_simple(model, threshold, preserve, patterns)
            
            # Attempt to compute actual gradients
            gradients = self._compute_gradients(model, accumulation_steps, loss_type)
            
            if gradients:
                # Use actual gradients for importance
                for name, param in model.named_parameters():
                    if self._should_modify_layer(name, patterns) and param.requires_grad and name in gradients:
                        importance = torch.abs(gradients[name])
                        
                        k = int(threshold * param.data.numel())
                        if k > 0:
                            # Find threshold value - remove weights with highest gradient magnitude
                            threshold_val = torch.topk(importance.flatten(), k).values[-1]
                            
                            # Create mask - keep low importance weights
                            mask = importance <= threshold_val
                            
                            # Apply preservation
                            if preserve > 0:
                                preserve_mask = torch.rand_like(param.data) < preserve
                                mask = mask | preserve_mask
                            
                            # Apply mask
                            param.data.mul_(mask.to(dtype=param.dtype))
                            modified.append(name)
            else:
                # Fallback to simplified version
                for name, param in model.named_parameters():
                    if self._should_modify_layer(name, patterns) and param.requires_grad:
                        # Use weight magnitude as proxy for importance
                        importance = torch.abs(param.data) + torch.randn_like(param.data) * 0.1
                        
                        k = int(threshold * param.data.numel())
                        if k > 0:
                            threshold_val = torch.topk(importance.flatten(), k).values[-1]
                            mask = importance <= threshold_val
                            
                            if preserve > 0:
                                preserve_mask = torch.rand_like(param.data) < preserve
                                mask = mask | preserve_mask
                            
                            param.data.mul_(mask.to(dtype=param.dtype))
                            modified.append(name)
                            
        except Exception as e:
            # If gradient computation fails, fallback to simplified version
            print(f"Gradient computation failed: {e}. Using simplified importance estimation.")
            return self._gradient_based_inverted_simple(model, threshold, preserve, patterns)
        
        return modified
    
    def _gradient_based_inverted_simple(self, model: nn.Module, threshold: float, preserve: float, patterns: List[str] = None) -> List[str]:
        """Simplified gradient-based pruning using weight magnitude as proxy"""
        modified = []
        for name, param in model.named_parameters():
            if self._should_modify_layer(name, patterns) and param.requires_grad:
                importance = torch.abs(param.data) + torch.randn_like(param.data) * 0.1
                k = int(threshold * param.data.numel())
                if k > 0:
                    threshold_val = torch.topk(importance.flatten(), k).values[-1]
                    mask = importance <= threshold_val
                    if preserve > 0:
                        preserve_mask = torch.rand_like(param.data) < preserve
                        mask = mask | preserve_mask
                    param.data.mul_(mask.to(dtype=param.dtype))
                    modified.append(name)
        return modified
    
    def _compute_gradients(self, model: nn.Module, accumulation_steps: int = 1, loss_type: str = "reconstruction") -> Dict[str, torch.Tensor]:
        """Compute actual gradients for weight importance"""
        gradients = {}
        accumulated_gradients = {}
        
        # Store original training mode
        was_training = model.training
        model.eval()
        
        try:
            with torch.enable_grad():
                # Accumulate gradients over multiple steps for stability
                for step in range(accumulation_steps):
                    # Zero existing gradients
                    model.zero_grad()
                    
                    # Generate sample input based on model type
                    sample_input = self._generate_sample_input(model)
                    if sample_input is None:
                        return {}
                    
                    # Forward pass
                    output = model(sample_input)
                    
                    # Compute loss based on specified type
                    loss = self._compute_importance_loss(output, sample_input, loss_type)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Accumulate gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if name not in accumulated_gradients:
                                accumulated_gradients[name] = param.grad.data.clone()
                            else:
                                accumulated_gradients[name] += param.grad.data
                
                # Average accumulated gradients
                for name, grad in accumulated_gradients.items():
                    gradients[name] = grad / accumulation_steps
                
                # Clear gradients to free memory
                model.zero_grad()
                
        except Exception as e:
            print(f"Error computing gradients: {e}")
            gradients = {}
        
        finally:
            # Restore original training mode
            model.train(was_training)
        
        return gradients
    
    def _generate_sample_input(self, model: nn.Module) -> torch.Tensor:
        """Generate appropriate sample input for the model"""
        try:
            # Get device
            device = next(model.parameters()).device
            
            # Try to detect model type and generate appropriate input
            # This is a heuristic approach - could be improved with model-specific logic
            
            # Check for common stable diffusion shapes
            if hasattr(model, 'in_channels'):
                # Likely a UNet or similar
                batch_size = 1
                channels = getattr(model, 'in_channels', 4)
                height = width = 64  # Use smaller size for efficiency
                return torch.randn(batch_size, channels, height, width, device=device)
            
            # Check for transformer-like models
            has_embedding = any('embed' in name for name, _ in model.named_modules())
            if has_embedding:
                # Likely a transformer
                batch_size = 1
                seq_length = 77  # Common for text transformers
                hidden_dim = 768  # Common dimension
                return torch.randn(batch_size, seq_length, hidden_dim, device=device)
            
            # Default: try to infer from first layer
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Image input
                    batch_size = 1
                    channels = module.in_channels
                    height = width = 64
                    return torch.randn(batch_size, channels, height, width, device=device)
                elif isinstance(module, nn.Linear) and 'embed' not in name:
                    # Vector input
                    batch_size = 1
                    input_dim = module.in_features
                    return torch.randn(batch_size, input_dim, device=device)
            
            # If we can't determine, return None
            return None
            
        except Exception as e:
            print(f"Error generating sample input: {e}")
            return None
    
    def _compute_importance_loss(self, output: torch.Tensor, input_tensor: torch.Tensor, loss_type: str = "reconstruction") -> torch.Tensor:
        """Compute loss for measuring weight importance"""
        try:
            if loss_type == "reconstruction":
                # Reconstruction loss
                if output.shape == input_tensor.shape:
                    return torch.nn.functional.mse_loss(output, input_tensor)
                else:
                    # Feature matching as fallback
                    return torch.nn.functional.l1_loss(output.mean(), input_tensor.mean())
                    
            elif loss_type == "magnitude":
                # Simple magnitude loss
                return output.abs().mean()
                
            elif loss_type == "perceptual":
                # Perceptual loss using feature statistics
                output_mean = output.mean(dim=list(range(2, output.dim())))
                output_std = output.std(dim=list(range(2, output.dim())))
                
                if input_tensor.shape == output.shape:
                    input_mean = input_tensor.mean(dim=list(range(2, input_tensor.dim())))
                    input_std = input_tensor.std(dim=list(range(2, input_tensor.dim())))
                    mean_loss = torch.nn.functional.mse_loss(output_mean, input_mean)
                    std_loss = torch.nn.functional.mse_loss(output_std, input_std)
                    return mean_loss + std_loss
                else:
                    # Use output statistics only
                    return output_mean.abs().mean() + output_std.abs().mean()
                    
            elif loss_type == "variance":
                # Maximize variance (inverse of typical loss)
                # High variance = high importance
                return -output.var()
                
            else:
                # Default to magnitude loss
                return output.abs().mean()
                
        except Exception as e:
            print(f"Error in loss computation: {e}. Using magnitude loss.")
            # Fallback to simple magnitude loss
            return output.abs().mean()


class LatentFormatConverter:
    """
    Convert between audio and image latent formats
    
    This node handles conversion between different latent tensor shapes:
    - Audio latents: [batch, channels, length] (3D)
    - Image latents: [batch, channels, height, width] (4D)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent samples to convert"}),
                "conversion_mode": ([
                    "audio_to_image",
                    "image_to_audio",
                    "auto_detect"
                ], {
                    "default": "auto_detect",
                    "tooltip": "Conversion direction"
                }),
                "target_height": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Target height when converting to image format"
                }),
                "target_width": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Target width when converting to image format"
                }),
                "reshape_method": ([
                    "reshape",
                    "interpolate",
                    "fold",
                    "tile"
                ], {
                    "default": "reshape",
                    "tooltip": "Method for reshaping tensors"
                }),
                "target_channels": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Target number of channels (-1 to keep original, 4 for SD, 64 for StableAudio)"
                }),
                "channel_mode": ([
                    "project",
                    "pad",
                    "interpolate",
                    "tile"
                ], {
                    "default": "project",
                    "tooltip": "Method for converting channels"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "convert_format"
    CATEGORY = "network_bending"
    OUTPUT_TOOLTIPS = ("Converted latent samples",)
    
    def convert_format(self, samples, conversion_mode, target_height, target_width, reshape_method, target_channels, channel_mode):
        # Get the latent tensor
        latent = samples.copy()
        latent_tensor = latent["samples"].clone()  # Clone to avoid modifying original
        
        # Determine conversion direction
        if conversion_mode == "auto_detect":
            if latent_tensor.dim() == 3:
                conversion_mode = "audio_to_image"
            elif latent_tensor.dim() == 4:
                conversion_mode = "image_to_audio"
            else:
                raise ValueError(f"Unexpected tensor dimensions: {latent_tensor.dim()}. Expected 3D (audio) or 4D (image) tensor.")
        
        # Validate conversion mode matches tensor dimensions
        if conversion_mode == "audio_to_image" and latent_tensor.dim() != 3:
            raise ValueError(f"audio_to_image expects 3D tensor, got {latent_tensor.dim()}D")
        elif conversion_mode == "image_to_audio" and latent_tensor.dim() != 4:
            raise ValueError(f"image_to_audio expects 4D tensor, got {latent_tensor.dim()}D")
        
        # Perform conversion
        if conversion_mode == "audio_to_image":
            converted = self._audio_to_image(latent_tensor, target_height, target_width, reshape_method)
        elif conversion_mode == "image_to_audio":
            converted = self._image_to_audio(latent_tensor, reshape_method)
        else:
            converted = latent_tensor
        
        # Handle channel conversion if needed
        if target_channels > 0 and converted.shape[1] != target_channels:
            converted = self._convert_channels(converted, target_channels, channel_mode)
        
        # Update the latent dictionary
        latent["samples"] = converted
        
        # Send feedback
        try:
            PromptServer.instance.send_sync("network_bending.feedback", {
                "message": f"Converted latent from {list(latent_tensor.shape)} to {list(converted.shape)}",
                "operation": conversion_mode,
                "input_shape": list(latent_tensor.shape),
                "output_shape": list(converted.shape),
                "method": reshape_method
            })
        except Exception:
            pass
        
        return (latent,)
    
    def _audio_to_image(self, tensor, height, width, method):
        """Convert audio latent (B, C, L) to image latent (B, C, H, W)"""
        batch, channels, length = tensor.shape
        
        if method == "reshape":
            # Calculate total elements needed
            total_needed = height * width
            
            if length < total_needed:
                # Pad if necessary
                padding = total_needed - length
                tensor = torch.nn.functional.pad(tensor, (0, padding), mode='constant', value=0)
            elif length > total_needed:
                # Truncate if necessary
                tensor = tensor[:, :, :total_needed]
            
            # Reshape to image format
            return tensor.reshape(batch, channels, height, width)
            
        elif method == "interpolate":
            # First reshape to square-ish shape
            temp_size = int(np.sqrt(length))
            if temp_size * temp_size < length:
                temp_size += 1
            
            # Pad if necessary
            if length < temp_size * temp_size:
                padding = temp_size * temp_size - length
                # Use edge padding for better continuity
                if length > 1:
                    # Pad with edge values for smoother interpolation
                    last_values = tensor[:, :, -1:].expand(-1, -1, padding)
                    tensor = torch.cat([tensor, last_values], dim=2)
                else:
                    tensor = torch.nn.functional.pad(tensor, (0, padding), mode='constant', value=0)
            
            # Reshape to temporary image
            temp_image = tensor[:, :, :temp_size*temp_size].reshape(batch, channels, temp_size, temp_size)
            
            # Interpolate to target size using bicubic for smoother results
            mode = 'bicubic' if height > temp_size or width > temp_size else 'bilinear'
            return torch.nn.functional.interpolate(temp_image, size=(height, width), mode=mode, align_corners=False)
            
        elif method == "fold":
            # Use fold operation to create 2D structure
            # This preserves more local relationships
            
            # Calculate optimal kernel size based on the relationship between input and output dimensions
            target_elements = height * width
            
            # Try to find a kernel size that minimizes padding/truncation
            best_kernel_size = 1
            min_waste = float('inf')
            
            for k in range(1, min(8, int(np.sqrt(length)) + 1)):  # Limit kernel size for efficiency
                needed = target_elements * k * k
                waste = abs(needed - length)
                if waste < min_waste:
                    min_waste = waste
                    best_kernel_size = k
            
            kernel_size = best_kernel_size
            
            # Ensure compatible dimensions
            unfolded_length = height * width * kernel_size * kernel_size
            if length < unfolded_length:
                # Pad with reflection for better continuity
                padding = unfolded_length - length
                # Use reflect padding if possible, otherwise use constant
                if length > 1:
                    tensor = torch.nn.functional.pad(tensor, (0, padding), mode='reflect')
                else:
                    tensor = torch.nn.functional.pad(tensor, (0, padding), mode='constant', value=0)
            elif length > unfolded_length:
                tensor = tensor[:, :, :unfolded_length]
            
            # Reshape for fold operation
            tensor = tensor.reshape(batch, channels * kernel_size * kernel_size, height * width)
            
            # Apply fold with overlapping regions
            stride = max(1, kernel_size // 2)  # Overlap for smoother results
            padding = (kernel_size - stride) // 2
            
            tensor = torch.nn.functional.fold(
                tensor, 
                output_size=(height, width), 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            
            # Create normalization divisor
            divisor = torch.ones(1, channels, height, width, device=tensor.device, dtype=tensor.dtype)
            # Unfold and fold to count overlaps
            divisor = torch.nn.functional.unfold(divisor, kernel_size=kernel_size, stride=stride, padding=padding)
            divisor = torch.nn.functional.fold(divisor, output_size=(height, width), kernel_size=kernel_size, stride=stride, padding=padding)
            
            # Normalize by overlap count
            return tensor / (divisor + 1e-6)
            
        elif method == "tile":
            # Tile the audio to fill image space
            repeat_factor = (height * width + length - 1) // length
            tensor = tensor.repeat(1, 1, repeat_factor)[:, :, :height * width]
            return tensor.reshape(batch, channels, height, width)
    
    def _image_to_audio(self, tensor, method):
        """Convert image latent (B, C, H, W) to audio latent (B, C, L)"""
        batch, channels, height, width = tensor.shape
        
        if method == "reshape":
            # Simple reshape to 1D
            return tensor.reshape(batch, channels, height * width)
            
        elif method == "interpolate":
            # First flatten
            flat = tensor.reshape(batch, channels, height * width)
            # Could add smoothing or filtering here if needed
            return flat
            
        elif method == "fold":
            # Use unfold to create overlapping windows
            kernel_size = min(3, height, width)
            unfolded = torch.nn.functional.unfold(tensor, kernel_size=kernel_size, stride=1, padding=0)
            # Average across kernel dimensions
            unfolded = unfolded.reshape(batch, channels, kernel_size * kernel_size, -1)
            return unfolded.mean(dim=2)
            
        elif method == "tile":
            # Simply flatten
            return tensor.reshape(batch, channels, height * width)
    
    def _convert_channels(self, tensor, target_channels, mode):
        """Convert number of channels in latent tensor"""
        batch = tensor.shape[0]
        current_channels = tensor.shape[1]
        
        if current_channels == target_channels:
            return tensor
        
        if mode == "project":
            # Use linear projection to convert channels
            # This is learnable in theory but we'll use a random projection
            device = tensor.device
            dtype = tensor.dtype
            
            # Flatten spatial dimensions
            if tensor.dim() == 4:  # Image format
                b, c, h, w = tensor.shape
                tensor_flat = tensor.reshape(b, c, h * w)
            else:  # Audio format
                tensor_flat = tensor
            
            # Create projection matrix
            if current_channels < target_channels:
                # Expanding channels
                projection = torch.randn(target_channels, current_channels, device=device, dtype=dtype)
                projection = projection / np.sqrt(current_channels)  # Xavier initialization
                result = torch.matmul(projection, tensor_flat)
            else:
                # Reducing channels
                projection = torch.randn(current_channels, target_channels, device=device, dtype=dtype)
                projection = projection / np.sqrt(target_channels)
                result = torch.matmul(projection.T, tensor_flat)
            
            # Reshape back
            if tensor.dim() == 4:
                result = result.reshape(b, target_channels, h, w)
            else:
                result = result
                
        elif mode == "pad":
            # Pad with zeros or truncate
            if current_channels < target_channels:
                padding = target_channels - current_channels
                if tensor.dim() == 4:
                    result = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, padding), mode='constant', value=0)
                else:
                    result = torch.nn.functional.pad(tensor, (0, 0, 0, padding), mode='constant', value=0)
            else:
                result = tensor[:, :target_channels]
                
        elif mode == "interpolate":
            # Interpolate channels
            if tensor.dim() == 4:
                # For 4D, we need to permute to make channels the last dimension for interpolation
                b, c, h, w = tensor.shape
                tensor_perm = tensor.permute(0, 2, 3, 1).reshape(b * h * w, c, 1)
                result_perm = torch.nn.functional.interpolate(tensor_perm, size=target_channels, mode='linear', align_corners=False)
                result = result_perm.reshape(b, h, w, target_channels).permute(0, 3, 1, 2)
            else:
                # For 3D
                b, c, l = tensor.shape
                tensor_perm = tensor.permute(0, 2, 1).reshape(b * l, c, 1)
                result_perm = torch.nn.functional.interpolate(tensor_perm, size=target_channels, mode='linear', align_corners=False)
                result = result_perm.reshape(b, l, target_channels).permute(0, 2, 1)
                
        elif mode == "tile":
            # Tile or fold channels
            if current_channels < target_channels:
                repeat_factor = (target_channels + current_channels - 1) // current_channels
                result = tensor.repeat(1, repeat_factor, *([1] * (tensor.dim() - 2)))
                result = result[:, :target_channels]
            else:
                # Average fold
                fold_size = current_channels // target_channels
                remainder = current_channels % target_channels
                
                if tensor.dim() == 4:
                    b, c, h, w = tensor.shape
                    # Take the first target_channels*fold_size channels and reshape
                    main_part = tensor[:, :target_channels*fold_size].reshape(b, target_channels, fold_size, h, w)
                    result = main_part.mean(dim=2)
                    
                    # Add remainder channels if any
                    if remainder > 0:
                        extra = tensor[:, -remainder:].mean(dim=1, keepdim=True)
                        result[:, :remainder] += extra.expand(-1, remainder, -1, -1) / 2
                else:
                    b, c, l = tensor.shape
                    main_part = tensor[:, :target_channels*fold_size].reshape(b, target_channels, fold_size, l)
                    result = main_part.mean(dim=2)
                    
                    if remainder > 0:
                        extra = tensor[:, -remainder:].mean(dim=1, keepdim=True)
                        result[:, :remainder] += extra.expand(-1, remainder, -1) / 2
        
        return result


class VAENetworkBending:
    """
    VAE Network Bending - Performs creative modifications on VAE models
    
    This node allows manipulation of VAE encoder/decoder networks including:
    - Weight noise injection in encoder/decoder
    - Asymmetric modifications (encoder vs decoder)
    - Latent space dimension manipulation
    - Channel-specific modifications
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The VAE model to modify"}),
                "operation": ([
                    "add_noise",
                    "scale_weights",
                    "asymmetric_noise",
                    "latent_space_expand",
                    "channel_corruption",
                    "encoder_decoder_swap",
                    "progressive_corruption"
                ], {
                    "default": "add_noise",
                    "tooltip": "VAE-specific bending operation"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Strength of the operation"
                }),
                "target_component": ([
                    "both",
                    "encoder",
                    "decoder",
                    "latent_layers"
                ], {
                    "default": "both",
                    "tooltip": "Which part of the VAE to modify"
                }),
                "preserve_mean": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve weight means to maintain stability"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                }),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "bend_vae"
    CATEGORY = "network_bending/vae"
    OUTPUT_TOOLTIPS = ("Modified VAE with network bending applied",)
    
    def bend_vae(self, vae, operation, intensity, target_component, preserve_mean, seed):
        # Clone the VAE
        import copy
        vae_clone = copy.deepcopy(vae)
        
        # Set random seed
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Get encoder and decoder
        encoder = vae_clone.encoder if hasattr(vae_clone, 'encoder') else vae_clone.first_stage_model.encoder
        decoder = vae_clone.decoder if hasattr(vae_clone, 'decoder') else vae_clone.first_stage_model.decoder
        
        modified_layers = []
        
        # Apply operations
        if operation == "add_noise":
            if target_component in ["both", "encoder"]:
                modified_layers.extend(self._add_noise_to_module(encoder, intensity, preserve_mean, "encoder"))
            if target_component in ["both", "decoder"]:
                modified_layers.extend(self._add_noise_to_module(decoder, intensity, preserve_mean, "decoder"))
                
        elif operation == "scale_weights":
            scale_factor = 1.0 + (intensity - 0.5) * 2
            if target_component in ["both", "encoder"]:
                modified_layers.extend(self._scale_module_weights(encoder, scale_factor, "encoder"))
            if target_component in ["both", "decoder"]:
                modified_layers.extend(self._scale_module_weights(decoder, scale_factor, "decoder"))
                
        elif operation == "asymmetric_noise":
            # Add more noise to decoder than encoder
            if target_component in ["both", "encoder"]:
                modified_layers.extend(self._add_noise_to_module(encoder, intensity * 0.3, preserve_mean, "encoder"))
            if target_component in ["both", "decoder"]:
                modified_layers.extend(self._add_noise_to_module(decoder, intensity, preserve_mean, "decoder"))
                
        elif operation == "channel_corruption":
            if target_component in ["both", "encoder"]:
                modified_layers.extend(self._corrupt_channels(encoder, intensity, "encoder"))
            if target_component in ["both", "decoder"]:
                modified_layers.extend(self._corrupt_channels(decoder, intensity, "decoder"))
                
        elif operation == "progressive_corruption":
            # Gradually increase corruption through layers
            if target_component in ["both", "encoder"]:
                modified_layers.extend(self._progressive_corruption(encoder, intensity, "encoder"))
            if target_component in ["both", "decoder"]:
                modified_layers.extend(self._progressive_corruption(decoder, intensity, "decoder"))
        
        # Send feedback
        try:
            PromptServer.instance.send_sync("network_bending.feedback", {
                "message": f"Applied {operation} to VAE {target_component} with intensity {intensity}",
                "operation": operation,
                "modified_layers": len(modified_layers),
                "target": target_component
            })
        except Exception:
            pass
        
        return (vae_clone,)
    
    def _add_noise_to_module(self, module, intensity, preserve_mean, prefix):
        modified = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                if preserve_mean:
                    noise = torch.randn_like(param.data)
                    noise = noise - noise.mean()  # Zero mean
                    noise = noise * intensity * param.data.std()
                else:
                    noise = torch.randn_like(param.data) * intensity * param.data.std()
                param.data.add_(noise)
                modified.append(f"{prefix}.{name}")
        return modified
    
    def _scale_module_weights(self, module, scale_factor, prefix):
        modified = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.mul_(scale_factor)
                modified.append(f"{prefix}.{name}")
        return modified
    
    def _corrupt_channels(self, module, intensity, prefix):
        modified = []
        for name, param in module.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                # Corrupt specific channels
                num_channels = param.shape[0]
                channels_to_corrupt = int(num_channels * intensity)
                if channels_to_corrupt > 0:
                    corrupt_indices = torch.randperm(num_channels)[:channels_to_corrupt]
                    param.data[corrupt_indices] *= torch.randn(channels_to_corrupt, 1, *([1] * (len(param.shape) - 2)), device=param.device)
                    modified.append(f"{prefix}.{name}")
        return modified
    
    def _progressive_corruption(self, module, max_intensity, prefix):
        modified = []
        layers = list(module.named_parameters())
        num_layers = len(layers)
        
        for idx, (name, param) in enumerate(layers):
            if param.requires_grad:
                # Linear progression of intensity
                layer_intensity = (idx / max(num_layers - 1, 1)) * max_intensity
                noise = torch.randn_like(param.data) * layer_intensity * param.data.std()
                param.data.add_(noise)
                modified.append(f"{prefix}.{name}")
        
        return modified


class VAEMixer:
    """
    Mix two VAE models together with various blending strategies
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_a": ("VAE", {"tooltip": "First VAE model"}),
                "vae_b": ("VAE", {"tooltip": "Second VAE model"}),
                "mix_mode": ([
                    "linear_blend",
                    "encoder_swap",
                    "decoder_swap",
                    "cross_architecture",
                    "frequency_mix",
                    "layer_shuffle"
                ], {
                    "default": "linear_blend",
                    "tooltip": "How to mix the VAE models"
                }),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "mix_vaes"
    CATEGORY = "network_bending/vae"
    
    def mix_vaes(self, vae_a, vae_b, mix_mode, mix_ratio):
        import copy
        result_vae = copy.deepcopy(vae_a)
        
        # Get components
        encoder_a = vae_a.encoder if hasattr(vae_a, 'encoder') else vae_a.first_stage_model.encoder
        decoder_a = vae_a.decoder if hasattr(vae_a, 'decoder') else vae_a.first_stage_model.decoder
        encoder_b = vae_b.encoder if hasattr(vae_b, 'encoder') else vae_b.first_stage_model.encoder
        decoder_b = vae_b.decoder if hasattr(vae_b, 'decoder') else vae_b.first_stage_model.decoder
        
        encoder_result = result_vae.encoder if hasattr(result_vae, 'encoder') else result_vae.first_stage_model.encoder
        decoder_result = result_vae.decoder if hasattr(result_vae, 'decoder') else result_vae.first_stage_model.decoder
        
        if mix_mode == "linear_blend":
            # Blend weights linearly
            self._linear_blend_modules(encoder_a, encoder_b, encoder_result, mix_ratio)
            self._linear_blend_modules(decoder_a, decoder_b, decoder_result, mix_ratio)
            
        elif mix_mode == "encoder_swap":
            # Use encoder from B, decoder from A
            if mix_ratio > 0.5:
                self._copy_module_weights(encoder_b, encoder_result)
                
        elif mix_mode == "decoder_swap":
            # Use decoder from B, encoder from A
            if mix_ratio > 0.5:
                self._copy_module_weights(decoder_b, decoder_result)
                
        elif mix_mode == "cross_architecture":
            # Mix encoder from A with decoder from B at mix_ratio
            self._linear_blend_modules(encoder_a, encoder_b, encoder_result, 1.0 - mix_ratio)
            self._linear_blend_modules(decoder_a, decoder_b, decoder_result, mix_ratio)
            
        elif mix_mode == "layer_shuffle":
            # Randomly select layers from each VAE
            self._shuffle_layers(encoder_a, encoder_b, encoder_result, mix_ratio)
            self._shuffle_layers(decoder_a, decoder_b, decoder_result, mix_ratio)
        
        return (result_vae,)
    
    def _linear_blend_modules(self, module_a, module_b, module_result, ratio):
        params_a = dict(module_a.named_parameters())
        params_b = dict(module_b.named_parameters())
        
        for name, param_result in module_result.named_parameters():
            if name in params_a and name in params_b:
                param_a = params_a[name]
                param_b = params_b[name]
                if param_a.shape == param_b.shape:
                    param_result.data = (1 - ratio) * param_a.data + ratio * param_b.data
    
    def _copy_module_weights(self, source, target):
        source_params = dict(source.named_parameters())
        for name, param in target.named_parameters():
            if name in source_params and source_params[name].shape == param.shape:
                param.data.copy_(source_params[name].data)
    
    def _shuffle_layers(self, module_a, module_b, module_result, ratio):
        params_a = dict(module_a.named_parameters())
        params_b = dict(module_b.named_parameters())
        
        for name, param_result in module_result.named_parameters():
            if name in params_a and name in params_b:
                if torch.rand(1).item() < ratio:
                    param_result.data.copy_(params_b[name].data)
                else:
                    param_result.data.copy_(params_a[name].data)


class VAELatentBending:
    """
    Perform network bending operations directly on VAE latent streams
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent samples to bend"}),
                "operation": ([
                    "add_noise",
                    "channel_swap",
                    "frequency_filter",
                    "spatial_corruption",
                    "value_quantization",
                    "dimension_warp",
                    "temporal_shift"
                ], {
                    "default": "add_noise",
                    "tooltip": "Latent bending operation"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "channel_specific": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply operation to specific channels only"
                }),
                "target_channels": ("STRING", {
                    "default": "0,1,2,3",
                    "tooltip": "Comma-separated channel indices when channel_specific is True"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "bend_latent"
    CATEGORY = "network_bending/vae"
    
    def bend_latent(self, samples, operation, intensity, channel_specific, target_channels):
        latent = samples.copy()
        latent_tensor = latent["samples"].clone()
        
        # Detect if we have 3D (audio) or 4D (image) latents
        is_audio = latent_tensor.dim() == 3
        
        # Parse target channels
        if channel_specific:
            channels = [int(c.strip()) for c in target_channels.split(',')]
        else:
            channels = list(range(latent_tensor.shape[1]))
        
        if operation == "add_noise":
            noise = torch.randn_like(latent_tensor) * intensity
            if channel_specific:
                for c in channels:
                    if c < latent_tensor.shape[1]:
                        latent_tensor[:, c] += noise[:, c]
            else:
                latent_tensor += noise
                
        elif operation == "channel_swap":
            if len(channels) >= 2:
                # Swap random pairs of channels
                for _ in range(int(intensity * len(channels))):
                    idx1, idx2 = random.sample(channels, 2)
                    if idx1 < latent_tensor.shape[1] and idx2 < latent_tensor.shape[1]:
                        latent_tensor[:, [idx1, idx2]] = latent_tensor[:, [idx2, idx1]]
                        
        elif operation == "frequency_filter":
            # Apply frequency domain filtering
            for c in channels:
                if c < latent_tensor.shape[1]:
                    if is_audio:
                        # For audio (3D), use 1D FFT
                        fft = torch.fft.fft(latent_tensor[:, c])
                        
                        # Create frequency mask
                        length = fft.shape[-1]
                        mask = torch.ones_like(fft, dtype=torch.float32)
                        
                        # Low-pass filter (keep low frequencies)
                        cutoff = int((1 - intensity) * length // 2)
                        mask[..., cutoff:length-cutoff] = 0
                        
                        # Apply mask and inverse FFT
                        fft_filtered = fft * mask
                        latent_tensor[:, c] = torch.fft.ifft(fft_filtered).real
                    else:
                        # For image (4D), use 2D FFT
                        fft = torch.fft.fft2(latent_tensor[:, c])
                        
                        # Create frequency mask
                        h, w = fft.shape[-2:]
                        mask = torch.ones_like(fft, dtype=torch.float32)
                        
                        # Low-pass filter (keep low frequencies)
                        cutoff = int((1 - intensity) * min(h, w) // 2)
                        mask[..., cutoff:h-cutoff, :] = 0
                        mask[..., :, cutoff:w-cutoff] = 0
                        
                        # Apply mask and inverse FFT
                        fft_filtered = fft * mask
                        latent_tensor[:, c] = torch.fft.ifft2(fft_filtered).real
                    
        elif operation == "spatial_corruption":
            if is_audio:
                # For audio, corrupt temporal regions
                b, c, length = latent_tensor.shape
                corruption_size = max(1, int(intensity * length * 0.1))
                
                for _ in range(int(intensity * 10)):
                    # Random position
                    pos = random.randint(0, length - corruption_size)
                    
                    # Apply corruption to selected channels
                    for ch in channels:
                        if ch < c:
                            latent_tensor[:, ch, pos:pos+corruption_size] *= random.uniform(0.5, 1.5)
            else:
                # For image, corrupt spatial regions
                b, c, h, w = latent_tensor.shape
                corruption_size = max(1, int(intensity * min(h, w) * 0.3))
                
                for _ in range(int(intensity * 10)):
                    # Random position
                    y = random.randint(0, h - corruption_size)
                    x = random.randint(0, w - corruption_size)
                    
                    # Apply corruption to selected channels
                    for ch in channels:
                        if ch < c:
                            latent_tensor[:, ch, y:y+corruption_size, x:x+corruption_size] *= random.uniform(0.5, 1.5)
                        
        elif operation == "value_quantization":
            # Quantize values (works for both 3D and 4D)
            levels = max(2, int((1 - intensity) * 256))
            for c in channels:
                if c < latent_tensor.shape[1]:
                    channel = latent_tensor[:, c]
                    min_val = channel.min()
                    max_val = channel.max()
                    
                    # Normalize, quantize, denormalize
                    normalized = (channel - min_val) / (max_val - min_val + 1e-8)
                    quantized = torch.round(normalized * (levels - 1)) / (levels - 1)
                    latent_tensor[:, c] = quantized * (max_val - min_val) + min_val
                    
        elif operation == "dimension_warp":
            if is_audio:
                # For audio, warp time dimension using 1D interpolation
                b, c, length = latent_tensor.shape
                device = latent_tensor.device
                
                # Create original coordinate grid (0 to length-1)
                x_orig = torch.arange(length, dtype=torch.float32, device=device)
                
                # Create warped coordinates
                x_normalized = torch.linspace(-1, 1, length, device=device)
                warp_offset = intensity * 0.1 * torch.sin(x_normalized * np.pi * 4)
                
                # Convert back to sample indices
                x_warped = x_orig + warp_offset * length * 0.1
                x_warped = torch.clamp(x_warped, 0, length - 1)
                
                # Apply warping to selected channels using 1D interpolation
                for ch in channels:
                    if ch < c:
                        # Process all batches at once
                        channel_data = latent_tensor[:, ch]  # [B, L]
                        
                        # Create interpolation indices and weights
                        indices_left = x_warped.long()
                        indices_right = torch.clamp(indices_left + 1, 0, length - 1)
                        weights = x_warped - indices_left.float()
                        
                        # Gather values at left and right indices for all batches
                        left_values = torch.gather(channel_data, 1, indices_left.unsqueeze(0).expand(b, -1))
                        right_values = torch.gather(channel_data, 1, indices_right.unsqueeze(0).expand(b, -1))
                        
                        # Perform interpolation
                        warped_values = left_values * (1 - weights) + right_values * weights
                        
                        latent_tensor[:, ch] = warped_values
            else:
                # For image, warp spatial dimensions
                b, c, h, w = latent_tensor.shape
                
                # Create coordinate grids
                y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
                
                # Add sinusoidal warping
                warp_y = y + intensity * 0.1 * torch.sin(x * np.pi * 4)
                warp_x = x + intensity * 0.1 * torch.sin(y * np.pi * 4)
                
                # Stack and reshape for grid_sample
                grid = torch.stack([warp_x, warp_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
                
                # Apply warping to selected channels
                for ch in channels:
                    if ch < c:
                        channel = latent_tensor[:, ch:ch+1]
                        warped = torch.nn.functional.grid_sample(channel, grid.to(channel.device), align_corners=False)
                        latent_tensor[:, ch:ch+1] = warped
        
        elif operation == "temporal_shift":
            # This operation is specifically for time-based data
            if is_audio:
                shift_amount = int(intensity * latent_tensor.shape[-1] * 0.1)
                if shift_amount > 0:
                    for ch in channels:
                        if ch < latent_tensor.shape[1]:
                            # Circular shift
                            latent_tensor[:, ch] = torch.roll(latent_tensor[:, ch], shifts=shift_amount, dims=-1)
            else:
                # For images, apply a diagonal shift effect
                b, c, h, w = latent_tensor.shape
                shift_pixels = int(intensity * min(h, w) * 0.1)
                if shift_pixels > 0:
                    for ch in channels:
                        if ch < c:
                            # Diagonal roll
                            latent_tensor[:, ch] = torch.roll(latent_tensor[:, ch], shifts=(shift_pixels, shift_pixels), dims=(-2, -1))
        
        latent["samples"] = latent_tensor
        
        return (latent,)


class VAEChannelManipulator:
    """
    Advanced channel manipulation for VAE latents
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Input latent samples"}),
                "operation": ([
                    "channel_attention",
                    "cross_channel_mixing",
                    "channel_dropout",
                    "channel_amplification",
                    "channel_rotation",
                    "channel_statistics_swap"
                ], {
                    "default": "channel_attention",
                }),
                "intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "preserve_energy": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve overall signal energy"
                }),
            },
            "optional": {
                "reference_latent": ("LATENT", {
                    "tooltip": "Reference latent for operations like statistics swap"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "manipulate_channels"
    CATEGORY = "network_bending/vae"
    
    def manipulate_channels(self, samples, operation, intensity, preserve_energy, reference_latent=None):
        latent = samples.copy()
        latent_tensor = latent["samples"].clone()
        
        # Detect if we have 3D (audio) or 4D (image) latents
        is_audio = latent_tensor.dim() == 3
        
        original_energy = torch.norm(latent_tensor)
        
        if operation == "channel_attention":
            # Apply self-attention across channels
            if is_audio:
                b, c, length = latent_tensor.shape
                # Reshape for attention
                x = latent_tensor.reshape(b, c, length)
                
                # Simple attention mechanism
                attention_scores = torch.bmm(x, x.transpose(1, 2)) / np.sqrt(length)
                attention_weights = torch.softmax(attention_scores * intensity, dim=-1)
                
                # Apply attention
                attended = torch.bmm(attention_weights, x)
                latent_tensor = attended
            else:
                b, c, h, w = latent_tensor.shape
                
                # Reshape for attention
                x = latent_tensor.reshape(b, c, h * w)
                
                # Simple attention mechanism
                attention_scores = torch.bmm(x, x.transpose(1, 2)) / np.sqrt(h * w)
                attention_weights = torch.softmax(attention_scores * intensity, dim=-1)
                
                # Apply attention
                attended = torch.bmm(attention_weights, x)
                latent_tensor = attended.reshape(b, c, h, w)
            
        elif operation == "cross_channel_mixing":
            # Mix information across channels
            if is_audio:
                b, c, length = latent_tensor.shape
            else:
                b, c, h, w = latent_tensor.shape
            
            # Create mixing matrix
            mix_matrix = torch.eye(c) * (1 - intensity) + torch.ones(c, c) * (intensity / c)
            mix_matrix = mix_matrix.to(latent_tensor.device)
            
            # Apply mixing
            reshaped = latent_tensor.reshape(b, c, -1)
            mixed = torch.bmm(mix_matrix.unsqueeze(0).repeat(b, 1, 1), reshaped)
            
            if is_audio:
                latent_tensor = mixed
            else:
                latent_tensor = mixed.reshape(b, c, h, w)
            
        elif operation == "channel_dropout":
            # Randomly drop channels
            dropout_mask = torch.rand(latent_tensor.shape[1]) > intensity
            dropout_mask = dropout_mask.to(latent_tensor.device)
            
            if is_audio:
                latent_tensor *= dropout_mask.view(1, -1, 1)
            else:
                latent_tensor *= dropout_mask.view(1, -1, 1, 1)
            
        elif operation == "channel_amplification":
            # Selectively amplify certain channels
            if is_audio:
                b, c, length = latent_tensor.shape
                # Calculate channel importance (based on variance)
                channel_var = latent_tensor.var(dim=(0, 2))
            else:
                b, c, h, w = latent_tensor.shape
                # Calculate channel importance (based on variance)
                channel_var = latent_tensor.var(dim=(0, 2, 3))
            
            importance = torch.softmax(channel_var, dim=0)
            
            # Amplify based on importance
            amplification = 1 + intensity * importance * 2
            
            if is_audio:
                latent_tensor *= amplification.view(1, -1, 1)
            else:
                latent_tensor *= amplification.view(1, -1, 1, 1)
            
        elif operation == "channel_rotation":
            # Rotate channels in feature space
            c = latent_tensor.shape[1]
            
            if c >= 2:
                # Create rotation matrix
                angle = intensity * np.pi
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                # Apply pairwise rotations
                for i in range(0, c - 1, 2):
                    ch1 = latent_tensor[:, i].clone()
                    ch2 = latent_tensor[:, i + 1].clone()
                    
                    latent_tensor[:, i] = cos_a * ch1 - sin_a * ch2
                    latent_tensor[:, i + 1] = sin_a * ch1 + cos_a * ch2
                    
        elif operation == "channel_statistics_swap":
            if reference_latent is not None:
                ref_tensor = reference_latent["samples"]
                
                # Swap statistics between latents
                for c in range(min(latent_tensor.shape[1], ref_tensor.shape[1])):
                    if torch.rand(1).item() < intensity:
                        # Get statistics
                        mean1 = latent_tensor[:, c].mean()
                        std1 = latent_tensor[:, c].std()
                        mean2 = ref_tensor[:, c].mean()
                        std2 = ref_tensor[:, c].std()
                        
                        # Normalize and rescale
                        latent_tensor[:, c] = (latent_tensor[:, c] - mean1) / (std1 + 1e-8) * std2 + mean2
        
        # Preserve energy if requested
        if preserve_energy:
            current_energy = torch.norm(latent_tensor)
            if current_energy > 0:
                latent_tensor *= original_energy / current_energy
        
        latent["samples"] = latent_tensor
        
        return (latent,)


# Register all nodes
NODE_CLASS_MAPPINGS = {
    "NetworkBending": NetworkBending,
    "NetworkBendingAdvanced": NetworkBendingAdvanced,
    "ModelMixer": ModelMixer,
    "InvertedPruning": InvertedPruning,
    "LatentFormatConverter": LatentFormatConverter,
    "VAENetworkBending": VAENetworkBending,
    "VAEMixer": VAEMixer,
    "VAELatentBending": VAELatentBending,
    "VAEChannelManipulator": VAEChannelManipulator,
}

# Add audio nodes to the mappings
NODE_CLASS_MAPPINGS.update(AUDIO_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "NetworkBending": "Network Bending",
    "NetworkBendingAdvanced": "Network Bending (Advanced)",
    "ModelMixer": "Model Mixer",
    "InvertedPruning": "Inverted Pruning",
    "LatentFormatConverter": "Latent Format Converter",
    "VAENetworkBending": "VAE Network Bending",
    "VAEMixer": "VAE Mixer",
    "VAELatentBending": "VAE Latent Bending",
    "VAEChannelManipulator": "VAE Channel Manipulator",
}

# Add audio node display names
NODE_DISPLAY_NAME_MAPPINGS.update(AUDIO_NODE_DISPLAY_NAME_MAPPINGS)

# Export web directory for UI components
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']