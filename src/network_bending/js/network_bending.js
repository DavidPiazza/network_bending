// Use absolute paths as ComfyUI serves these from /scripts
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Register the network bending extension
app.registerExtension({
    name: "network_bending",
    
    async setup() {
        // Listen for feedback messages from the server
        api.addEventListener("network_bending.feedback", (event) => {
            const data = event.detail;
            
            // Create a more informative message
            let message = data.message;
            if (data.total_layers > 10) {
                message += ` (showing first 10 of ${data.total_layers} layers)`;
            }
            
            // Display modified layers if available
            if (data.modified_layers && data.modified_layers.length > 0) {
                message += "\n\nModified layers:\n" + data.modified_layers.join("\n");
            }
            
            // Show the feedback to the user
            app.ui.dialog.show(message);
        });
    },
    
    // Add custom widgets to network bending nodes
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NetworkBending" || nodeData.name === "NetworkBendingAdvanced") {
            // Add a custom widget for layer pattern suggestions
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Add layer pattern helper button
                const widget = this.widgets.find(w => w.name === "target_layers");
                if (widget) {
                    // Add a button widget after the target_layers input
                    const buttonWidget = this.addWidget("button", "Layer Patterns Help", null, () => {
                        const helpText = `Layer Pattern Examples:
• all - Modify all layers
• conv - Convolutional layers
• attention - Attention layers
• linear - Linear/Dense layers
• norm - Normalization layers
• embedding - Embedding layers
• output - Output layers
• layer.0 - Specific layer by name
• block_1 - Specific block
• down - Downsampling layers
• up - Upsampling layers

You can combine patterns with commas:
• conv,attention - Both conv and attention
• linear,output - Linear and output layers`;
                        
                        app.ui.dialog.show(helpText);
                    });
                }
                
                // Add operation descriptions
                const operationWidget = this.widgets.find(w => w.name === "operation");
                if (operationWidget) {
                    const descriptionWidget = this.addWidget("button", "Operation Info", null, () => {
                        const descriptions = {
                            "add_noise": "Adds random Gaussian noise to weights. Lower intensity = subtle changes, higher = more corruption.",
                            "scale_weights": "Scales all weights by a factor. 0.5 intensity = no change, <0.5 = decrease, >0.5 = increase.",
                            "prune_weights": "Sets small weights to zero. Higher intensity = more aggressive pruning.",
                            "randomize_weights": "Replaces a portion of weights with random values. Intensity controls percentage replaced.",
                            "smooth_weights": "Applies spatial smoothing to weight matrices, creating a blurring effect.",
                            "quantize_weights": "Reduces weight precision to discrete levels. Lower intensity = fewer levels.",
                            "layer_swap": "Swaps weights between similar layers in the network.",
                            "activation_replace": "Replaces activation functions with alternatives.",
                            "weight_transpose": "Transposes weight matrices where applicable.",
                            "channel_shuffle": "Randomly shuffles channels in convolutional layers.",
                            "frequency_filter": "Applies frequency domain filtering to weights.",
                            "weight_clustering": "Groups similar weights together using clustering."
                        };
                        
                        const currentOp = operationWidget.value;
                        const desc = descriptions[currentOp] || "No description available.";
                        
                        app.ui.dialog.show(`${currentOp}:\n\n${desc}`);
                    });
                }
                
                return result;
            };
        }
        
        // Add mix mode descriptions for ModelMixer
        if (nodeData.name === "ModelMixer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                const mixModeWidget = this.widgets.find(w => w.name === "mix_mode");
                if (mixModeWidget) {
                    const descriptionWidget = this.addWidget("button", "Mix Mode Info", null, () => {
                        const descriptions = {
                            "linear_interpolation": "Simple weighted average of all weights. Most stable mixing method.",
                            "weighted_sum": "Weighted sum with normalization. Can emphasize certain features.",
                            "layer_wise_mix": "Different mix ratios for different layer types.",
                            "frequency_blend": "Mixes models in frequency domain for smoother blending.",
                            "random_mix": "Randomly selects weights from either model."
                        };
                        
                        const currentMode = mixModeWidget.value;
                        const desc = descriptions[currentMode] || "No description available.";
                        
                        app.ui.dialog.show(`${currentMode}:\n\n${desc}`);
                    });
                }
                
                return result;
            };
        }
        
        // Add descriptions for LatentFormatConverter
        if (nodeData.name === "LatentFormatConverter") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Add conversion mode help
                const conversionModeWidget = this.widgets.find(w => w.name === "conversion_mode");
                if (conversionModeWidget) {
                    const modeHelpWidget = this.addWidget("button", "Conversion Mode Info", null, () => {
                        const descriptions = {
                            "audio_to_image": "Converts 3D audio latents [B,C,L] to 4D image latents [B,C,H,W]. Use when feeding audio-generated latents to image models.",
                            "image_to_audio": "Converts 4D image latents [B,C,H,W] to 3D audio latents [B,C,L]. Use when feeding image-generated latents to audio models.",
                            "auto_detect": "Automatically detects format based on tensor dimensions and converts accordingly."
                        };
                        
                        const currentMode = conversionModeWidget.value;
                        const desc = descriptions[currentMode] || "No description available.";
                        
                        app.ui.dialog.show(`${currentMode}:\n\n${desc}`);
                    });
                }
                
                // Add reshape method help
                const reshapeMethodWidget = this.widgets.find(w => w.name === "reshape_method");
                if (reshapeMethodWidget) {
                    const methodHelpWidget = this.addWidget("button", "Reshape Method Info", null, () => {
                        const descriptions = {
                            "reshape": "Direct reshape with padding/truncation. Fast but may lose spatial relationships.",
                            "interpolate": "Reshapes to intermediate size then interpolates. Better for preserving features.",
                            "fold": "Uses fold/unfold operations. Best for preserving local patterns and relationships.",
                            "tile": "Tiles the input to fill output space. Good for repeating patterns."
                        };
                        
                        const currentMethod = reshapeMethodWidget.value;
                        const desc = descriptions[currentMethod] || "No description available.";
                        
                        app.ui.dialog.show(`${currentMethod}:\n\n${desc}\n\nFor audio→image: fold often works best\nFor image→audio: reshape is usually sufficient`);
                    });
                }
                
                // Add channel mode help
                const channelModeWidget = this.widgets.find(w => w.name === "channel_mode");
                if (channelModeWidget) {
                    const channelHelpWidget = this.addWidget("button", "Channel Mode Info", null, () => {
                        const descriptions = {
                            "project": "Uses random projection matrix to convert channels. Best for maintaining information.",
                            "pad": "Pads with zeros or truncates channels. Simple but may lose information.",
                            "interpolate": "Interpolates between channels. Smooth transition but may blur features.",
                            "tile": "Tiles or averages channels. Good for repeating patterns."
                        };
                        
                        const currentMode = channelModeWidget.value;
                        const desc = descriptions[currentMode] || "No description available.";
                        
                        app.ui.dialog.show(`${currentMode}:\n\n${desc}\n\nCommon channel counts:\n• SD/SDXL: 4 channels\n• StableAudio: 64 channels\n• Set to -1 to keep original`);
                    });
                }
                
                return result;
            };
        }
        
        // Add descriptions for VAENetworkBending
        if (nodeData.name === "VAENetworkBending") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                const operationWidget = this.widgets.find(w => w.name === "operation");
                if (operationWidget) {
                    const descriptionWidget = this.addWidget("button", "VAE Operation Info", null, () => {
                        const descriptions = {
                            "add_noise": "Adds Gaussian noise to VAE encoder/decoder weights. Can create artistic artifacts.",
                            "scale_weights": "Scales VAE weights. Can make encoding/decoding more or less aggressive.",
                            "asymmetric_noise": "Adds more noise to decoder than encoder. Creates decoding artifacts while preserving encoding.",
                            "latent_space_expand": "Modifies latent space dimensions. Can create interesting distortions.",
                            "channel_corruption": "Corrupts specific channels in the VAE. Creates color/feature artifacts.",
                            "encoder_decoder_swap": "Experimental: attempts to swap encoder/decoder roles.",
                            "progressive_corruption": "Gradually increases corruption through VAE layers. Creates gradient effects."
                        };
                        
                        const currentOp = operationWidget.value;
                        const desc = descriptions[currentOp] || "No description available.";
                        
                        app.ui.dialog.show(`VAE ${currentOp}:\n\n${desc}`);
                    });
                }
                
                // Add target component help
                const targetWidget = this.widgets.find(w => w.name === "target_component");
                if (targetWidget) {
                    const targetHelpWidget = this.addWidget("button", "Target Component Info", null, () => {
                        app.ui.dialog.show(`Target Component:\n\n• both - Modify both encoder and decoder\n• encoder - Only modify the encoder (image→latent)\n• decoder - Only modify the decoder (latent→image)\n• latent_layers - Target layers that process the latent space\n\nEncoder modifications affect how images are compressed.\nDecoder modifications affect how latents are reconstructed.`);
                    });
                }
                
                return result;
            };
        }
        
        // Add descriptions for VAEMixer
        if (nodeData.name === "VAEMixer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                const mixModeWidget = this.widgets.find(w => w.name === "mix_mode");
                if (mixModeWidget) {
                    const descriptionWidget = this.addWidget("button", "VAE Mix Mode Info", null, () => {
                        const descriptions = {
                            "linear_blend": "Blends both encoder and decoder weights linearly. Most stable option.",
                            "encoder_swap": "Uses encoder from VAE B with decoder from VAE A.",
                            "decoder_swap": "Uses decoder from VAE B with encoder from VAE A.",
                            "cross_architecture": "Mixes encoder more from A, decoder more from B (or vice versa).",
                            "frequency_mix": "Mixes VAEs in frequency domain for smoother blending.",
                            "layer_shuffle": "Randomly selects layers from each VAE. Creates hybrid architectures."
                        };
                        
                        const currentMode = mixModeWidget.value;
                        const desc = descriptions[currentMode] || "No description available.";
                        
                        app.ui.dialog.show(`${currentMode}:\n\n${desc}`);
                    });
                }
                
                return result;
            };
        }
        
        // Add descriptions for VAELatentBending
        if (nodeData.name === "VAELatentBending") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                const operationWidget = this.widgets.find(w => w.name === "operation");
                if (operationWidget) {
                    const descriptionWidget = this.addWidget("button", "Latent Operation Info", null, () => {
                        const descriptions = {
                            "add_noise": "Adds noise directly to latent values. Creates texture variations.",
                            "channel_swap": "Swaps latent channels. Can create color shifts and inversions.",
                            "frequency_filter": "Applies frequency filtering to latents. Can smooth or sharpen features.",
                            "spatial_corruption": "Corrupts spatial regions in latents. Creates localized artifacts.",
                            "value_quantization": "Quantizes latent values. Creates posterization-like effects.",
                            "dimension_warp": "Warps spatial dimensions. Creates distortion effects.",
                            "temporal_shift": "Shifts values temporally. Useful for animation effects."
                        };
                        
                        const currentOp = operationWidget.value;
                        const desc = descriptions[currentOp] || "No description available.";
                        
                        app.ui.dialog.show(`Latent ${currentOp}:\n\n${desc}`);
                    });
                }
                
                return result;
            };
        }
        
        // Add descriptions for VAEChannelManipulator
        if (nodeData.name === "VAEChannelManipulator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                const operationWidget = this.widgets.find(w => w.name === "operation");
                if (operationWidget) {
                    const descriptionWidget = this.addWidget("button", "Channel Operation Info", null, () => {
                        const descriptions = {
                            "channel_attention": "Applies self-attention across channels. Enhances channel relationships.",
                            "cross_channel_mixing": "Mixes information between channels. Creates color blending effects.",
                            "channel_dropout": "Randomly drops channels. Can create missing color effects.",
                            "channel_amplification": "Amplifies important channels based on variance. Enhances dominant features.",
                            "channel_rotation": "Rotates channels in feature space. Creates color shifts.",
                            "channel_statistics_swap": "Swaps channel statistics with reference. Style transfer effect."
                        };
                        
                        const currentOp = operationWidget.value;
                        const desc = descriptions[currentOp] || "No description available.";
                        
                        app.ui.dialog.show(`Channel ${currentOp}:\n\n${desc}\n\nTip: Enable 'preserve_energy' to maintain overall signal strength.`);
                    });
                }
                
                return result;
            };
        }
    },
    
    // Add node color coding
    nodeCreated(node) {
        if (node.comfyClass === "NetworkBending") {
            node.color = "#432";  // Dark red color for basic network bending
            node.bgcolor = "#653";
        } else if (node.comfyClass === "NetworkBendingAdvanced") {
            node.color = "#234";  // Dark blue for advanced
            node.bgcolor = "#356";
        } else if (node.comfyClass === "ModelMixer") {
            node.color = "#243";  // Dark teal for mixer
            node.bgcolor = "#365";
        } else if (node.comfyClass === "LatentFormatConverter") {
            node.color = "#324";  // Dark purple for converter
            node.bgcolor = "#546";
        } else if (node.comfyClass === "VAENetworkBending") {
            node.color = "#423";  // Dark orange for VAE bending
            node.bgcolor = "#645";
        } else if (node.comfyClass === "VAEMixer") {
            node.color = "#342";  // Dark green for VAE mixer
            node.bgcolor = "#564";
        } else if (node.comfyClass === "VAELatentBending") {
            node.color = "#234";  // Dark cyan for latent bending
            node.bgcolor = "#456";
        } else if (node.comfyClass === "VAEChannelManipulator") {
            node.color = "#324";  // Dark magenta for channel manipulation
            node.bgcolor = "#546";
        }
    }
}); 