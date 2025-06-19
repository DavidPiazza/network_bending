import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

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
        }
    }
}); 