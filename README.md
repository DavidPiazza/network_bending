# Network Bending for ComfyUI

A custom node pack for ComfyUI that enables creative manipulation and "bending" of neural network models. Perform various operations on loaded model checkpoints to create unique and experimental effects.

## Features

### 🎛️ Network Bending Node
The main node for applying various network bending operations:

- **Add Noise**: Inject Gaussian noise into model weights
- **Scale Weights**: Scale model weights up or down
- **Prune Weights**: Remove small weights (sparsification)
- **Randomize Weights**: Replace portions of weights with random values
- **Smooth Weights**: Apply spatial smoothing to weight matrices
- **Quantize Weights**: Reduce weight precision to discrete levels

### 🔧 Network Bending Advanced Node
Advanced operations for more complex manipulations:

- **Layer Swap**: Swap weights between similar layers
- **Activation Replace**: Replace activation functions
- **Weight Transpose**: Transpose weight matrices
- **Channel Shuffle**: Randomly shuffle channels in conv layers
- **Frequency Filter**: Apply frequency domain filtering
- **Weight Clustering**: Group similar weights together

### 🎨 Model Mixer Node
Mix two models together with various blending modes:

- **Linear Interpolation**: Simple weighted average of weights
- **Weighted Sum**: Weighted sum with normalization
- **Layer-wise Mix**: Different mix ratios for different layers
- **Frequency Blend**: Mix models in frequency domain
- **Random Mix**: Randomly select weights from either model

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/network_bending.git
```

2. Install dependencies (if any):
```bash
cd network_bending
pip install -r requirements.txt  # if requirements.txt exists
```

3. Restart ComfyUI

## Usage

### Basic Network Bending

1. Load a model checkpoint using the standard ComfyUI loader nodes
2. Add a "Network Bending" node to your workflow
3. Connect the MODEL output to the Network Bending input
4. Configure parameters:
   - **Operation**: Select the type of bending operation
   - **Intensity**: Control the strength (0.0 = no effect, 1.0 = maximum)
   - **Target Layers**: Specify which layers to modify (e.g., "conv", "attention", or "all")
   - **Seed**: Set for reproducible results (-1 for random)

### Layer Targeting

You can target specific layers using patterns:
- `all` - Modify all layers
- `conv` - Target convolutional layers
- `attention` - Target attention layers
- `linear` - Target linear/dense layers
- `norm` - Target normalization layers
- `embedding` - Target embedding layers
- Combine patterns with commas: `conv,attention`

### Model Mixing

1. Load two model checkpoints
2. Add a "Model Mixer" node
3. Connect both models to the mixer inputs
4. Set the mix mode and ratio
5. Connect the output to your generation pipeline

## Parameters

### Network Bending Node

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| operation | Select | Type of bending operation | See operations list |
| intensity | Float | Strength of the operation | 0.0 - 1.0 |
| target_layers | String | Layer patterns to target | Comma-separated patterns |
| seed | Int | Random seed | -1 to 2^63-1 |

### Model Mixer Node

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| mix_mode | Select | Blending mode | See mix modes list |
| mix_ratio | Float | Blend ratio | 0.0 - 1.0 |

## UI Features

- **Visual Feedback**: Operations display information about modified layers
- **Help Buttons**: Click "Layer Patterns Help" or "Operation Info" for detailed information
- **Color Coding**: Different node types have distinct colors for easy identification
- **Real-time Updates**: See which layers were modified after each operation

## Examples

### Subtle Model Variation
- Operation: `add_noise`
- Intensity: `0.05`
- Target Layers: `attention`
- Creates subtle variations while preserving model behavior

### Aggressive Glitch Art
- Operation: `randomize_weights`
- Intensity: `0.3`
- Target Layers: `conv`
- Creates glitchy, unpredictable outputs

### Model Hybridization
- Use Model Mixer with `linear_interpolation`
- Mix Ratio: `0.5`
- Creates a balanced hybrid of two models

## Tips

1. **Start with low intensity values** (0.01-0.1) and increase gradually
2. **Target specific layers** for more controlled effects
3. **Use seeds** for reproducible results
4. **Combine operations** by chaining multiple bending nodes
5. **Save interesting results** as new model checkpoints

## Troubleshooting

- **No effect visible**: Increase intensity or check target_layers pattern
- **Model breaks completely**: Reduce intensity or target fewer layers
- **Out of memory**: Operations clone the model, ensure sufficient VRAM

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ComfyUI team for the excellent framework
- Inspired by circuit bending and glitch art techniques

