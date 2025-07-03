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

### 🔄 Latent Format Converter Node
Convert between audio and image latent formats to enable cross-modal workflows:

- **Audio to Image**: Convert 3D audio latents [B,C,L] to 4D image latents [B,C,H,W]
- **Image to Audio**: Convert 4D image latents [B,C,H,W] to 3D audio latents [B,C,L]
- **Auto Detect**: Automatically detect format and convert appropriately
- **Channel Conversion**: Convert between different channel counts (e.g., 4→64 for SD→StableAudio)
- **Multiple Methods**: Choose from reshape, interpolate, fold, or tile methods for both spatial and channel dims

### 🌀 VAE Network Bending Node
Apply network bending operations specifically to VAE models:

- **Add Noise**: Inject noise into encoder/decoder weights
- **Scale Weights**: Scale VAE weights for more/less aggressive encoding
- **Asymmetric Noise**: Different noise levels for encoder vs decoder
- **Channel Corruption**: Corrupt specific channels in the VAE
- **Progressive Corruption**: Gradually increase corruption through layers
- **Target Components**: Choose encoder, decoder, or both

### 🎯 VAE Mixer Node
Mix two VAE models with specialized blending strategies:

- **Linear Blend**: Blend both encoder and decoder linearly
- **Encoder Swap**: Use encoder from one VAE with decoder from another
- **Decoder Swap**: Use decoder from one VAE with encoder from another
- **Cross Architecture**: Mix encoder/decoder with different ratios
- **Layer Shuffle**: Randomly select layers from each VAE

### 💫 VAE Latent Bending Node
Perform operations directly on VAE latent streams:

- **Add Noise**: Add noise to latent values for texture variations
- **Channel Swap**: Swap latent channels for color shifts
- **Frequency Filter**: Apply frequency domain filtering
- **Spatial Corruption**: Corrupt specific spatial regions
- **Value Quantization**: Create posterization effects
- **Dimension Warp**: Warp spatial dimensions for distortions

### 🔮 VAE Channel Manipulator Node
Advanced channel-level operations on VAE latents:

- **Channel Attention**: Apply self-attention across channels
- **Cross Channel Mixing**: Mix information between channels
- **Channel Dropout**: Randomly drop channels
- **Channel Amplification**: Amplify important channels
- **Channel Rotation**: Rotate channels in feature space
- **Channel Statistics Swap**: Swap statistics with reference latent

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

### Latent Format Conversion

1. When you encounter dimension mismatch errors between audio and image models
2. Add a "Latent Format Converter" node between the latent source and destination
3. Set conversion mode (or use auto_detect)
4. Choose appropriate reshape method (fold works well for audio→image)
5. Adjust target dimensions if converting to image format
6. Set target_channels to match your model (4 for SD/SDXL, 64 for StableAudio)
7. Use "project" channel mode for best results when converting channels

### VAE Network Bending

1. Load a VAE or get one from a checkpoint loader
2. Add a "VAE Network Bending" node
3. Connect the VAE to the input
4. Choose operation and target component (encoder/decoder/both)
5. Set intensity and preserve_mean option
6. Use the bent VAE for encoding/decoding

### VAE Latent Manipulation

1. Encode an image to latent space using VAE
2. Add "VAE Latent Bending" or "VAE Channel Manipulator" nodes
3. Connect latents through the manipulation nodes
4. Experiment with different operations and intensities
5. Decode back to image to see the effects

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

### Latent Format Converter Node

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| conversion_mode | Select | Conversion direction | audio_to_image, image_to_audio, auto_detect |
| target_height | Int | Height for image format | 8 - 512 (steps of 8) |
| target_width | Int | Width for image format | 8 - 512 (steps of 8) |
| reshape_method | Select | Method for reshaping | reshape, interpolate, fold, tile |
| target_channels | Int | Target channel count | -1 to 256 (-1 keeps original) |
| channel_mode | Select | Channel conversion method | project, pad, interpolate, tile |

### VAE Network Bending Node

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| operation | Select | VAE bending operation | See VAE operations list |
| intensity | Float | Strength of the operation | 0.0 - 1.0 |
| target_component | Select | Part of VAE to modify | both, encoder, decoder, latent_layers |
| preserve_mean | Boolean | Preserve weight means | True/False |
| seed | Int | Random seed | -1 to 2^63-1 |

### VAE Mixer Node

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| mix_mode | Select | VAE blending mode | See VAE mix modes list |
| mix_ratio | Float | Blend ratio | 0.0 - 1.0 |

### VAE Latent Bending Node

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| operation | Select | Latent operation | See latent operations list |
| intensity | Float | Operation strength | 0.0 - 1.0 |
| channel_specific | Boolean | Target specific channels | True/False |
| target_channels | String | Channel indices | Comma-separated (e.g., "0,1,2,3") |

### VAE Channel Manipulator Node

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| operation | Select | Channel operation | See channel operations list |
| intensity | Float | Operation strength | 0.0 - 1.0 |
| preserve_energy | Boolean | Preserve signal energy | True/False |
| reference_latent | Latent | Reference for some ops | Optional latent input |

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

### Cross-Modal Generation
- Use Latent Format Converter with `auto_detect`
- Reshape Method: `fold` (for audio→image)
- Enables using audio latents with image models

### VAE Glitch Effects
- VAE Network Bending: `channel_corruption`
- Target Component: `decoder`
- Intensity: `0.2`
- Creates colorful decoding artifacts

### Dreamy VAE Processing
- VAE Latent Bending: `frequency_filter`
- Intensity: `0.4`
- Creates soft, dream-like images

### Style Transfer via VAE
- VAE Channel Manipulator: `channel_statistics_swap`
- Intensity: `0.7`
- Use reference latent from style image
- Transfers color characteristics between images

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

