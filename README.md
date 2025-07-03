# Network Bending for ComfyUI

A comprehensive custom node pack for ComfyUI that enables creative manipulation and "bending" of neural network models, including both image and audio models. Inspired by circuit bending techniques in electronic music, this package allows you to creatively corrupt, modify, and blend neural networks to create unique and experimental effects.

## Overview

Network Bending brings the concept of circuit bending to AI models. Just as circuit bending involves creatively short-circuiting electronic devices to create new sounds, network bending involves modifying neural network weights and architectures to create unexpected and artistic outputs. This package supports both Stable Diffusion image models and Stable Audio models.

## Project Structure

```
network_bending/
├── src/network_bending/
│   ├── nodes.py                    # Core network bending nodes
│   ├── audio_nodes/               # Audio-specific nodes
│   │   ├── audio_latent_nodes.py # Audio VAE and latent manipulation
│   │   ├── audio_style_transfer.py # Audio style transfer capabilities
│   │   └── README.md              # Audio nodes documentation
│   ├── audio_workflows/           # Pre-built audio workflows
│   └── js/                        # UI components
│       └── network_bending.js     # Custom UI with visual feedback
├── examples/
│   └── basic_workflow.json        # Example ComfyUI workflow
├── tests/                         # Test suite
├── requirements-audio.txt         # Audio processing dependencies
└── pyproject.toml                 # Modern Python packaging
```

## Features

### Core Network Bending Nodes

#### Network Bending Node
The main node for applying various network bending operations to image models:

- **Add Noise**: Inject Gaussian noise into model weights for subtle variations
- **Scale Weights**: Scale model weights up or down to amplify or dampen features
- **Prune Weights**: Remove small weights for sparsification and efficiency
- **Randomize Weights**: Replace portions of weights with random values for chaos
- **Smooth Weights**: Apply spatial smoothing to weight matrices for softer outputs
- **Quantize Weights**: Reduce weight precision to discrete levels for lo-fi effects

#### Network Bending Advanced Node
Advanced operations for more complex manipulations:

- **Layer Swap**: Swap weights between similar layers
- **Activation Replace**: Replace activation functions
- **Weight Transpose**: Transpose weight matrices
- **Channel Shuffle**: Randomly shuffle channels in convolutional layers
- **Frequency Filter**: Apply frequency domain filtering
- **Weight Clustering**: Group similar weights together

#### Model Mixer Node
Mix two models together with various blending modes:

- **Linear Interpolation**: Simple weighted average of weights
- **Weighted Sum**: Weighted sum with normalization
- **Layer-wise Mix**: Different mix ratios for different layers
- **Frequency Blend**: Mix models in frequency domain
- **Random Mix**: Randomly select weights from either model

### Audio Processing Nodes

The package includes comprehensive audio processing capabilities for Stable Audio models:

#### Audio VAE Nodes
- **Audio VAE Encode**: Encode audio waveforms into latent space
- **Audio VAE Decode**: Decode latent representations back to audio

#### Audio Latent Manipulation
- **Audio Latent Interpolate**: Smoothly interpolate between audio latents
- **Audio Latent Blend**: Blend multiple audio latents with various modes
- **Audio Latent Manipulator**: Direct manipulation of audio latent features
- **Audio Feature Extractor**: Extract specific features from audio latents

#### Audio Style Transfer
- **Audio Style Transfer**: Transfer style from one audio to another
- **Audio Latent Guidance**: Guide generation with reference audio
- **Audio Reference Encoder**: Encode reference audio for style transfer

### UI Features

The package includes custom JavaScript UI components that provide:

- **Visual Feedback**: Real-time display of modified layers and operation results
- **Help System**: Built-in help buttons with detailed information
- **Color Coding**: Different node types have distinct colors for easy identification
- **Operation Preview**: See which layers will be affected before applying
- **Progress Tracking**: Monitor long-running operations

## Installation

### Basic Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DavidPiazza/network_bending.git
```

2. Restart ComfyUI

### For Audio Features

If you want to use the audio processing nodes, install additional dependencies:

```bash
cd ComfyUI/custom_nodes/network_bending
pip install -r requirements-audio.txt
```

### Dependencies

Core dependencies (automatically installed):
- PyTorch >= 2.0.0
- NumPy >= 1.21.0

Audio dependencies (optional):
- torchaudio >= 2.0.0
- librosa >= 0.10.0
- scipy >= 1.7.0

Optional audio dependencies for extended format support:
- soundfile >= 0.12.0
- audioread >= 3.0.0
- resampy >= 0.4.0

## Usage Guide

### Basic Network Bending for Images

1. Load a model checkpoint using the standard ComfyUI loader nodes
2. Add a "Network Bending" node to your workflow
3. Connect the MODEL output to the Network Bending input
4. Configure parameters:
   - **Operation**: Select the type of bending operation
   - **Intensity**: Control the strength (0.0 = no effect, 1.0 = maximum)
   - **Target Layers**: Specify which layers to modify (e.g., "conv", "attention", or "all")
   - **Seed**: Set for reproducible results (-1 for random)

### Audio Network Bending

1. **Load an audio model** (e.g., Stable Audio VAE)
2. **Add audio processing nodes** from the network_bending/audio category
3. **Connect audio inputs** either from file loaders or generated audio
4. **Apply bending operations** to the audio latents
5. **Decode back to audio** using Audio VAE Decode

Example audio workflow:
```
Audio Input → Audio VAE Encode → Audio Latent Manipulator → Audio VAE Decode → Audio Output
```

### Advanced Techniques

#### Layered Bending
Chain multiple bending operations with different intensities:
```
Model → Smooth (0.1) → Add Noise (0.05) → Quantize (0.3) → Output
```

#### Cross-Modal Bending
Apply image model bending techniques to audio models or vice versa for experimental results.

#### Targeted Corruption
Use specific layer patterns to corrupt only certain aspects:
- `"transformer"` - Target transformer blocks
- `"resnet"` - Target ResNet blocks
- `"unet"` - Target U-Net components

## Technical Details

### Architecture

The package is built with a modular architecture:

1. **Core Bending Engine**: Base operations that work on any PyTorch model
2. **Model Adapters**: Specific handling for ComfyUI model wrappers
3. **Operation Library**: Extensible set of bending operations
4. **UI Integration**: Real-time feedback through ComfyUI's server

### Implementation Notes

- All operations clone the model to avoid modifying the original
- Supports both CPU and GPU operations
- Memory-efficient implementations for large models
- Thread-safe for parallel processing
- Preserves model metadata and configurations

### Performance Considerations

- Operations scale with model size
- GPU recommended for real-time feedback
- Batch processing supported for efficiency
- Automatic memory management for large models

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

The project uses:
- `ruff` for linting
- `mypy` for type checking
- `pre-commit` hooks for code quality
- `black` for code formatting

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Examples and Workflows

Check the `examples/` directory for pre-built workflows:

- `basic_workflow.json`: Simple network bending setup
- Audio workflows available in `src/network_bending/audio_workflows/`

## Troubleshooting

- **No effect visible**: Increase intensity or check target_layers pattern
- **Model breaks completely**: Reduce intensity or target fewer layers
- **Out of memory**: Operations clone the model, ensure sufficient VRAM

### Audio-Specific Issues

- **No audio output**: Check audio dependencies are installed
- **Format not supported**: Install optional dependencies (soundfile, audioread)
- **Memory issues with long audio**: Process in chunks or reduce sample rate

## License

This project is licensed under the GNU General Public License v3 (GPL-3.0) - see the LICENSE file for details.

## Acknowledgments

- ComfyUI team for the excellent framework
- Anthropic for Opus 4 and Claude Code

