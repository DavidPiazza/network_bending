# Audio Latent Space Manipulation Nodes for Stable Audio

This collection of nodes enables sophisticated audio manipulation and style transfer by working directly in the latent space of Stable Audio's VAE. This approach allows for fine-grained control over audio generation without relying solely on text prompts.

## Overview

The latent space approach offers several advantages:
- **Direct audio-to-audio transformations** without text bottlenecks
- **Style transfer** between different audio sources
- **Hybrid sound generation** by blending multiple audio characteristics
- **Feature-based manipulation** using audio analysis
- **Real-time guidance** during the diffusion process

## Node Categories

### 1. Basic Encoding/Decoding

#### AudioVAEEncode
Encodes audio waveforms into Stable Audio's latent space.
- **Inputs**: Audio waveform, VAE model
- **Parameters**: Normalize (bool), target sample rate
- **Outputs**: Audio latent, latent info

#### AudioVAEDecode
Decodes audio latents back to waveforms.
- **Inputs**: Audio latent, VAE model
- **Parameters**: Denormalize (bool)
- **Outputs**: Audio waveform

### 2. Latent Manipulation

#### AudioLatentInterpolate
Smoothly interpolate between two audio latents.
- **Modes**: Linear, spherical (SLERP), cubic, sine
- **Use cases**: Morphing between sounds, creating transitions

#### AudioLatentBlend
Blend multiple audio latents with various blend modes.
- **Modes**: Add, multiply, screen, overlay, soft light, difference, exclusion
- **Supports**: Up to 4 audio inputs with individual weights

#### AudioLatentManipulator
Apply various transformations to audio latents.
- **Operations**:
  - Frequency shift: Shift frequency content
  - Temporal stretch: Time-stretch without pitch change
  - Harmonic emphasis: Enhance harmonic content
  - Noise injection: Add controlled noise
  - Dynamic range: Compress/expand dynamics
  - Spatial transform: Rotate in latent space
  - Resonance filter: Apply resonant filtering

### 3. Audio Analysis

#### AudioFeatureExtractor
Extract audio features for guided manipulation.
- **Features**: Spectral centroid, rolloff, zero-crossing rate, MFCC, bandwidth, RMS energy, tempo, onset strength
- **Use**: Guide latent manipulations based on audio characteristics

### 4. Style Transfer

#### AudioStyleTransfer
Transfer style characteristics between audio sources.
- **Modes**:
  - Global: Statistics matching
  - Frequency bands: Band-wise style transfer
  - Temporal segments: Time-based style transfer
  - Adaptive: Correlation-based transfer
  - Neural style: Deep feature matching

#### AudioReferenceEncoder
Encode multiple audio references into a combined representation.
- **Combination modes**: Average, weighted, PCA, attention
- **Features**: Segment extraction, weighted blending

### 5. Generation Guidance

#### AudioLatentGuidance
Guide the diffusion process with reference latents.
- **Modes**: Additive, multiplicative, attention, gradient
- **Control**: Start/end steps, guidance strength

## Workflow Examples

### Basic Style Transfer
```
1. Load content and style audio
2. Encode both with AudioVAEEncode
3. Apply AudioStyleTransfer
4. Decode result with AudioVAEDecode
```

### Hybrid Sound Creation
```
1. Load multiple audio sources
2. Encode all with AudioVAEEncode
3. Use AudioLatentBlend to combine
4. Apply AudioLatentManipulator for fine-tuning
5. Decode final result
```

### Feature-Guided Manipulation
```
1. Load reference audio
2. Extract features with AudioFeatureExtractor
3. Encode target audio
4. Use AudioLatentManipulator with features
5. Decode result
```

### Guided Generation
```
1. Encode reference audio
2. Apply AudioLatentGuidance to model
3. Generate new audio with guidance
4. Fine-tune with style transfer if needed
```

## Technical Details

### Latent Space Properties
- Stable Audio's VAE compresses audio at ~21.5Hz latent rate
- Latents preserve both spectral and temporal information
- Manipulation in latent space is computationally efficient

### Best Practices
1. **Normalize audio** before encoding for consistent results
2. **Match sample rates** between audio sources
3. **Use appropriate blend modes** for your use case
4. **Experiment with strength parameters** - start low and increase
5. **Combine multiple techniques** for complex transformations

### Performance Considerations
- Encoding/decoding is GPU-accelerated
- Latent manipulations are memory-efficient
- Real-time processing possible with optimization
- Batch processing supported for multiple audio files

## Advanced Techniques

### Creating Instrument Hybrids
Blend characteristics of different instruments:
1. Encode each instrument separately
2. Use frequency-band style transfer
3. Apply harmonic emphasis
4. Fine-tune with latent interpolation

### Environmental Audio Design
Create complex soundscapes:
1. Encode multiple environmental sounds
2. Use temporal segment transfer
3. Apply spatial transformations
4. Blend with attention-based combination

### Rhythmic Pattern Transfer
Transfer rhythm while preserving timbre:
1. Extract tempo/onset features
2. Use temporal stretch with feature guidance
3. Apply style transfer in temporal segments

## Troubleshooting

### Common Issues
- **Artifacts in output**: Reduce manipulation strength
- **Loss of clarity**: Check normalization settings
- **Mismatched lengths**: Ensure proper padding/cropping
- **Memory issues**: Process in smaller segments

### Tips for Quality
- Use high-quality source audio (44.1kHz recommended)
- Avoid extreme parameter values
- Test different interpolation modes
- Monitor latent statistics for anomalies

## Integration with Stable Audio

These nodes are designed to work seamlessly with Stable Audio's generation pipeline:
- Compatible with text conditioning
- Can be used before/after generation
- Supports timing embeddings
- Works with all Stable Audio checkpoints

## Future Enhancements

Planned features:
- Multi-scale latent manipulation
- Learned projection heads for other modalities
- Advanced spectral processing
- Real-time preview capabilities
- Extended feature extraction options

## Credits

This implementation is based on research in:
- Audio VAE architectures
- Style transfer techniques
- Latent space manipulation
- Audio feature analysis

For more information and updates, visit the project repository.