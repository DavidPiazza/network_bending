### Latent Bending for Generative Audio: Operations and Math

This document summarizes latent-space operations implemented in `network_bending` that act directly on VAE latent codes (audio or image-shaped). It focuses on the mathematical formulation, expected perceptual effect for audio, and quick demo recipes.

Notes on shapes:
- Audio latents typically have shape `[B, C, T]` (batch, channels, time).
- Image latents have shape `[B, C, H, W]`. Many ops below are implemented to work for both; for the workshop focus on the `[B, C, T]` case.

---

## 1) Add Noise
Math:
- Let latent `x ∈ ℝ^{B×C×T}`. Sample `ε ~ N(0, I)` with shape of `x`.
- `x' = x + σ ε`, where `σ = intensity`.
- Channel-specific: apply only to selected channel indices.

Perceptual effect:
- Adds broadband texture; increasing `σ` raises noise floor, roughness, and can reveal VAE priors after decode.

Demo recipe:
- Intensity: 0.05–0.2 for subtle to audible noise.
- Compare full mix vs. channel-specific (e.g., channels `0,2`).

---

## 2) Channel Swap (random pairwise)
Math:
- Choose random pairs `(i, j)` among selected channels; swap `x[:, i, :] ↔ x[:, j, :]` repeated `K ≈ ⌊intensity · |channels|⌋` times.

Perceptual effect:
- Reassigns latent subspaces; after decode, can cause timbral re-mapping, stereo/weird color shifts, or structural rearrangements depending on VAE channelization.

Demo recipe:
- Intensity: 0.1–0.5. Play original then swapped; A/B the timbre and stereo image.

---

## 3) Frequency Filter (latent-domain low-pass via FFT)
Math (audio `[B, C, T]`):
- For each channel `c`, compute `X_c = FFT_1D(x_c)`.
- Build mask `M ∈ {0,1}^T` that keeps low frequencies: cutoff `k = ⌊(1 - intensity)·T/2⌋`.
- Set `M[k: T-k] = 0`, leaving only the lowest bands around DC.
- `x'_c = IFFT_1D(X_c ⊙ M).real`.

Perceptual effect:
- Smooths/defocuses latent trajectories; after decode, leads to blurrier transients, softened details, and more sustained character.

Demo recipe:
- Intensity: 0.2 (gentle), 0.6 (strong). Show transient loss on percussive material.

---

## 4) Spatial Corruption (random masked scaling)
Math (audio):
- Pick `num_regions ≈ ⌊10·intensity⌋`. Region length `L = max(1, ⌊0.1·intensity·T⌋)`.
- For each region, random start `s`, apply multiplicative factor `α ~ U[0.5, 1.5]` to `x[:, ch, s:s+L]` for chosen channels.

Perceptual effect:
- Local discontinuities in latent time; after decode, yields glitches, dropouts, or bursts depending on severity.

Demo recipe:
- Intensity: 0.1 for gentle “flutter,” 0.5+ for glitch art. Use short rhythmic audio to make artifacts obvious.

---

## 5) Value Quantization
Math:
- For each channel `c`, compute `m = min(x_c)`, `M = max(x_c)`; normalize `y = (x_c - m)/(M - m + 1e-8)`.
- Choose levels `L = max(2, ⌊(1 - intensity)·256⌋)`.
- Quantize: `q = round(y·(L-1))/(L-1)`; denormalize `x'_c = q·(M - m) + m`.

Perceptual effect:
- Reduces latent precision; after decode, introduces stepped/lo-fi texture, harsher transients, or “bitcrush”-like quality.

Demo recipe:
- Intensity: 0.2 (still smooth), 0.8–0.95 (coarse/aliasy). Compare sustained pads vs. drums.

---

## 6) Dimension Warp (sinusoidal coordinate warp → interpolation)
Math (audio):
- Build normalized grid `t̂ ∈ [-1, 1]^T` and offset `Δ(t̂) = a·sin(4π t̂)` with `a = 0.1·intensity`.
- Map indices `t' = clamp(t + 0.1·T·Δ(t̂), 0, T-1)`; linearly interpolate `x'[:, ch, t] = (1-w)·x[:, ch, ⌊t'⌋] + w·x[:, ch, ⌈t'⌉]`, `w = frac(t')`.

Perceptual effect:
- Wavy temporal bending; after decode, gentle chorusing/warble at low intensities, surreal timing/timbre at higher values.

Demo recipe:
- Intensity: 0.1–0.3 for musical warble. Use held tones and percussive hits to show smearing.

---

## 7) Temporal Shift (circular roll)
Math (audio):
- Shift amount `s = ⌊0.1·intensity·T⌋`.
- `x'[:, ch, :] = roll(x[:, ch, :], shifts=s, dim=-1)`.

Perceptual effect:
- Pure time offset in latent domain; after decode, preserves timbre but re-aligns events. Useful for rhythmic alignment comparisons.

Demo recipe:
- Intensity: 0.1–0.4. Demonstrate phase/time offset relative to original.

---

## 8) Channel Attention (self-similarity weighting)
Math:
- Audio case: reshape `x → X ∈ ℝ^{B×C×T}`. Compute channel-channel similarity `S = X·Xᵀ / √T` per batch.
- Weights `A = softmax(intensity·S, dim=-1)`; output `X' = A·X`.

Perceptual effect:
- Redistributes energy across channels based on mutual similarity; after decode, emphasizes dominant latent directions, often clearer main timbre with reduced detail.

Demo recipe:
- Intensity: 0.3–0.7. Play before/after on complex textures.

---

## 9) Cross-Channel Mixing (convex mixing matrix)
Math:
- Let channels = `C`. Mixing matrix `M = (1 - α)·I + (α/C)·1 1ᵀ`, with `α = intensity`.
- Flatten spatial dims: `X ∈ ℝ^{B×C×T}` → `X̃ ∈ ℝ^{B×C×T}`; apply `X' = M·X̃` (batched matmul per time index).

Perceptual effect:
- Averages features across channels; after decode, smoothes and homogenizes timbre; high values can lose separation.

Demo recipe:
- Intensity: 0.2–0.6. Show progressive loss of separation/stereo interest.

---

## 10) Channel Dropout
Math:
- Sample mask `m ∈ {0,1}^C` with `P(m_c=1) = 1 - intensity`.
- Apply `x' = x ⊙ m` (broadcast over time), where dropped channels are zeros.

Perceptual effect:
- Removes entire latent subspaces; after decode, can mute components or cause dramatic timbre changes.

Demo recipe:
- Intensity: 0.2–0.5. Repeat with different RNG seeds to show variability.

---

## 11) Channel Amplification (variance-weighted)
Math:
- Compute per-channel variance `v_c = Var_t[x[:, c, :]]` and importance `p = softmax(v)`.
- Gain per channel `g_c = 1 + 2·intensity·p_c`.
- Apply `x'[:, c, :] = g_c · x[:, c, :]`.

Perceptual effect:
- Boosts channels carrying more variation; after decode, highlights “expressive” dimensions, often perceived as increased presence or brightness.

Demo recipe:
- Intensity: 0.2–0.6. Compare spectral centroid shifts.

---

## 12) Channel Rotation (pairwise 2D rotations)
Math:
- For pairs `(i, i+1)`, angle `θ = π·intensity`. Apply
- `[x'_i, x'_{i+1}]ᵀ = [[cosθ, -sinθ], [sinθ, cosθ]] · [x_i, x_{i+1}]ᵀ`.

Perceptual effect:
- Rotates latent axes; after decode, smoothly morphs between channel-specific timbres.

Demo recipe:
- Intensity: 0.25, 0.5, 0.75. Sweep intensity and play morph continuum.

---

## 13) Channel Statistics Swap (with reference)
Math:
- Given reference `r`, for each channel `c` (with probability `p=intensity`):
- Compute `(μ_x, σ_x)` from `x[:, c, :]`, `(μ_r, σ_r)` from `r[:, c, :]`.
- Normalize and rescale: `x'[:, c, :] = (x[:, c, :] - μ_x)/(σ_x+1e-8) · σ_r + μ_r`.

Perceptual effect:
- Imposes reference channel statistics; after decode, partial style transfer in dynamics/energy without direct content copying.

Demo recipe:
- Intensity: 0.3–0.8. Use contrasting references; A/B to show timbral bias shift.

---

## Additional Audio-Focused Latent Manipulations (from `audio_latent_nodes`)

These operate on audio latents (`AUDIO_LATENT`) and are useful complementary demos:

- Frequency Shift: roll along frequency axis in FFT domain; pitch/brightness bias without strict pitch shift.
- Temporal Stretch: interpolate along time axis to time-stretch/compress, with crop/pad to original length.
- Harmonic Emphasis: detect spectral peaks via max-pooling in frequency magnitude and boost them.
- Noise Injection: add noise with optional feature-guided scaling.
- Dynamic Range: soft compress/expand values with thresholding.
- Spatial Transform: simple 2D rotation across channel pairs for 2D latents.
- Resonance Filter: Gaussian band in frequency domain centered at target frequency.

Each of these maps to the same general math as above; see code for exact kernels.

---

## Practical Tips for Demos

- Start subtle: many effects decode nonlinearly; 0.1–0.3 intensity is often enough.
- Use contrasting materials: percussive vs. sustained to illustrate time vs. timbre effects.
- Normalize loudness before playback to avoid loudness bias.
- For reproducibility, fix RNG seeds when demonstrating stochastic ops (noise, dropout, corruption).

---

## Quick Slide Cheats

- Show each op with 3 bullets: Math, Ear, Knob (intensity range).
- Add 5–8s audio examples per op, same source clip for fair comparison.
- Include a “combo” slide: e.g., low-pass + warp + amplification.

---

References to implementation
- `network_bending/nodes.py` → classes `VAELatentBending`, `VAEChannelManipulator`.
- `network_bending/audio_nodes/audio_latent_nodes.py` → `AudioLatentManipulator` and related utilities.


