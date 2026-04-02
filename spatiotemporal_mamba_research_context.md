# Spatiotemporal Mamba for Coronary Angiogram Segmentation — Research Context

## Transfer document for Claude Code continuation

*Generated from claude.ai conversation, April 2026*

---

## 1. Problem statement

I have grayscale fluoroscopic coronary angiograms with per-frame pixel-level annotations for three classes: **background**, **catheter**, and **coronary artery**. The data arrives as `nFrame × 512 × 512` UInt8 arrays (variable frame count per study, typically 30–150 frames at ~15 fps). I have a working convolutional segmentation model and am exploring whether transformer or state-space-model (SSM) architectures can improve results by exploiting temporal information.

## 2. Why temporal modeling matters for this data

The temporal axis in coronary angiography is fundamentally different from the spatial axes:

- **Contrast bolus dynamics**: A radiopaque contrast agent is injected and flows through the coronary vasculature. The intensity at each pixel evolves through time as contrast arrives, peaks, and washes out. This creates a causal, sequential signal along the time axis.
- **Cardiac motion**: The heart beats during acquisition, causing periodic displacement of arteries between frames.
- **The key discriminative signal**: In early-fill frames where contrast is faint, or in regions where catheter and artery overlap spatially, single-frame appearance may be ambiguous. The temporal waveform shape (bolus arrival time, peak intensity, washout rate) is what distinguishes artery from catheter from background. This is the same signal my existing wavelet analysis pipeline extracts analytically.

A model that treats time as "just another spatial axis" (like standard 3D convolutions or volumetric SSMs) can learn some of these patterns implicitly, but is not structurally encouraged to learn the bolus waveform as a temporal phenomenon.

## 3. Current setup

- **U-Mamba installed and working** (Bo Wang lab, https://github.com/bowang-lab/U-Mamba)
- U-Mamba is built on **nnU-Net v2** framework
- Uses standard nnU-Net CLI: `nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`, `nnUNetv2_predict`
- Custom trainers: `nnUNetTrainerUMambaEnc`, `nnUNetTrainerUMambaBot`
- U-Mamba treats data as **3D volumetric** (time axis = depth axis), with no explicit temporal modeling
- Mamba packages installed: `causal-conv1d>=1.2.0`, `mamba-ssm`
- **Important**: AMP can cause NaN in Mamba modules. Use `nnUNetTrainerUMambaEncNoAMP` if needed.

## 4. The experiment: temporal-first scan order

### Hypothesis

If we modify U-Mamba's SSM scan order so that the 1D sequence traverses the **temporal axis first** (all frames at each spatial position before moving to the next spatial position), the SSM hidden state will be forced to model the pixel-wise time series — essentially learning a compressed representation of the contrast bolus waveform at each location. This should improve segmentation compared to the default spatial-first scan order, particularly in temporally ambiguous frames.

### Experimental design — three configurations on the same data

1. **Standard U-Mamba** (default scan order) — current baseline
2. **U-Mamba with temporal-first scan** — modified tensor permutation before SSM forward pass
3. **Standard nnU-Net** (no Mamba, pure convolution) — control baseline

### What needs to be modified

The core change is in how the 3D feature tensor is flattened into a 1D sequence before entering the Mamba SSM layer. In U-Mamba's encoder (`nnUNetTrainerUMambaEnc`), the Mamba block receives a tensor of shape `(B, C, D, H, W)` where D is the depth/time axis. The default flattening typically rasterizes in spatial-first order. We need to permute so the temporal (D) axis is innermost in the flattened sequence.

**Conceptually**: Instead of scanning as `(d0,h0,w0), (d0,h0,w1), ..., (d0,h1,w0), ...` (spatial-first), we want `(d0,h0,w0), (d1,h0,w0), (d2,h0,w0), ..., (d0,h0,w1), (d1,h0,w1), ...` (temporal-first).

This is a reshape/permute operation on the tensor before the `mamba_ssm` forward call, and the inverse permutation on the output. No new architecture, no new CUDA code, no new training scripts.

### Where to find the code

Look in the U-Mamba source tree for:
- The Mamba block definition (likely in `umamba/nnunetv2/` or similar)
- The point where the 3D feature map is reshaped into a 1D sequence for the SSM
- The `mamba_ssm` forward call

The modification is a `torch.permute()` or `tensor.reshape()` change at that flattening point.

## 5. Background: architectures considered and rejected

### RF-DETR
- Real-time detection transformer from Roboflow (2025)
- Single-image, no temporal mechanism
- Instance segmentation (bounding box + mask), not semantic segmentation
- Poor fit for thin branching vascular structures

### SAM 3 (Segment Anything Model 3, Meta, Nov 2025)
- Has temporal mechanism via memory-based tracker (inherited from SAM 2)
- But temporal mechanism is for **object tracking/identity preservation**, not for learning temporal signal dynamics
- Memory bank stores mask features from previous frames, uses cross-attention for propagation
- Designed for "where did this object go?", not "how does this pixel's intensity evolve?"
- Massive model, open-vocabulary — overkill for 3-class segmentation
- Fine-tuning paths exist (MedSAM3, SAM3-Adapter) but heavy

### SwinUNETR (MONAI)
- 3D Swin Transformer encoder + CNN UNet decoder
- Shifted-window self-attention across spatiotemporal dimensions
- Medical pretrained weights on 5050 CT scans
- Well-integrated in MONAI framework
- Good option but quadratic complexity in window size; no explicit temporal-first modeling
- Strong fallback if Mamba approach doesn't pan out

### Vivim (Video Vision Mamba, IEEE TCSVT)
- Paper: https://arxiv.org/abs/2401.14168
- Code: https://github.com/scott-yjyang/Vivim
- Architecturally the most compelling match: explicit tri-directional spatiotemporal scanning
  - Temporal forward SSM (causal bolus arrival)
  - Temporal backward SSM (full waveform context)
  - Spatial forward SSM (vessel topology)
- Validated on grayscale medical video (ultrasound thyroid, breast, colonoscopy polyp)
- **BUT**: codebase immature (incomplete TODO list, no released weights, custom training scripts, requires building causal-conv1d and mamba from source). Does NOT integrate with nnU-Net.
- The scan-order experiment in U-Mamba is a lightweight way to test the core hypothesis before committing to Vivim's full architecture.

### VideoMamba
- Pure Mamba for video understanding (classification, not segmentation)
- Spatial-first bidirectional scan found optimal
- Linear complexity advantage over transformers when frames > 256
- No segmentation decoder

### UlikeMamba (comprehensive 3D Mamba analysis)
- Outperforms nnUNet, SwinUNETR, U-Mamba on AMOS/TotalSegmentator/BraTS
- Uses 3D depthwise convolutions before SSM + multi-scale modeling + Tri-scan
- Volumetric (no explicit temporal modeling)

## 6. Mamba-3 and the complex-valued state update

The U-Mamba installation on this machine has been adapted to use **Mamba-3** (Lahoti et al., March 2026, arXiv:2603.15569). Mamba-3 introduces three improvements over Mamba-2:

1. **More expressive recurrence** derived from SSM discretization (exponential-trapezoidal discretization rather than zero-order hold)
2. **Complex-valued state update rule** — the hidden state $h$ and state matrix $A$ operate in $\mathbb{C}$ rather than $\mathbb{R}$, enabling richer state tracking
3. **Multi-input, multi-output (MIMO) formulation** — better performance without increasing decode latency

The complex-valued states are the most intriguing aspect for our spatiotemporal application. I do not yet fully understand the mechanism, but the key insight is that complex exponentials $e^{i\omega t}$ naturally represent oscillatory dynamics. A complex-valued $A$ matrix in the state update $h_t = A h_{t-1} + B x_t$ can represent damped oscillations in the hidden state — which is physically what the contrast bolus waveform modulated by cardiac motion looks like. Real-valued states can only represent exponential decay/growth; complex-valued states can represent rotation and oscillation natively.

**Open question for future investigation**: Does the Mamba-3 complex-valued architecture lend itself better to spatiotemporal modeling than Mamba-1/2? Specifically:
- Can the complex eigenvalues of $A$ learn to separate cardiac-frequency oscillation from lower-frequency bolus dynamics?
- Does this partially solve the cardiac motion problem identified in our temporal-first scan analysis (where periodic arterial displacement contaminates the pixel-wise time series)?
- Is the MIMO formulation useful for multi-class segmentation (background/catheter/artery as multiple outputs)?

This should be revisited once we understand the Mamba-3 architecture more deeply.

## 7. Cardiac motion and scan order limitations

Pure temporal-first scanning has a known limitation for coronary angiography: the heart beats during acquisition, causing arteries to shift by 10-20+ pixels between systole and diastole. A single pixel's time series then conflates the bolus waveform with periodic displacement artifacts — the pixel sees "bright, bright, dark (artery moved away), dark, bright (artery came back)" rather than a clean bolus curve.

Possible mitigations (not yet implemented):
- **Neighborhood temporal scanning**: Scan a small spatial patch (e.g., 3×3) at each time step rather than a single pixel. If the artery shifts within the patch, the signal remains coherent.
- **Vivim-style parallel branches**: Separate spatial and temporal scanning so the spatial branch can track where the artery is while the temporal branch tracks intensity evolution.
- **Cardiac-phase conditioning**: Gate the temporal scan to process only frames at similar cardiac phases (requires ECG or surrogate signal). Diastolic-only subsequences have minimal motion.
- **Complex-valued Mamba-3 states**: The oscillatory representation may naturally disentangle periodic cardiac motion from aperiodic bolus dynamics (see section 6).

## 8. Key references

- U-Mamba: Ma et al., "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation", arXiv:2401.04722 (2024)
- Vivim: Yang et al., "Vivim: a Video Vision Mamba for Medical Video Segmentation", IEEE TCSVT (2025), arXiv:2401.14168
- VideoMamba: Park et al., "VideoMamba: Spatio-Temporal Selective State Space Model", ECCV 2024, arXiv:2407.08476
- SwinUNETR: Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", arXiv:2201.01266
- Video Swin Transformer: Liu et al., CVPR 2022, arXiv:2106.13230
- Mamba: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv:2312.00752 (2023)
- Mamba-3: Lahoti et al., "Mamba-3: Improved Sequence Modeling using State Space Principles", arXiv:2603.15569 (March 2026)
- UlikeMamba: "A Comprehensive Analysis of Mamba for 3D Volumetric Medical Image Segmentation", arXiv:2503.19308

## 9. Data format notes

- Raw data: `nFrame × 512 × 512`, UInt8, grayscale
- Annotations: same dimensions, integer labels (0=background, 1=catheter, 2=artery)
- For nnU-Net: convert to NIfTI (.nii.gz), treat temporal axis as z-axis
- nnU-Net dataset structure: `Dataset_XXX/imagesTr/`, `labelsTr/`, `dataset.json`
- Custom data server: Angiostore (HDF5-based)

## 10. Next steps for Claude Code

### Immediate task: Locate and understand the scan-order code

1. **Find the Mamba block** in the installed U-Mamba source tree
2. **Identify the exact point** where the 3D feature tensor is flattened into a 1D sequence before the SSM forward pass
3. **Understand the current scan order** — how is (D, H, W) mapped to the 1D sequence?

### Staged experimental design

The experiments below are designed to test two interacting hypotheses: (a) whether temporal-first scan order improves segmentation, and (b) whether Mamba-3's complex-valued states specifically benefit temporal modeling. The stages are ordered by implementation difficulty and scientific informativeness.

**Stage 1: Temporal-first scan order (Mamba-3 as-is)**

Modify the tensor permutation so the SSM scans the temporal axis first (all frames at each spatial location as a contiguous subsequence, then moving to the next spatial location). This is a reshape/permute change only — no new architecture.

Create a new nnU-Net trainer class (e.g., `nnUNetTrainerUMambaEncTemporalFirst`).

Three-way comparison on the same data:
- Standard U-Mamba with Mamba-3 (default spatial-first scan) — baseline
- U-Mamba with Mamba-3 (temporal-first scan) — experimental
- Standard nnU-Net (no Mamba, pure convolution) — control

This answers: *Does temporal-first scan order help at all?*

**Stage 2: Complex vs real in temporal-first (isolating the interaction)**

Compare Mamba-3 (complex-valued, with RoPE-based rotation) vs Mamba-2 (real-valued) using the **same temporal-first scan order**. Also compare both in the default spatial-first scan order.

This is a 2×2 design:
- Mamba-3 (complex) × spatial-first
- Mamba-3 (complex) × temporal-first
- Mamba-2 (real) × spatial-first
- Mamba-2 (real) × temporal-first

This answers the critical question: *Do complex eigenvalues help specifically when scanning temporally?*

The reasoning: Mamba-3's complex states allow the SSM hidden state to **rotate** (oscillate) rather than only decay. Along the temporal axis, there is genuine oscillatory structure — the cardiac motion creates a quasi-periodic modulation of pixel intensity at the heart rate (~1 Hz). Along the spatial axes, there is no such periodicity. Therefore:
- If Mamba-3 outperforms Mamba-2 specifically in temporal-first but not spatial-first, the complex eigenvalues are capturing cardiac-frequency dynamics — strong evidence for asymmetric design.
- If Mamba-3 outperforms Mamba-2 regardless of scan order, the rotation is doing something more generic (better positional encoding), and asymmetric design is less critical.
- If neither scan order nor complex/real matters, the temporal signal may not be discriminative beyond what convolutions already capture.

**Stage 3: Asymmetric architecture (complex temporal + real spatial branches)**

If Stages 1-2 show that temporal-first + complex states help, implement a parallel-branch design within U-Mamba/nnU-Net:

- **Temporal branch**: Mamba-3 with complex/RoPE states. Scans temporal-first. The rotation angle θ_t can learn to lock onto the cardiac frequency. The scalar decay tracks the bolus envelope. Hidden state magnitude encodes bolus amplitude; phase encodes cardiac cycle position.
- **Spatial branch**: Real-valued (Mamba-2 or Mamba-3 without rotation). Scans spatially within each frame. No periodic structure to exploit — just vessel geometry, catheter position, local context.
- **Fusion**: Concatenate or add branch outputs before the decoder.

This is essentially a Vivim-like tri-directional design but with Mamba-3's complex states in the temporal branch, implemented within the nnU-Net framework. The physics motivation: the temporal dimension has oscillatory dynamics (cardiac motion) that benefit from complex representation; the spatial dimensions don't. Match the inductive bias to the data structure on a per-axis basis.

**Stage 4 (future): Cardiac-phase conditioning**

If the complex temporal branch learns rotation angles that correlate with cardiac phase, explore gating the temporal scan to process only phase-coherent frames. Requires ECG or surrogate cardiac signal (photodiode/pulse-ox synchronization hardware project). Deferred for now.

## 11. Technical environment

- Python/PyTorch stack
- nnU-Net v2 installed
- U-Mamba installed (bowang-lab), **adapted to use Mamba-3** (arXiv:2603.15569)
- `mamba-ssm` and `causal-conv1d` packages installed (Mamba-3 compatible versions)
- GPU: NVIDIA V100 (p3.2xlarge AWS) — 16GB HBM2
- Also developing in Julia, Wolfram Language (separate pipelines)
