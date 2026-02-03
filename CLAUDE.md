# Self-Forcing Codebase: Comprehensive Technical Summary

## Overview

This codebase implements **Self-Forcing**, a novel training methodology for autoregressive video diffusion models that bridges the train-test distribution gap by simulating the inference process during training. The approach enables real-time streaming video generation while maintaining state-of-the-art quality.

**Key Innovation**: During training, the model performs autoregressive rollout with KV caching, matching the inference-time behavior. This eliminates the distribution mismatch where models are trained on clean context but tested on generated context.

## Project Structure

```
Self-Forcing-main/
├── model/               # Model implementations (DMD, SiD, CausVid, etc.)
├── trainer/             # Training orchestration
├── pipeline/            # Inference and training pipelines
├── wan/                 # Wan video model architecture
│   ├── modules/         # Core model components (attention, VAE, text encoder)
│   └── configs/         # Model configurations
├── utils/               # Utilities (dataset, loss, scheduler, distributed)
├── configs/             # Training configurations
└── scripts/             # Data preparation scripts
```

---

## Core Architecture

### 1. Base Model Architecture ([model/base.py](model/base.py))

#### BaseModel Class (Lines 12-96)
The foundation class that initializes all model components:

**Key Components**:
- **Generator** ([model/base.py:30](model/base.py#L30)): Causal diffusion model (`WanDiffusionWrapper` with `is_causal=True`)
- **Real Score** ([model/base.py:33](model/base.py#L33)): Pre-trained teacher model (non-causal, frozen)
- **Fake Score** ([model/base.py:36](model/base.py#L36)): Trainable critic/discriminator (non-causal)
- **Text Encoder** ([model/base.py:39](model/base.py#L39)): UMT5-XXL for text conditioning (frozen)
- **VAE** ([model/base.py:42](model/base.py#L42)): Video encoder/decoder (frozen)

**Timestep Generation** ([model/base.py:48-95](model/base.py#L48-95)):
```python
def _get_timestep(self, min_timestep, max_timestep, batch_size, num_frame,
                  num_frame_per_block, uniform_timestep=False)
```
- Supports uniform timesteps (same across frames) or block-wise timesteps
- Handles independent first frame for image-to-video tasks
- Ensures consistent timesteps within blocks for autoregressive generation

#### SelfForcingModel Class ([model/base.py:98-223](model/base.py#L98-223))

**Core Training Loop** ([model/base.py:103-180](model/base.py#L103-180)):
```python
def _run_generator(self, image_or_video_shape, conditional_dict, initial_latent=None)
```

1. **Variable-length Training** ([model/base.py:131-147](model/base.py#L131-147)):
   - Randomly samples number of frames from `[21, num_training_frames]`
   - Must be multiple of `num_frame_per_block` (default: 3 frames)
   - Enables training on varying video lengths

2. **Backward Simulation** ([model/base.py:149-153](model/base.py#L149-153)):
   - Calls `_consistency_backward_simulation()` to generate synthetic trajectories
   - Uses consistency sampling to create training data without real videos
   - Returns generated frames and denoising timestep range

3. **Context Re-encoding** ([model/base.py:155-165](model/base.py#L155-165)):
   - For videos >21 frames, re-encodes the last generated frame
   - Converts video latent → pixel → image latent
   - Ensures proper context handling for long videos

4. **Gradient Masking** ([model/base.py:169-177](model/base.py#L169-177)):
   - Only computes gradients for last 21 frames
   - First chunk uses image latents (no gradient)
   - Reduces memory usage for long sequences

**Consistency Backward Simulation** ([model/base.py:182-204](model/base.py#L182-204)):
- Core innovation from DMD2 paper (Sec 4.5)
- Simulates noisy inputs at various denoising steps
- Uses consistency sampler to avoid real video data requirement

---

### 2. Wan Diffusion Architecture

#### WanDiffusionWrapper ([utils/wan_wrapper.py:115-143](utils/wan_wrapper.py#L115-143))

**Model Selection**:
- Causal mode: Uses `CausalWanModel` ([wan/modules/causal_model.py](wan/modules/causal_model.py))
- Non-causal mode: Uses standard `WanModel` ([wan/modules/model.py](wan/modules/model.py))

**Flow Matching Scheduler** ([utils/wan_wrapper.py:136-139](utils/wan_wrapper.py#L136-139)):
```python
self.scheduler = FlowMatchScheduler(
    shift=timestep_shift,  # Default: 5.0-8.0
    sigma_min=0.0,
    extra_one_step=True
)
```

#### CausalWanModel ([wan/modules/causal_model.py](wan/modules/causal_model.py))

**CausalWanSelfAttention** ([wan/modules/causal_model.py:58-240](wan/modules/causal_model.py#L58-240)):

Key features:
- **KV Cache Management** ([wan/modules/causal_model.py:204-235](wan/modules/causal_model.py#L204-235)):
  - Stores keys/values for previous frames
  - Local attention window with configurable size
  - Sink tokens for stable attention patterns
  - Automatic cache eviction when full

- **Rope Positioning** ([wan/modules/causal_model.py:27-55](wan/modules/causal_model.py#L27-55)):
  ```python
  def causal_rope_apply(x, grid_sizes, freqs, start_frame=0)
  ```
  - Applies rotary embeddings with frame-aware offsets
  - Splits frequencies into temporal, height, width components

- **FlexAttention** ([wan/modules/causal_model.py:156-192](wan/modules/causal_model.py#L156-192)):
  - Uses PyTorch's compiled flex_attention
  - Supports teacher forcing (concatenated clean/noisy inputs)
  - Block masking for causal constraints

**Attention Block** ([wan/modules/causal_model.py:243-350](wan/modules/causal_model.py#L243-350)):
```python
class CausalWanAttentionBlock:
    - Self-attention (causal)
    - Cross-attention (with text embeddings)
    - Feed-forward network
    - Adaptive layer normalization (AdaLN)
```

#### Standard WanModel ([wan/modules/model.py](wan/modules/model.py))

**WanSelfAttention** ([wan/modules/model.py:102-156](wan/modules/model.py#L102-156)):
- Standard bidirectional attention for teacher/critic
- RMSNorm for query/key normalization ([wan/modules/model.py:70-86](wan/modules/model.py#L70-86))
- Flash attention for efficiency ([wan/modules/model.py:146-151](wan/modules/model.py#L146-151))

**WanT2VCrossAttention** ([wan/modules/model.py:159-194](wan/modules/model.py#L159-194)):
- Cross-attention cache for text embeddings
- Computed once and reused across timesteps

---

### 3. Training Algorithms

#### DMD (Distribution Matching Distillation) ([model/dmd.py](model/dmd.py))

**Core Loss Computation** ([model/dmd.py:54-126](model/dmd.py#L54-126)):

```python
def _compute_kl_grad(self, noisy_image_or_video, estimated_clean_image_or_video,
                     timestep, conditional_dict, unconditional_dict)
```

**Step-by-Step Process**:

1. **Fake Score** ([model/dmd.py:75-91](model/dmd.py#L75-91)):
   ```python
   pred_fake_image = pred_fake_image_cond +
                     (pred_fake_image_cond - pred_fake_image_uncond) * fake_guidance_scale
   ```
   - Trainable student model prediction
   - Optional classifier-free guidance

2. **Real Score** ([model/dmd.py:96-110](model/dmd.py#L96-110)):
   ```python
   pred_real_image = pred_real_image_cond +
                     (pred_real_image_cond - pred_real_image_uncond) * real_guidance_scale
   ```
   - Frozen teacher model prediction
   - Uses CFG with `real_guidance_scale` (typically 3.0)

3. **Gradient Computation** ([model/dmd.py:113](model/dmd.py#L113)):
   ```python
   grad = pred_fake_image - pred_real_image
   ```
   - Core DMD gradient (Eq. 7 from paper)

4. **Normalization** ([model/dmd.py:116-120](model/dmd.py#L116-120)):
   ```python
   p_real = estimated_clean_image_or_video - pred_real_image
   normalizer = torch.abs(p_real).mean(dim=[1,2,3,4], keepdim=True)
   grad = grad / normalizer
   ```
   - DMD Eq. 8: gradient scaling

**Generator Loss** ([model/dmd.py:128-194](model/dmd.py#L128-194)):

```python
def compute_distribution_matching_loss(self, image_or_video, conditional_dict,
                                       unconditional_dict, gradient_mask)
```

1. **Timestep Sampling** ([model/dmd.py:154-163](model/dmd.py#L154-163)):
   - Uniform timestep across frames
   - Timestep scheduling: `min_timestep = denoised_timestep_to` (if enabled)
   - Timestep shifting: `t' = shift * t / (1 + (shift-1) * t)`

2. **Noise Addition** ([model/dmd.py:172-177](model/dmd.py#L172-177)):
   ```python
   noisy_latent = scheduler.add_noise(image_or_video, noise, timestep)
   ```

3. **Loss Calculation** ([model/dmd.py:188-193](model/dmd.py#L188-193)):
   ```python
   dmd_loss = 0.5 * F.mse_loss(
       original_latent.double()[gradient_mask],
       (original_latent.double() - grad.double()).detach()[gradient_mask]
   )
   ```
   - Implicit gradient descent on KL divergence
   - Only computed on masked regions (last 21 frames)

**Critic Loss** ([model/dmd.py:237-332](model/dmd.py#L237-332)):

1. **Generate Samples** ([model/dmd.py:261-266](model/dmd.py#L261-266)):
   - Run generator without gradients
   - Backward simulation produces synthetic videos

2. **Teacher Forcing with Positional Embedding Reuse** ([wan/modules/causal_model.py:118-134](wan/modules/causal_model.py#L118-134)):

   **Key Innovation**: When training the critic, clean and noisy frames are concatenated but receive **identical positional embeddings**:

   ```python
   # Input: [Clean_0, Clean_1, Clean_2, Noisy_0, Noisy_1, Noisy_2]
   # Split into chunks
   q_chunk = torch.chunk(q, 2, dim=1)  # [clean, noisy]
   k_chunk = torch.chunk(k, 2, dim=1)

   # Apply SAME RoPE to both clean and noisy parts
   for ii in range(2):
       rq = rope_apply(q_chunk[ii], grid_sizes, freqs)  # Same freqs!
       rk = rope_apply(k_chunk[ii], grid_sizes, freqs)  # Same freqs!
   ```

   **Why This Matters**:
   - **Without this trick**: Noisy frames would get positions [3, 4, 5], different from clean [0, 1, 2]
   - **With this trick**: Noisy frames get positions [0, 1, 2], same as clean frames
   - **Result**: Model learns position-invariant features, eliminating train-test distribution mismatch

   **Attention Pattern** ([wan/modules/causal_model.py:622-632](wan/modules/causal_model.py#L622-632)):
   ```python
   def attention_mask(b, h, q_idx, kv_idx):
       # Clean frames: standard block-wise causal
       clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])

       # Noisy frames: attend to clean context + noisy within block
       C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
       C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
       noise_mask = (q_idx >= clean_ends) & (C1 | C2)

       return (q_idx == kv_idx) | clean_mask | noise_mask
   ```

   **Benefits**:
   - Processes clean and noisy frames in one forward pass (2x faster)
   - Clean section has no gradients (only trains on noisy predictions)
   - Eliminates positional bias between training and inference

3. **Denoising Loss** ([model/dmd.py:293-325](model/dmd.py#L293-325)):
   ```python
   denoising_loss = denoising_loss_func(
       x=generated_image,
       x_pred=pred_fake_image,
       noise=critic_noise,
       flow_pred=flow_pred  # If using flow matching
   )
   ```
   - Trains fake score to predict clean data
   - Supports multiple loss types: flow, x0, noise, v-prediction

#### SiD (Score Identity Distillation) ([model/sid.py](model/sid.py))

**SiD Loss** ([model/sid.py:47-145](model/sid.py#L47-145)):

```python
sid_loss = (pred_real_image - pred_fake_image) *
           ((pred_real_image - original_latent) -
            sid_alpha * (pred_real_image - pred_fake_image))
```
([model/sid.py:128](model/sid.py#L128))

**Key Difference from DMD**:
- Direct loss on score difference
- `sid_alpha` hyperparameter (default: 1.0) controls regularization
- Potentially more stable than DMD gradient
- Same critic training as DMD ([model/sid.py:188-283](model/sid.py#L188-283))

#### CausVid (Baseline) ([model/causvid.py](model/causvid.py))

**Simpler Backward Simulation** ([model/causvid.py:184-253](model/causvid.py#L184-253)):

```python
def _run_generator(self, image_or_video_shape, conditional_dict, clean_latent)
```

1. **Discrete Timesteps** ([model/causvid.py:202-221](model/causvid.py#L202-221)):
   - Pre-computes noisy inputs at fixed timesteps
   - Stacks all timesteps: `[B, T, F, C, H, W]`
   - Randomly selects one per frame

2. **No Autoregressive Rollout**:
   - Doesn't perform multi-step generation during training
   - Uses clean latents as context
   - Less expensive but has train-test mismatch

---

### 4. Self-Forcing Training Pipeline

#### SelfForcingTrainingPipeline ([pipeline/self_forcing_training.py](pipeline/self_forcing_training.py))

**Initialization** ([pipeline/self_forcing_training.py:8-39](pipeline/self_forcing_training.py#L8-39)):

```python
def __init__(self, denoising_step_list, scheduler, generator,
             num_frame_per_block=3, independent_first_frame=False,
             same_step_across_blocks=False, last_step_only=False,
             num_max_frames=21, context_noise=0)
```

**Key Parameters**:
- `denoising_step_list`: Timesteps for backward simulation (e.g., [1000, 750, 500, 250])
- `num_frame_per_block`: Frames generated autoregressively per step
- `independent_first_frame`: Whether first frame uses different conditioning
- `same_step_across_blocks`: Use same denoising step for all blocks
- `context_noise`: Noise added to context frames

**KV Cache Initialization** ([pipeline/self_forcing_training.py:239-267](pipeline/self_forcing_training.py#L239-267)):

```python
def _initialize_kv_cache(self, batch_size, dtype, device):
    for _ in range(num_transformer_blocks):  # 30 blocks
        kv_cache.append({
            "k": zeros([batch_size, kv_cache_size, 12, 128]),
            "v": zeros([batch_size, kv_cache_size, 12, 128]),
            "global_end_index": 0,
            "local_end_index": 0
        })
```
- 30 transformer blocks × 12 attention heads × 128 head dimension
- `kv_cache_size = num_max_frames * 1560` (frame sequence length)

**Inference with Trajectory** ([pipeline/self_forcing_training.py:60-237](pipeline/self_forcing_training.py#L60-237)):

Core autoregressive generation loop:

1. **Initial Frame Processing** ([pipeline/self_forcing_training.py:114-129](pipeline/self_forcing_training.py#L114-129)):
   ```python
   if initial_latent is not None:
       output[:, :1] = initial_latent
       generator(initial_latent, timestep=0, kv_cache=kv_cache1)
   ```
   - Encodes image conditioning
   - Populates KV cache with first frame features

2. **Autoregressive Rollout** ([pipeline/self_forcing_training.py:139-219](pipeline/self_forcing_training.py#L139-219)):
   ```python
   for block_index, current_num_frames in enumerate(all_num_frames):
       noisy_input = noise[:, current_start:current_start+current_num_frames]

       # Spatial denoising loop
       for index, current_timestep in enumerate(denoising_step_list):
           exit_flag = (index == exit_flags[block_index])

           if not exit_flag:
               with torch.no_grad():
                   denoised_pred = generator(noisy_input, timestep, kv_cache)
                   noisy_input = scheduler.add_noise(denoised_pred, ...)
           else:
               # Compute gradients only at random exit point
               denoised_pred = generator(noisy_input, timestep, kv_cache)
               break
   ```

3. **Random Exit Strategy** ([pipeline/self_forcing_training.py:136](pipeline/self_forcing_training.py#L136)):
   - Randomly selects which denoising step to compute gradients
   - `last_step_only=True`: Always exit at final step
   - `same_step_across_blocks=True`: Same exit for all blocks

4. **Context Update** ([pipeline/self_forcing_training.py:199-216](pipeline/self_forcing_training.py#L199-216)):
   ```python
   context_timestep = torch.ones_like(timestep) * context_noise
   denoised_pred_noisy = scheduler.add_noise(denoised_pred, context_timestep)
   with torch.no_grad():
       generator(denoised_pred_noisy, context_timestep, kv_cache)
   ```
   - Re-runs generator at `t=context_noise` (default: 0)
   - Updates KV cache with clean/noisy context
   - Simulates inference-time conditions

5. **Gradient Masking** ([pipeline/self_forcing_training.py:137-175](pipeline/self_forcing_training.py#L137-175)):
   ```python
   start_gradient_frame_index = num_output_frames - 21
   if current_start_frame < start_gradient_frame_index:
       with torch.no_grad():
           denoised_pred = generator(...)
   else:
       denoised_pred = generator(...)  # Compute gradients
   ```
   - Only last 21 frames have gradients
   - Reduces memory for long videos

---

### 5. Training Orchestration

#### Trainer Class ([trainer/distillation.py](trainer/distillation.py))

**Initialization** ([trainer/distillation.py:20-181](trainer/distillation.py#L20-181)):

1. **Distributed Setup** ([trainer/distillation.py:25-45](trainer/distillation.py#L25-45)):
   - FSDP (Fully Sharded Data Parallel)
   - Random seed: `config.seed + global_rank`
   - Mixed precision: bfloat16

2. **Model Selection** ([trainer/distillation.py:61-68](trainer/distillation.py#L61-68)):
   ```python
   if config.distribution_loss == "causvid":
       model = CausVid(config, device)
   elif config.distribution_loss == "dmd":
       model = DMD(config, device)
   elif config.distribution_loss == "sid":
       model = SiD(config, device)
   ```

3. **FSDP Wrapping** ([trainer/distillation.py:73-100](trainer/distillation.py#L73-100)):
   ```python
   model.generator = fsdp_wrap(
       model.generator,
       sharding_strategy="hybrid_full",
       wrap_strategy="size"  # Wraps by parameter count
   )
   ```
   - Generator, real_score, fake_score, text_encoder all wrapped
   - Reduces memory footprint across GPUs

4. **Optimizers** ([trainer/distillation.py:106-120](trainer/distillation.py#L106-120)):
   ```python
   generator_optimizer = AdamW(
       generator.parameters(),
       lr=2e-6,  # Generator LR
       betas=(0.0, 0.999)
   )

   critic_optimizer = AdamW(
       fake_score.parameters(),
       lr=4e-7,  # Critic LR (5x lower)
       betas=(0.0, 0.999)
   )
   ```

5. **EMA (Exponential Moving Average)** ([trainer/distillation.py:140-176](trainer/distillation.py#L140-176)):
   ```python
   if ema_weight > 0.0:
       generator_ema = EMA_FSDP(generator, decay=0.99)
   ```
   - Starts at iteration 200
   - Maintains smooth version of generator weights

**Training Loop** ([trainer/distillation.py:312-389](trainer/distillation.py#L312-389)):

```python
def train(self):
    while True:
        TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)

        # Train generator every 5 iterations
        if TRAIN_GENERATOR:
            generator_optimizer.zero_grad()
            batch = next(dataloader)
            extra = fwdbwd_one_step(batch, train_generator=True)
            generator_optimizer.step()
            if generator_ema:
                generator_ema.update(generator)

        # Train critic every iteration
        critic_optimizer.zero_grad()
        batch = next(dataloader)
        extra = fwdbwd_one_step(batch, train_generator=False)
        critic_optimizer.step()

        step += 1
```
([trainer/distillation.py:315-340](trainer/distillation.py#L315-340))

**Forward-Backward Pass** ([trainer/distillation.py:209-280](trainer/distillation.py#L209-280)):

```python
def fwdbwd_one_step(self, batch, train_generator):
    # Extract text prompts
    text_prompts = batch["prompts"]

    # Encode text (cached)
    conditional_dict = text_encoder(text_prompts)
    unconditional_dict = text_encoder([negative_prompt] * batch_size)

    if train_generator:
        # DMD generator loss
        generator_loss = model.generator_loss(
            image_or_video_shape=[batch_size, 21, 16, 60, 104],
            conditional_dict, unconditional_dict
        )
        generator_loss.backward()
        generator.clip_grad_norm_(max_grad_norm=10.0)
        return log_dict
    else:
        # Critic denoising loss
        critic_loss = model.critic_loss(
            image_or_video_shape, conditional_dict, unconditional_dict
        )
        critic_loss.backward()
        fake_score.clip_grad_norm_(max_grad_norm=10.0)
        return log_dict
```

---

### 6. Loss Functions

#### Loss Types ([utils/loss.py](utils/loss.py))

**1. X0 Prediction Loss** ([utils/loss.py:27-35](utils/loss.py#L27-35)):
```python
loss = mean((x - x_pred)^2)
```
- Directly regresses clean data

**2. V-Prediction Loss** ([utils/loss.py:38-47](utils/loss.py#L38-47)):
```python
weights = 1 / (1 - alpha_t)
loss = mean(weights * (x - x_pred)^2)
```
- Weighted by noise schedule

**3. Noise Prediction Loss** ([utils/loss.py:50-58](utils/loss.py#L50-58)):
```python
loss = mean((noise - noise_pred)^2)
```
- Standard diffusion objective

**4. Flow Prediction Loss** ([utils/loss.py:61-69](utils/loss.py#L61-69)):
```python
flow_pred = noise - x
loss = mean((flow_pred - target_flow)^2)
```
- Flow matching objective (default for Self-Forcing)

---

## Training Configuration

### Default Config ([configs/self_forcing_dmd.yaml](configs/self_forcing_dmd.yaml))

```yaml
# Model Configuration
generator_ckpt: checkpoints/ode_init.pt      # ODE-initialized weights
real_name: Wan2.1-T2V-14B                    # Teacher model (14B params)
fake_name: Wan2.1-T2V-1.3B                   # Student/critic (1.3B params)

# Denoising Steps
denoising_step_list: [1000, 750, 500, 250]  # Backward simulation timesteps
warp_denoising_step: true                    # Map to actual scheduler timesteps

# Training Hyperparameters
num_train_timestep: 1000
timestep_shift: 5.0                          # Flow matching shift
guidance_scale: 3.0                          # CFG scale for teacher
denoising_loss_type: flow                    # Flow matching loss

# Optimization
lr: 2.0e-6                                   # Generator learning rate
lr_critic: 4.0e-7                            # Critic learning rate (5x lower)
beta1: 0.0                                   # Adam beta1
beta2: 0.999                                 # Adam beta2
weight_decay: 0.01
batch_size: 1                                # Per-GPU batch size
total_batch_size: 64                         # Across all GPUs

# Autoregressive Settings
num_frame_per_block: 3                       # Frames per AR step
num_training_frames: 21                      # Total frames
independent_first_frame: false
same_step_across_blocks: true                # Use same exit for all blocks
last_step_only: false                        # Exit at random steps
context_noise: 0                             # Noise added to context

# Training Schedule
dfake_gen_update_ratio: 5                    # Update generator every 5 iters
ema_weight: 0.99
ema_start_step: 200
log_iters: 50                                # Save checkpoint interval

# Data
data_path: prompts/vidprom_filtered_extended.txt  # Text prompts only
negative_prompt: "色调艳丽，过曝，静态，..."     # Chinese negative prompt

# Distributed Training
sharding_strategy: hybrid_full               # FSDP strategy
mixed_precision: true                        # Use bfloat16
gradient_checkpointing: true
```

**Hardware Requirements**:
- **Training**: 64 H100 GPUs, ~2 hours for 600 iterations
- **Inference**: Single RTX 4090 (real-time streaming)

---

## Data Requirements

### Training Data
**Key Innovation: Data-Free Training**
- Uses **text prompts only** ([configs/self_forcing_dmd.yaml:31](configs/self_forcing_dmd.yaml#L31))
- No video data required (backward simulation generates synthetic data)
- Relies on ODE-initialized checkpoint ([configs/self_forcing_dmd.yaml:1](configs/self_forcing_dmd.yaml#L1))

### ODE Initialization
The ODE initialization process:
1. Train standard diffusion model on video data
2. Solve ODE to find deterministic mapping
3. Use as initialization for Self-Forcing
4. Details: See [CausVid paper](https://github.com/tianweiy/CausVid)

### Text Encoder
- **Model**: UMT5-XXL ([utils/wan_wrapper.py:18-30](utils/wan_wrapper.py#L18-30))
- **Input**: Chinese/English text prompts
- **Output**: 512-token embeddings with mask
- **Recommendation**: Use long, detailed prompts (model trained on extended prompts)

---

## Inference Pipeline

### Autoregressive Generation

**Pipeline Flow**:
1. **Text Encoding** → Prompt embeddings
2. **Initial Frame** (optional) → Image latent via VAE
3. **Autoregressive Loop**:
   - Generate `num_frame_per_block` frames
   - Cache KV states
   - Update context with generated frames
   - Repeat until desired length
4. **VAE Decoding** → Pixel-space video

### KV Cache Management
- **Global Index**: Total frames generated
- **Local Index**: Position in circular buffer
- **Eviction**: FIFO with sink tokens preserved
- **Size**: 21 frames × 1560 tokens/frame = 32,760 tokens

### Speed Optimizations
- **torch.compile**: Recommended for best performance
- **TAEHV-VAE**: Faster decoding (slight quality loss)
- **FP8 Linear**: Further speedup (more quality loss)

---

## Key Innovations Summary

### 1. Self-Forcing Training
- **Autoregressive rollout during training** ([pipeline/self_forcing_training.py:139-219](pipeline/self_forcing_training.py#L139-219))
- Matches inference-time distribution
- Eliminates train-test gap

### 2. Backward Simulation
- **Consistency-based trajectory generation** ([model/base.py:182-204](model/base.py#L182-204))
- No real video data needed
- Random exit points for efficiency

### 3. Chunk-wise Generation
- **Variable-length training** ([model/base.py:131-147](model/base.py#L131-147))
- 3-frame blocks with KV caching
- Efficient long-form video generation

### 4. Distribution Matching
- **DMD loss** ([model/dmd.py:54-126](model/dmd.py#L54-126)): KL-divergence minimization
- **SiD loss** ([model/sid.py:47-145](model/sid.py#L47-145)): Score identity regularization
- Teacher-student distillation

### 5. Critic Training
- **Separate denoising loss** ([model/dmd.py:237-332](model/dmd.py#L237-332))
- Lower learning rate (5x)
- Stabilizes adversarial training

---

## File Reference Index

### Core Model Files
- [model/base.py](model/base.py): Base model architecture, timestep generation
- [model/dmd.py](model/dmd.py): DMD training algorithm
- [model/sid.py](model/sid.py): SiD training algorithm
- [model/causvid.py](model/causvid.py): CausVid baseline

### Architecture Files
- [wan/modules/model.py](wan/modules/model.py): Standard Wan transformer
- [wan/modules/causal_model.py](wan/modules/causal_model.py): Causal attention with KV cache
- [wan/modules/vae.py](wan/modules/vae.py): Video VAE
- [wan/modules/t5.py](wan/modules/t5.py): UMT5 text encoder

### Training Files
- [trainer/distillation.py](trainer/distillation.py): Main training orchestration
- [pipeline/self_forcing_training.py](pipeline/self_forcing_training.py): Autoregressive pipeline
- [utils/loss.py](utils/loss.py): Loss function implementations
- [utils/scheduler.py](utils/scheduler.py): Flow matching scheduler

### Utility Files
- [utils/wan_wrapper.py](utils/wan_wrapper.py): Model/VAE/encoder wrappers
- [utils/distributed.py](utils/distributed.py): FSDP utilities
- [utils/dataset.py](utils/dataset.py): Dataset loaders

### Configuration Files
- [configs/self_forcing_dmd.yaml](configs/self_forcing_dmd.yaml): DMD training config
- [configs/self_forcing_sid.yaml](configs/self_forcing_sid.yaml): SiD training config
- [configs/default_config.yaml](configs/default_config.yaml): Default hyperparameters

### Entry Points
- [train.py](train.py): Training script
- [inference.py](inference.py): CLI inference
- [demo.py](demo.py): GUI demo

---

## Mathematical Formulation

### DMD Loss (Core Equation)

**Generator Objective**:
```
L_gen = 0.5 * E[||x₀ - (x₀ - ∇_KL)||²]
```
where:
- `x₀`: Generated clean latent
- `∇_KL = (s_fake(x_t) - s_real(x_t)) / ||x₀ - s_real(x_t)||`: Normalized KL gradient
- `s_fake`, `s_real`: Student and teacher score functions

**Critic Objective**:
```
L_critic = E[||v_flow - (ε - x₀)||²]
```
where:
- `v_flow`: Predicted flow
- `ε`: Noise
- `x₀`: Generated sample (detached)

### Flow Matching
```
x_t = (1 - σ_t) * x₀ + σ_t * ε
v_t = ε - x₀
x₀ = x_t - σ_t * v_t
```

---

## Performance Characteristics

### Memory Usage
- **Generator**: ~5GB (1.3B params in bfloat16)
- **Teacher**: ~26GB (14B params)
- **KV Cache**: ~1.5GB (per sample, 21 frames)
- **Peak Training**: ~40GB per GPU (with gradient checkpointing)

### Speed
- **Training**: 600 iterations in 2 hours (64 H100s)
- **Inference**: Real-time on RTX 4090 (with optimizations)
- **Throughput**: ~1 video/sec (21 frames, 480×832)

### Scalability
- **Frames**: Supports arbitrary length (tested up to 81+ frames)
- **Batch Size**: 1 per GPU (memory constrained)
- **GPUs**: Scales linearly with FSDP

---

## Citations and References

**Papers**:
- Self-Forcing: [arXiv:2506.08009](https://arxiv.org/abs/2506.08009)
- DMD: [arXiv:2311.18828](https://arxiv.org/abs/2311.18828)
- DMD2: [arXiv:2405.14867](https://arxiv.org/abs/2405.14867)
- Consistency Models: [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- CausVid: [GitHub](https://github.com/tianweiy/CausVid)

**Acknowledgements**:
- Built on [CausVid](https://github.com/tianweiy/CausVid) by Tianwei Yin
- Uses [Wan2.1](https://github.com/Wan-Video/Wan2.1) video model

---

## Quick Start Summary

### Training
```bash
# Download ODE checkpoint and prompts
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts

# Train with DMD (64 GPUs)
torchrun --nnodes=8 --nproc_per_node=8 \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd
```

### Inference
```bash
# Download model
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .

# Generate videos
python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --use_ema
```

### GUI Demo
```bash
python demo.py
```

---

*This document provides a comprehensive technical overview of the Self-Forcing codebase. For implementation details, refer to the cited line numbers and files.*
