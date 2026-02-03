# Model Architecture Breakdown: All Models Explained

## TL;DR: How Many Models?

**Answer: 3 Main Neural Networks + 2 Frozen Encoders**

| Model | Name in Code | Size | Trainable? | Role |
|-------|--------------|------|------------|------|
| **1. Generator** | `self.generator` | 1.3B params | ✅ YES | Student model (causal autoregressive) |
| **2. Teacher** | `self.real_score` | 14B params | ❌ NO (frozen) | Teacher model (bidirectional) |
| **3. Critic** | `self.fake_score` | 1.3B params | ✅ YES | Student critic (bidirectional) |
| **4. Text Encoder** | `self.text_encoder` | UMT5-XXL | ❌ NO (frozen) | Text → embeddings |
| **5. VAE** | `self.vae` | ~100M params | ❌ NO (frozen) | Latent ↔ pixel conversion |

**Confusing Terminology Alert!**
- "Generator" = Student's **causal** model (what you inference with)
- "Fake Score" = Student's **bidirectional** critic/discriminator
- "Real Score" = Teacher's **bidirectional** model

Let me break this down in detail!

---

## Complete Model Breakdown

### Model 1: Generator (The Student - Causal)

**Code Location**: [model/base.py:30-31](model/base.py#L30-31)
```python
self.generator = WanDiffusionWrapper(
    **getattr(args, "model_kwargs", {}),
    is_causal=True  # ← KEY: Causal version!
)
self.generator.model.requires_grad_(True)
```

**Configuration**: [configs/self_forcing_dmd.yaml:1](configs/self_forcing_dmd.yaml#L1)
```yaml
generator_ckpt: checkpoints/ode_init.pt  # Initialized with ODE
# Uses default Wan2.1-T2V-1.3B model
```

**Properties**:
- **Architecture**: `CausalWanModel` ([wan/modules/causal_model.py](wan/modules/causal_model.py))
- **Size**: **1.3 billion parameters**
- **Trainable**: ✅ YES
- **Attention**: Causal (autoregressive) with KV caching
- **Purpose**: Generate videos frame-by-frame at inference
- **Output**: Clean latents x₀ from noise

**Key Features**:
- Uses **blockwise causal attention** (3-frame chunks)
- Supports **KV caching** for efficient generation
- Has **independent first frame** option for I2V
- **This is what you use for inference!**

**Training**:
- Optimizer: AdamW with lr=2e-6 ([trainer/distillation.py:106-112](trainer/distillation.py#L106-112))
- Updated every 5 iterations (generator update ratio)
- Uses EMA for stability (decay=0.99)

---

### Model 2: Real Score (The Teacher - Bidirectional)

**Code Location**: [model/base.py:33-34](model/base.py#L33-34)
```python
self.real_score = WanDiffusionWrapper(
    model_name=self.real_model_name,  # From config
    is_causal=False  # ← Bidirectional!
)
self.real_score.model.requires_grad_(False)  # Frozen!
```

**Configuration**: [configs/self_forcing_dmd.yaml:5](configs/self_forcing_dmd.yaml#L5)
```yaml
real_name: Wan2.1-T2V-14B  # The big teacher model
```

**Properties**:
- **Architecture**: Standard `WanModel` (bidirectional)
- **Size**: **14 billion parameters** (10x larger than student!)
- **Trainable**: ❌ NO - completely frozen
- **Attention**: Full bidirectional (no causal masking)
- **Purpose**: Provide "ground truth" score for distillation
- **Output**: Denoised prediction from noisy input

**Why Bidirectional?**
- Can attend to **all frames** (past and future)
- More accurate but **cannot be used autoregressively**
- Acts as the **oracle** for what good videos look like

**Role in Training**:
- Never updated during training
- Provides target distribution for student to match
- Used in DMD loss to compute KL gradient

---

### Model 3: Fake Score (The Critic - Bidirectional)

**Code Location**: [model/base.py:36-37](model/base.py#L36-37)
```python
self.fake_score = WanDiffusionWrapper(
    model_name=self.fake_model_name,  # Usually same as generator size
    is_causal=False  # ← Bidirectional!
)
self.fake_score.model.requires_grad_(True)  # Trainable!
```

**Configuration**: [configs/self_forcing_dmd.yaml:28](configs/self_forcing_dmd.yaml#L28) (implicit - no fake_name specified)
```yaml
# Uses default Wan2.1-T2V-1.3B (same size as generator)
```

**Properties**:
- **Architecture**: Standard `WanModel` (bidirectional)
- **Size**: **1.3 billion parameters** (same as generator)
- **Trainable**: ✅ YES
- **Attention**: Full bidirectional
- **Purpose**: Learn to denoise student's generated samples
- **Output**: Denoised prediction (same as teacher)

**Why Do We Need This?**
Great question! Here's the reasoning:

1. **Generator is causal** (autoregressive) - can only attend to past
2. **Teacher is bidirectional** - can attend to all frames
3. **Critic bridges the gap** - learns to evaluate student samples with full context

**The Critic's Job**:
- Takes generator's samples x₀
- Evaluates them with **bidirectional attention** (like teacher)
- Learns to denoise similarly to teacher
- Provides **student-side score** for KL gradient

**Training** ([trainer/distillation.py:114-120](trainer/distillation.py#L114-120)):
```python
self.critic_optimizer = torch.optim.AdamW(
    self.model.fake_score.parameters(),
    lr=4e-7,  # 5x LOWER than generator!
    betas=(0.0, 0.999)
)
```
- Updated **every iteration** (more frequent than generator)
- Lower learning rate for stability

---

### Model 4: Text Encoder (Frozen)

**Code Location**: [model/base.py:39-40](model/base.py#L39-40)
```python
self.text_encoder = WanTextEncoder()
self.text_encoder.requires_grad_(False)
```

**Implementation**: [utils/wan_wrapper.py:14-50](utils/wan_wrapper.py#L14-50)
```python
class WanTextEncoder(torch.nn.Module):
    def __init__(self):
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            dtype=torch.float32
        )
        # Load from: wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
```

**Properties**:
- **Architecture**: UMT5-XXL encoder
- **Size**: Several billion parameters
- **Trainable**: ❌ NO
- **Input**: Text strings (Chinese/English)
- **Output**: 512-token embeddings

**Role**:
- Converts prompts to embeddings
- Used by all three diffusion models
- Frozen to save compute

---

### Model 5: VAE (Frozen)

**Code Location**: [model/base.py:42-43](model/base.py#L42-43)
```python
self.vae = WanVAEWrapper()
self.vae.requires_grad_(False)
```

**Implementation**: [utils/wan_wrapper.py:53-112](utils/wan_wrapper.py#L53-112)
```python
class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16
        )
```

**Properties**:
- **Architecture**: 3D Video VAE
- **Size**: ~100M parameters
- **Trainable**: ❌ NO
- **Compression**: 8×8×4 spatiotemporal compression
- **Latent Channels**: 16

**Role**:
- **Encode**: Pixel video [B,C,T,H,W] → Latent [B,T,16,H/8,W/8]
- **Decode**: Latent → Pixel video
- Only used for visualization during training

---

## The "Student" vs "Teacher" Terminology Confusion

### Common Confusion:

People often think:
- "Student" = Generator only
- "Teacher" = Real score only

### Actual Reality:

**"Student"** has TWO models:
1. **Generator** (causal) - produces samples
2. **Fake Score/Critic** (bidirectional) - evaluates samples

**"Teacher"** has ONE model:
1. **Real Score** (bidirectional) - frozen oracle

### Why This Design?

```
Training Flow:
┌──────────────────────────────────────────────────────────┐
│                    STUDENT SYSTEM                        │
│  ┌─────────────┐                  ┌────────────┐        │
│  │  Generator  │ ──samples x₀──→ │   Critic   │        │
│  │  (Causal)   │                  │(Bidirectional)│     │
│  │  1.3B       │                  │   1.3B     │        │
│  └─────────────┘                  └──────┬─────┘        │
│                                           │               │
│                                      pred_fake            │
└───────────────────────────────────────────┼──────────────┘
                                            │
                   ┌────────────────────────┴────────┐
                   │      DMD Loss Computation       │
                   │  grad = pred_fake - pred_real   │
                   └────────────────────────┬────────┘
                                            │
┌───────────────────────────────────────────┼──────────────┐
│                   TEACHER SYSTEM          │              │
│                                      pred_real           │
│                                           │               │
│                                    ┌──────┴─────┐        │
│                                    │   Teacher  │        │
│                                    │(Bidirectional)│     │
│                                    │    14B     │        │
│                                    └────────────┘        │
└──────────────────────────────────────────────────────────┘
```

---

## Training Data Flow

### Generator Update (Every 5 Iterations)

**Code**: [model/dmd.py:196-235](model/dmd.py#L196-235)

```python
# 1. Generate samples
x₀ = generator.run_with_backward_simulation()  # Uses KV cache, causal

# 2. Add noise
x_t = scheduler.add_noise(x₀, noise, timestep)

# 3. Get both scores (bidirectional evaluation)
with torch.no_grad():
    pred_fake = fake_score(x_t)   # Student critic
    pred_real = real_score(x_t)   # Teacher
    grad = (pred_fake - pred_real) / normalizer

# 4. DMD loss
loss = ||x₀ - (x₀ - grad).detach()||²

# 5. Backprop ONLY through generator
loss.backward()  # Updates generator weights
generator_optimizer.step()
```

**Gradient Flow**:
- ✅ Generator parameters updated
- ❌ Critic NOT updated
- ❌ Teacher NOT updated

---

### Critic Update (Every Iteration)

**Code**: [model/dmd.py:237-332](model/dmd.py#L237-332)

```python
# 1. Generate samples (no gradient to generator)
with torch.no_grad():
    x₀ = generator.run_with_backward_simulation()

# 2. Add noise
x_t = scheduler.add_noise(x₀.detach(), noise, timestep)

# 3. Critic denoises
pred_fake = fake_score(x_t)

# 4. Standard denoising loss
loss = denoising_loss_func(
    x=x₀.detach(),
    x_pred=pred_fake,
    ...
)

# 5. Backprop ONLY through critic
loss.backward()  # Updates critic weights
critic_optimizer.step()
```

**Gradient Flow**:
- ❌ Generator NOT updated
- ✅ Critic parameters updated
- ❌ Teacher NOT updated

---

## Parameter Counts and Memory

### Training Memory Breakdown

From [configs/self_forcing_dmd.yaml](configs/self_forcing_dmd.yaml):

| Component | Size | Memory (FP16) | Trainable |
|-----------|------|---------------|-----------|
| Generator | 1.3B | ~2.6 GB | ✅ YES (+ gradients ~5 GB) |
| Teacher (real_score) | 14B | ~28 GB | ❌ NO |
| Critic (fake_score) | 1.3B | ~2.6 GB | ✅ YES (+ gradients ~5 GB) |
| Text Encoder | ~4B | ~8 GB | ❌ NO |
| VAE | ~100M | ~200 MB | ❌ NO |
| **Total** | **~21B** | **~49 GB** | **Only 2.6B trainable** |

With FSDP sharding across 64 GPUs:
- Each GPU: ~1 GB model weights
- Plus activations, KV cache, gradients
- Total: ~40GB per H100 GPU

---

## Why Three Diffusion Models?

### The Core Problem:

1. **Goal**: Train causal student to match bidirectional teacher
2. **Challenge**: They have different attention patterns!
   - Generator: Can only see past (causal)
   - Teacher: Can see past + future (bidirectional)

### The Solution: Three Models

**Generator (Causal)**:
- Generates samples autoregressively
- Uses KV caching for efficiency
- **Cannot be evaluated with bidirectional attention**

**Critic (Bidirectional)**:
- Evaluates generator's samples
- Uses same attention as teacher
- Acts as "student's score function"

**Teacher (Bidirectional)**:
- Provides target distribution
- Much larger (14B vs 1.3B)
- Frozen oracle

### The DMD Triangle:

```
          Generator (Causal 1.3B)
                 │
                 │ generates samples
                 ↓
               x₀ samples
                 │
         ┌───────┴───────┐
         │               │
    add noise        add noise
         │               │
         ↓               ↓
    ┌────────┐      ┌────────┐
    │ Critic │      │Teacher │
    │  1.3B  │      │  14B   │
    │Bidir   │      │ Bidir  │
    └────┬───┘      └───┬────┘
         │              │
    pred_fake      pred_real
         │              │
         └──────┬───────┘
                │
         KL Gradient (∇_KL)
                │
                ↓
        Update Generator
```

---

## Inference: Which Models Are Used?

### At Inference Time:

**Models Used**:
- ✅ **Generator** (causal) - the only one needed!
- ✅ **Text Encoder** - for prompt embeddings
- ✅ **VAE** - for latent → pixel decoding

**Models NOT Used**:
- ❌ Critic (fake_score) - not needed
- ❌ Teacher (real_score) - not needed

**Memory**: Only ~3 GB (generator + text encoder + VAE)

---

## Model Naming: The Confusing Convention

### Why "Generator" and "Fake Score"?

The naming comes from **GAN terminology**:

**In GANs**:
- Generator: Creates fake samples
- Discriminator: Distinguishes real vs fake

**In DMD** (adapted from GAN distillation):
- Generator: Creates fake samples (student's causal model)
- Fake Score: Scores fake samples (student's bidirectional critic)
- Real Score: Scores real distribution (teacher)

### Better Names (My Opinion):

| Current Name | What It Actually Is | Better Name |
|--------------|---------------------|-------------|
| `generator` | Student causal model | `student_generator` |
| `fake_score` | Student bidirectional critic | `student_critic` |
| `real_score` | Teacher model | `teacher_model` |

---

## Summary Table

| Model | Architecture | Attention | Params | Trainable | Update Freq | Purpose |
|-------|--------------|-----------|--------|-----------|-------------|---------|
| **Generator** | CausalWanModel | Causal (with KV cache) | 1.3B | ✅ Yes | Every 5 iters | Generate videos autoregressively |
| **Critic (fake_score)** | WanModel | Bidirectional | 1.3B | ✅ Yes | Every iter | Evaluate student samples |
| **Teacher (real_score)** | WanModel | Bidirectional | 14B | ❌ No | Never | Provide target distribution |
| **Text Encoder** | UMT5-XXL | N/A | ~4B | ❌ No | Never | Text → embeddings |
| **VAE** | 3D VAE | N/A | ~100M | ❌ No | Never | Latent ↔ pixel |

---

## Key Takeaways

1. **Three diffusion models**, not two:
   - Generator (student, causal)
   - Critic (student, bidirectional)
   - Teacher (frozen, bidirectional)

2. **The "student" has two components**:
   - Generator for producing samples
   - Critic for evaluating samples

3. **Asymmetric training**:
   - Generator updated every 5 iters (slower)
   - Critic updated every iter (faster)
   - Teacher never updated

4. **At inference**: Only generator + encoders needed

5. **The naming is confusing** because it's borrowed from GAN literature, but the architecture makes sense once you understand the roles!

---

## Code References

| Component | Initialization | Training | Config |
|-----------|---------------|----------|--------|
| Generator | [model/base.py:30](model/base.py#L30) | [model/dmd.py:196](model/dmd.py#L196) | [configs:1](configs/self_forcing_dmd.yaml#L1) |
| Critic | [model/base.py:36](model/base.py#L36) | [model/dmd.py:237](model/dmd.py#L237) | [configs:28](configs/self_forcing_dmd.yaml#L28) |
| Teacher | [model/base.py:33](model/base.py#L33) | - (frozen) | [configs:5](configs/self_forcing_dmd.yaml#L5) |
| Optimizers | [trainer:106-120](trainer/distillation.py#L106-120) | [trainer:315-340](trainer/distillation.py#L315-340) | [configs:25-30](configs/self_forcing_dmd.yaml#L25-30) |
