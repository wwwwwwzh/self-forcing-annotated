# KV Cache Architecture: Deep Dive

## TL;DR Answer to Your Questions

**Q1: Do they keep KV cache as an outside reservoir and use it for every denoising step?**
**A: YES** - The KV cache is a **single shared reservoir** stored outside the DiT backbone that is **reused and updated** across ALL denoising steps within an autoregressive block.

**Q2: Do they store per denoising step KV cache?**
**A: NO** - There is **ONE KV cache per transformer layer** (30 layers total), NOT per denoising step. The cache accumulates context from ALL previous frames but gets updated with different timesteps during the multi-step denoising process.

---

## Detailed Architecture

### 1. KV Cache Storage Structure

The KV cache is organized as a **list of 30 dictionaries** (one per transformer layer):

```python
# From pipeline/self_forcing_training.py:239-253
self.kv_cache1 = [
    {
        "k": torch.zeros([batch_size, kv_cache_size, 12, 128]),  # Keys
        "v": torch.zeros([batch_size, kv_cache_size, 12, 128]),  # Values
        "global_end_index": 0,  # Total frames generated
        "local_end_index": 0    # Position in circular buffer
    }
    for _ in range(30)  # 30 transformer blocks
]
```

**Key Parameters**:
- `kv_cache_size = 21 * 1560 = 32,760` tokens (21 frames max)
- `12` attention heads
- `128` head dimension
- **Total size per layer**: ~1.5GB in bfloat16

**Location**: [pipeline/self_forcing_training.py:239-253](pipeline/self_forcing_training.py#L239-253)

---

## 2. How KV Cache Interacts with DiT Backbone

### **Single Shared Reservoir Paradigm**

The KV cache acts as an **external memory bank** that is:
1. **Initialized once** per video generation
2. **Shared across ALL denoising steps**
3. **Accumulated across frames** (autoregressive)
4. **Updated in-place** with new frames

### **Lifecycle During Generation**

```
┌─────────────────────────────────────────────────────────────┐
│                    INITIALIZATION                           │
│  _initialize_kv_cache() → Creates empty 30-layer cache     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              AUTOREGRESSIVE LOOP (Frame 0)                  │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Generate at t=1000                                 │
│    ├─ DiT forward pass                                      │
│    ├─ Compute Q, K, V for current frame                     │
│    ├─ UPDATE: kv_cache["k"][0:1560] = K                     │
│    ├─ UPDATE: kv_cache["v"][0:1560] = V                     │
│    └─ Attend: Q @ [K]  (only current frame)                 │
│                                                              │
│  Step 2: Generate at t=750                                  │
│    ├─ DiT forward pass (SAME CACHE)                         │
│    ├─ Compute Q, K, V for current frame                     │
│    ├─ UPDATE: kv_cache["k"][0:1560] = K_new                 │
│    ├─ UPDATE: kv_cache["v"][0:1560] = V_new                 │
│    └─ Attend: Q @ [K_new]  (cache overwritten!)             │
│                                                              │
│  Step 3: Generate at t=500 ... (continues)                  │
│                                                              │
│  Step 4: Context update at t=0                              │
│    ├─ Re-run DiT with denoised output                       │
│    ├─ UPDATE: kv_cache["k"][0:1560] = K_clean               │
│    └─ Cache now contains CLEAN context for next frame       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            AUTOREGRESSIVE LOOP (Frame 1-3)                  │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Generate 3 new frames at t=1000                    │
│    ├─ DiT forward pass                                      │
│    ├─ Compute Q, K, V for 3 frames                          │
│    ├─ UPDATE: kv_cache["k"][1560:6240] = K                  │
│    ├─ UPDATE: kv_cache["v"][1560:6240] = V                  │
│    └─ Attend: Q @ [K_frame0, K_frame1-3]  (causal)          │
│            ↑                                                 │
│       USES FRAME 0 CONTEXT FROM CACHE                        │
│                                                              │
│  Step 2-4: Continue denoising (same cache updates)          │
│                                                              │
│  Step 5: Context update                                     │
│    └─ UPDATE: kv_cache stores clean frames 0-3              │
└─────────────────────────────────────────────────────────────┘
                         (repeats)
```

**Key Code Location**: [pipeline/self_forcing_training.py:140-219](pipeline/self_forcing_training.py#L140-219)

---

## 3. Cache Update Mechanism During Denoising

### **Critical Insight**: Cache is Overwritten, Not Accumulated Per Step

Looking at [pipeline/self_forcing_training.py:145-194](pipeline/self_forcing_training.py#L145-194):

```python
# Spatial denoising loop (multiple timesteps)
for index, current_timestep in enumerate(denoising_step_list):  # [1000, 750, 500, 250]
    timestep = torch.ones([batch_size, current_num_frames]) * current_timestep

    # SAME kv_cache1 passed to all denoising steps!
    _, denoised_pred = self.generator(
        noisy_input,
        timestep=timestep,
        kv_cache=self.kv_cache1,  # ← SHARED CACHE
        current_start=current_start_frame * 1560
    )
```

**What Happens Inside DiT** ([wan/modules/causal_model.py:193-235](wan/modules/causal_model.py#L193-235)):

```python
def forward(self, x, kv_cache=None, current_start=0):
    # Compute Q, K, V from input
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    # Calculate positions in cache
    current_end = current_start + q.shape[1]  # e.g., 1560 for 1 frame
    local_start_index = current_start
    local_end_index = current_end

    # OVERWRITE cache at same position for each timestep!
    kv_cache["k"][:, local_start_index:local_end_index] = k  # ← OVERWRITES
    kv_cache["v"][:, local_start_index:local_end_index] = v

    # Attend over ALL cached frames
    x = attention(
        q,
        kv_cache["k"][:, 0:local_end_index],  # All previous + current
        kv_cache["v"][:, 0:local_end_index]
    )

    return x
```

### **Example with Numbers**

```
Frame 0 (t=1000):
  - DiT computes K₀(t=1000), V₀(t=1000)
  - kv_cache["k"][0:1560] = K₀(t=1000)
  - kv_cache["v"][0:1560] = V₀(t=1000)

Frame 0 (t=750):
  - DiT computes K₀(t=750), V₀(t=750)  [different because timestep changed!]
  - kv_cache["k"][0:1560] = K₀(t=750)  ← OVERWRITES previous
  - kv_cache["v"][0:1560] = V₀(t=750)

Frame 0 (t=500):
  - kv_cache["k"][0:1560] = K₀(t=500)  ← OVERWRITES again

... continues until final timestep

Frame 0 (t=0, context update):
  - kv_cache["k"][0:1560] = K₀(t=0)    ← FINAL clean context
  - This is what Frame 1-3 will attend to!
```

---

## 4. Context Update Step (Critical Detail)

After finishing denoising, they **re-run the model at t=0** ([pipeline/self_forcing_training.py:199-216](pipeline/self_forcing_training.py#L199-216)):

```python
# Step 3.3: rerun with timestep zero to update the cache
context_timestep = torch.ones_like(timestep) * self.context_noise  # Usually 0

# Optionally add noise to context (default: 0)
denoised_pred_noisy = scheduler.add_noise(
    denoised_pred,
    torch.randn_like(denoised_pred),
    context_timestep
)

with torch.no_grad():
    self.generator(
        noisy_image_or_video=denoised_pred_noisy,
        timestep=context_timestep,
        kv_cache=self.kv_cache1,  # Updates cache with "clean" features
        current_start=current_start_frame * 1560
    )
```

**Why?**
- During denoising, cache contains K/V from intermediate noisy states
- After denoising, we want cache to contain K/V from **clean/final output**
- This simulates inference conditions where next frames attend to clean context

---

## 5. Cache Accumulation Across Frames

The `current_start` parameter controls **where in the cache** to write:

```python
# Frame 0: current_start = 0
generator(..., kv_cache, current_start=0)
# Updates kv_cache["k"][0:1560]

# Frame 1-3: current_start = 1560  (1 frame * 1560 tokens)
generator(..., kv_cache, current_start=1560)
# Updates kv_cache["k"][1560:6240]  (3 frames * 1560 = 4680 tokens)
# Can still attend to kv_cache["k"][0:1560] from Frame 0!

# Frame 4-6: current_start = 6240
generator(..., kv_cache, current_start=6240)
# Updates kv_cache["k"][6240:10920]
# Can attend to ALL previous frames [0:6240]
```

**Global vs Local Indices** ([wan/modules/causal_model.py:201-235](wan/modules/causal_model.py#L201-235)):
- `global_end_index`: Total frames generated (never decreases)
- `local_end_index`: Position in circular buffer (can wrap)
- Supports **local attention** with sliding window

---

## 6. Cross-Attention Cache (Separate!)

There's also a **cross-attention cache** for text embeddings ([pipeline/self_forcing_training.py:255-267](pipeline/self_forcing_training.py#L255-267)):

```python
self.crossattn_cache = [
    {
        "k": torch.zeros([batch_size, 512, 12, 128]),  # Text tokens (fixed)
        "v": torch.zeros([batch_size, 512, 12, 128]),
        "is_init": False
    }
    for _ in range(30)
]
```

**Difference**:
- Computed **once** from text embeddings
- **Never changes** during generation
- Reused across all frames and timesteps

---

## 7. Training vs. Inference Behavior

### **Training Mode** ([pipeline/self_forcing_training.py](pipeline/self_forcing_training.py))
- KV cache initialized fresh for each training sample
- Randomly exits denoising at different steps (efficiency)
- Gradients only computed for last 21 frames

### **Inference Mode** ([pipeline/causal_inference.py](pipeline/causal_inference.py))
- KV cache **persists across batches** (can generate arbitrarily long videos)
- Cache can be **reset** between videos ([pipeline/causal_inference.py:125-132](pipeline/causal_inference.py#L125-132))
- All denoising steps executed (no random exits)

---

## 8. Memory Management

### **Circular Buffer for Long Videos**

When cache fills up ([wan/modules/causal_model.py:206-222](wan/modules/causal_model.py#L206-222)):

```python
if local_end_index > kv_cache_size:
    # Evict oldest tokens (FIFO)
    num_evicted = num_new_tokens + local_end_index - kv_cache_size
    num_kept = local_end_index - num_evicted - sink_tokens

    # Shift left (discard oldest)
    kv_cache["k"][:, sink_tokens:sink_tokens+num_kept] = \
        kv_cache["k"][:, sink_tokens+num_evicted:...].clone()

    # Insert new at end
    kv_cache["k"][:, new_start:new_end] = new_k
```

**Sink Tokens**: First `sink_size` frames are **never evicted** (attention stability)

---

## 9. Why This Design?

### **Advantages**
1. **Memory Efficiency**: One cache for all timesteps (not 4x for 4 denoising steps)
2. **Inference Matching**: Cache behavior identical during training and inference
3. **Flexibility**: Can generate arbitrarily long videos with fixed memory
4. **Speed**: Avoids recomputing K/V for previous frames

### **Key Insight**
The cache doesn't need to store intermediate denoising states because:
- Only the **final clean output** matters for future frames
- Intermediate K/V from t=1000, 750, 500 are **transient**
- Context update at t=0 ensures cache has **clean features**

---

## 10. Visual Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                         DiT BACKBONE                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐   │
│  │ Block 0  │  │ Block 1  │  │ Block 2  │  ...  │ Block 29 │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘   │
│       │             │             │                    │          │
│       ↓             ↓             ↓                    ↓          │
└───────┼─────────────┼─────────────┼────────────────────┼──────────┘
        │             │             │                    │
        │     READ/WRITE to external KV cache           │
        ↓             ↓             ↓                    ↓
┌──────────────────────────────────────────────────────────────────┐
│                    EXTERNAL KV CACHE                             │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Layer 0: [K: 32760×12×128, V: 32760×12×128]         │       │
│  │          global_end=6240, local_end=6240            │       │
│  └──────────────────────────────────────────────────────┘       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Layer 1: [K: 32760×12×128, V: 32760×12×128]         │       │
│  └──────────────────────────────────────────────────────┘       │
│  ...                                                              │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Layer 29: [K: 32760×12×128, V: 32760×12×128]        │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
         ↑                                           ↑
         └─── Shared across ALL timesteps ───────────┘
              (t=1000, 750, 500, 250, 0)
```

---

## 11. Code Flow Example

**Generating Frame 1-3 with 4 denoising steps:**

```python
# 1. Initialize (once)
pipeline._initialize_kv_cache()  # kv_cache1 = [30 empty dicts]

# 2. Encode Frame 0 (context)
generator(frame0, t=0, kv_cache=kv_cache1, current_start=0)
# kv_cache1[layer_i]["k"][0:1560] = K_frame0_clean

# 3. Start Frame 1-3
current_start_frame = 1
noisy_input = noise[:, 0:3]  # 3 frames

# 4. Denoising loop
for timestep in [1000, 750, 500, 250]:
    # SAME kv_cache1 for all timesteps!
    _, pred = generator(
        noisy_input,
        t=timestep,
        kv_cache=kv_cache1,  # ← SHARED
        current_start=1560   # Start after Frame 0
    )
    # Inside: kv_cache1[i]["k"][1560:6240] gets OVERWRITTEN each iteration

    noisy_input = add_noise(pred, next_timestep)

# 5. Context update
generator(pred, t=0, kv_cache=kv_cache1, current_start=1560)
# kv_cache1[i]["k"][1560:6240] = K_frames1-3_clean

# 6. Now Frame 4-6 can use clean context from Frames 0-3
#    by reading kv_cache1[i]["k"][0:6240]
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Number of KV caches** | 1 per transformer layer (30 total) |
| **Storage duration** | Entire video generation |
| **Updated when** | Every DiT forward pass |
| **Updated where** | Position = `current_start : current_start + num_tokens` |
| **Per-timestep storage?** | ❌ NO - same cache overwritten |
| **Per-frame storage?** | ✅ YES - accumulates across frames |
| **Size** | 21 frames × 1560 tokens = 32,760 max |
| **Eviction policy** | FIFO with sink tokens |
| **Used for** | Causal self-attention only (not cross-attention) |

---

## Key Takeaway

The KV cache is a **stateful external memory** that:
- Lives **outside** the DiT weights
- Is **shared** across all denoising timesteps within a frame block
- **Accumulates** clean context from previous frames
- Gets **overwritten** at the same position during multi-step denoising
- Is **finalized** with a context update at t=0 before moving to next frames

This design enables efficient autoregressive generation while maintaining consistency between training and inference.
