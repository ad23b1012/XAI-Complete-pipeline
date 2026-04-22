# 🧠 XAI Emotion Recognition — Full Pipeline Walkthrough

---

## The Big Picture

The goal is to **not just classify an emotion, but explain *why* the model made that prediction** in plain English — automatically, without a human expert.

To do that, the pipeline chains together three kinds of intelligence:

| Layer | Tool | What it provides |
|---|---|---|
| **Geometric** | MediaPipe + AU Extractor | Measurable, named facial muscle movements |
| **Visual** | POSTER V2 + Grad-ECLIP | Where the neural network was "looking" |
| **Language** | Qwen2.5-0.5B | Converts the above evidence into a human-readable sentence |

Every output fed to the LLM is **derived from objective measurements** — not hallucinated. That's what makes this XAI rather than just AI.

---

## Step 1 — Face Detection with MediaPipe FaceLandmarker

**File:** `src/face_detection/detector.py`

### What happens

You give the pipeline a raw image (any `.jpg`/`.png`). The first job is to find the face and extract its **structure**.

MediaPipe's `FaceLandmarker` (Tasks SDK, not the old legacy API) runs a neural network that detects **468 facial landmarks** — specific anatomical points on the face. Think of these as:

- Eye corners, eyelid edges, iris edges
- Nose tip, nose bridge, nostrils
- Lip corners, lip edges, cupid's bow
- Jawline, cheekbones, forehead

Each landmark is returned as a **normalized (x, y, z) coordinate** — i.e., values between 0 and 1, relative to the image dimensions. So landmark 61 (left lip corner) might be `(0.42, 0.67, -0.003)`.

### Why 468?

The FaceLandmarker model can return up to **478 points** (the extra 10 are iris landmarks). We truncate to exactly 468 because:
- POSTER V2's internal projection layer is built for exactly **468 × 2** inputs
- Iris landmarks add noise with no benefit for emotion recognition

### What gets output

```
FaceDetectionResult:
  landmarks       → (468, 3) numpy array — normalized x, y, z
  landmarks_pixel → (468, 2) numpy array — actual pixel positions in the image
  face_crop_pil   → PIL Image of just the face (with 20% padding around it)
  bbox            → (x1, y1, x2, y2) bounding box
```

The **face crop** is what gets sent to the emotion classifier. The **raw landmarks** go to both the AU extractor and POSTER V2.

> **XAI role:** The landmarks are the geometric "skeleton" of the face. Everything that follows — both the classification and the explanation — is grounded in this structure.

---

## Step 2 — Action Unit (AU) Feature Extraction

**File:** `src/face_detection/au_extractor.py`

### What are Action Units?

The **Facial Action Coding System (FACS)** is a scientific framework developed by Paul Ekman that describes every visible face movement as a combination of muscle contractions called **Action Units (AUs)**.

For example:
- **AU6** = Cheek Raiser (the muscle that raises your cheeks when you genuinely smile)
- **AU12** = Lip Corner Puller (the main muscle that creates a smile)
- **AU4** = Brow Lowerer (furrows the brow — associated with anger or concentration)

AUs are **universal** — they work across cultures and are used clinically by psychologists.

### How the extractor works

The AU extractor takes the 468 landmarks and computes **geometric distances** between specific landmark pairs.

For example, to detect **AU26 (Jaw Drop / mouth open)**:
```
mouth_open_dist = distance(upper_lip_center, lower_lip_center)
normalized = mouth_open_dist / face_width  # normalize by face size
if normalized > 0.03:  → AU26 is ACTIVE
```

This is done for ~10 different AU types, each with a calibrated threshold (tuned on FER2013). Here's the full list:

| AU | Name | Landmark pair measured |
|---|---|---|
| AU1 | Inner Brow Raise | Inner eyebrow point vs. reference |
| AU2 | Outer Brow Raise | Outer eyebrow point vs. reference |
| AU4 | Brow Lowerer | Eyebrow-to-eye distance |
| AU5 | Upper Lid Raise | Upper eyelid height |
| AU6+7 | Cheek Raise / Lid Tightener | Lower eyelid / cheek region |
| AU12 | Lip Corner Puller | Horizontal spread of lip corners |
| AU15 | Lip Corner Depressor | Downward pull of lip corners |
| AU17 | Chin Raiser | Chin point vs. lower lip |
| AU25 | Lips Part | Vertical lip gap |
| AU26 | Jaw Drop | Upper–lower lip center distance |

### What gets output

```
AUExtractionResult:
  active_aus    → ["AU6", "AU12", "AU25"]   (list of active action unit names)
  au_values     → {"AU12": 0.031, "AU6": 0.027, ...}  (raw distance values)
  
Formatted for prompt:
  "AU6 (Cheek Raiser): active [0.027]
   AU12 (Lip Corner Puller): active [0.031]
   AU25 (Lips Part): active [0.022]"
```

> **XAI role:** AUs translate the raw numbers into **named, human-interpretable facial muscle events**. When the LLM later says "lip corners are pulled upward (AU12)", this is where that information came from — not a guess.

---

## Step 3 — Emotion Classification with POSTER V2

**File:** `src/emotion/model.py`

### The architecture

POSTER V2 is a **two-branch, cross-attention model** specifically designed for Facial Expression Recognition. Here's what happens inside it:

**Branch 1 — Image features (IR-50 backbone):**
- The face crop (224×224 RGB) is fed into a custom **IR-50** (Improved Residual 50-layer) backbone
- IR-50 is derived from face recognition networks — it's better than standard ResNet-50 at understanding facial structure
- It produces two feature maps at different scales:
  - `feat_s3` from layer 3 → richer spatial detail (256 channels)
  - `feat_s4` from layer 4 → higher-level semantics (512 channels)
- Both are projected to 512-dim and flattened into **image tokens** (sequences of spatial patches)

**Branch 2 — Landmark queries (our extension):**
- The 468 landmarks (x, y coordinates only — shape `468 × 2`) are flattened to a vector of 936 numbers
- This goes through a dense projection layer:
  ```
  Linear(936 → 512) → ReLU → Dropout → Linear(512 → 49 × 512)
  ```
- Output: **49 landmark query tokens** (each a 512-dim vector), one per facial region
- If no landmarks are available, it falls back to **learned fallback parameters** (49 trainable vectors)

**Cross-Attention (the bridge between branches):**
- The 49 landmark queries ask *"what's happening at each facial region in the image?"*
- **Window Cross-Attention** lets each query token attend to the image tokens
- This is repeated twice (depth=2), refining the representation each time
- Result: 49 context-enriched vectors, averaged → 1 final 512-dim feature vector

**Classification head:**
```
Dropout(0.5) → Linear(512 → 7)  → softmax → 7 class probabilities
```

The 7 classes are: `angry, disgust, fear, happy, sad, surprise, neutral`

### Why this design matters for XAI

Because the classification is guided by **real geometric landmark positions** (not random learned tokens), the model's attention is already anchored to meaningful facial anatomy. This makes the Grad-ECLIP step (next) more interpretable.

### What gets output

```python
probabilities = [0.02, 0.01, 0.03, 0.87, 0.04, 0.02, 0.01]
                 angry  disg  fear  happy  sad  surp  neut

top prediction: "happy" with 87.0% confidence
top-3: [("happy", 0.87), ("sad", 0.04), ("fear", 0.03)]
```

---

## Step 4 — Attention Map with Grad-ECLIP

**File:** `src/attention/grad_eclip.py`

### The core question

The model predicted "happy" — but **which pixels in the face image caused that prediction?** This is the classic XAI question.

Standard methods like Grad-CAM answer it, but they produce blurry, low-quality maps — especially on transformer-based models like POSTER V2. Grad-ECLIP is a better approach adapted from Zhao et al. (ICML 2024).

### How Grad-ECLIP works

Grad-ECLIP uses **hooks** — listeners attached to the target layer (the last IR-50 block, `backbone.layer4[-1]`). These hooks capture:

1. **Forward activations** — the feature map produced by that layer during the forward pass
   - Shape: `(1, 512, H_feat, W_feat)` — 512 feature channels at spatial resolution
   
2. **Backward gradients** — how much each activation contributed to the final "happy" score
   - Shape: `(1, 512, H_feat, W_feat)` — computed by calling `.backward()` on the "happy" logit

### The Grad-ECLIP formula

```python
# Channel importance: which feature channels matter for "happy"?
channel_weights = gradients.mean(dim=[2, 3])     # → (1, 512, 1, 1)

# Spatial importance: where in the feature map are strong gradients?
spatial_weights = gradients.abs().mean(dim=1)    # → (1, 1, H, W)

# Combined: weight each activation by both channel AND spatial importance
cam = (channel_weights * activations * spatial_weights).sum(dim=1)

# ReLU: keep only positive contributions (negative = suppresses "happy")
cam = relu(cam)

# Normalize to [0, 1]
cam = cam / cam.max()
```

**What makes Grad-ECLIP different from Grad-CAM:**
- Grad-CAM only uses channel importance: `weights = gradients.mean(spatial dims)`, then `sum(weights * activations)`
- Grad-ECLIP adds **spatial importance** (`gradients.abs().mean(channels)`), which highlights *where* in the image the gradients are strongest
- The combined weighting produces sharper, more precise heatmaps — especially on hybrid CNN-Attention models like POSTER V2

### What gets output

A **2D numpy array** (shape: `H_feat × W_feat`, typically `7 × 7` from layer 4) where each value ∈ [0, 1] represents how much that spatial region contributed to the prediction.

This gets upsampled to the full image size for visualization.

> **XAI role:** This is the "visual proof" — the heatmap shows that the model was looking at the mouth and cheeks, not the forehead, when it predicted "happy". Without this, the classification would be a black box.

---

## Step 5 — Semantic Region Parsing

**File:** `src/attention/region_parser.py`

### The problem

The attention heatmap is just numbers — a 7×7 grid. To be useful in an explanation, we need to convert it into **words** ("the model focused on the mouth and eye regions").

### How it works

The face image is divided into **11 semantic regions** based on approximate pixel proportions:
```
forehead, left_eyebrow, right_eyebrow,
left_eye, right_eye, nose,
left_cheek, right_cheek,
mouth, chin, jaw
```

For each region:
1. A bounding box is defined as a fraction of the face crop (e.g., "mouth = y: 65–85%, x: 25–75%")
2. The average heatmap intensity within that bounding box is computed
3. If intensity > 0.30 (top 30% threshold), the region is marked as **active**
4. Regions are sorted by intensity → ranked by importance

### What gets output

```
attention_regions: ["mouth", "left_eye", "right_eye", "left_cheek"]
attention_summary: "Primary focus: mouth (0.82), left eye (0.61), right eye (0.58), left cheek (0.44)"
```

> **XAI role:** This converts raw pixel heat into named anatomical regions — the bridge between the neural network's internal representation and human language.

---

## Step 6 — VRAM Swap

**File:** `src/pipeline.py` — `_unload_classifier()` → `_load_vlm()`

This is a practical engineering step, not a scientific one.

The classifier (POSTER V2) uses ~2.5 GB VRAM. The LLM (Qwen) needs ~1.5 GB. Together they'd exceed 6 GB. So the pipeline:

1. Calls `del self.classifier` and `del self.attention_gen`
2. `gc.collect()` + `torch.cuda.empty_cache()` — forces Python garbage collector and CUDA to release the memory
3. Then loads the Qwen model fresh

No information is lost — all predictions, AU features, and attention regions are already stored in the `PredictionResult` object in CPU RAM.

---

## Step 7 — Prompt Construction

**File:** `src/explainer/prompt_builder.py`

### The key insight

The LLM doesn't see the image. It only sees **text evidence** assembled from all the previous steps. This is intentional — it forces the explanation to be grounded in measured observations, not hallucinated from the image.

### What goes into the prompt

The `PromptBuilder` assembles five pieces of structured evidence:

```
1. Predicted emotion + confidence
   → "happy (87.0%)"

2. Active Action Units (from Step 2)
   → "AU6 (Cheek Raiser): active [0.027]
      AU12 (Lip Corner Puller): active [0.031]
      AU25 (Lips Part): active [0.022]"

3. Model attention regions (from Step 5)
   → "Primary focus: mouth (0.82), left eye (0.61), right eye (0.58)"

4. Top alternative predictions
   → "sad: 4.0%, fear: 3.0%"

5. FACS reference (hardcoded from literature)
   → "Expected features for happy: lip corners pulled up, cheeks raised,
      crow's feet wrinkles around eyes"
```

The FACS reference is used for **cross-validation** — if the detected AUs match the expected FACS pattern, the LLM can call it out as confirming evidence. If they don't match, it notes the discrepancy.

### The final prompt structure

```
You are an expert in facial expression analysis...

## Evidence

Predicted Emotion: happy (87.0%)

Facial Action Units Detected:
  AU6 (Cheek Raiser): active [0.027]
  AU12 (Lip Corner Puller): active [0.031]
  ...

Model Attention Regions:
  Primary focus: mouth (0.82), left eye (0.61)...

Top Alternative Predictions:
  sad: 4.0%, fear: 3.0%

Expected FACS Features for happy:
  - lip corners pulled up (smile)
  - cheeks raised
  - crow's feet wrinkles

## Task

Explain WHY this face shows "happy". Reference specific AUs, 
explain how they match FACS, note attention alignment, note 
any ambiguity. Under 100 words. Clinical, precise, evidence-based.
```

> **XAI role:** This prompt is the entire XAI pipeline condensed into text. Every claim the LLM can make is backed by a measured number from Step 2, 3, or 4. The LLM can only *synthesize* — it cannot invent.

---

## Step 8 — Explanation Generation with Qwen2.5-0.5B

**File:** `src/explainer/vlm_engine.py`

### The model

**Qwen/Qwen2.5-0.5B-Instruct** is a 0.5 billion parameter instruction-tuned language model from Alibaba. Despite being tiny (compared to 7B+ models), it:
- Follows instruction prompts reliably
- Fits in ~1.5 GB VRAM at fp16
- Generates coherent, structured text in ~2–5 seconds

### How it generates

The prompt (from Step 7) is formatted as a **chat message**:
```python
messages = [
    {"role": "system", "content": "You are an expert XAI assistant..."},
    {"role": "user",   "content": <the full evidence prompt>}
]
```

`apply_chat_template()` converts this to the model's special token format (Qwen uses `<|im_start|>` / `<|im_end|>` markers).

Generation settings:
- `do_sample=False` → **greedy decoding** (always picks the most likely next token)
- `max_new_tokens=150` → hard cap on output length
- `temperature=0.3` → not used with greedy, kept for reference
- `use_cache=True` → KV-cache for faster generation

### What gets output

```
"The model predicted happiness with high confidence (87%), 
supported by the activation of AU12 (Lip Corner Puller) and 
AU6 (Cheek Raiser), which are the primary FACS indicators of 
a genuine Duchenne smile. The model's attention was concentrated 
on the mouth and eye regions, consistent with these AU activations. 
The high confidence and absence of closely competing alternatives 
suggest a clear, unambiguous expression."
```

Note: any markdown artifacts (` ```text ` etc.) from the model are stripped automatically.

---

## Step 9 — Visualization

**File:** `src/visualization.py`

Three images are generated and saved to `outputs/<image_name>/`:

| Output file | What it shows |
|---|---|
| `face_crop.jpg` | The detected face region cropped from the original image |
| `xai_panel.png` | A combined 3-panel figure: original face + landmark overlay + Grad-ECLIP heatmap |
| `result.json` | All structured data: emotion, confidence, AUs, attention, explanation, timing |

The **XAI panel** is the key deliverable — it lets a viewer see the raw image, the structural skeleton (landmarks), and the model's visual attention in one glance.

---

## Full Data Flow Summary

```
Raw Image
   │
   ▼
[MediaPipe FaceLandmarker]
   │   468 landmarks (x, y, z)
   │   face crop (PIL)
   │
   ├──────────────────────────────────────►
   │                                       │
   ▼                                       ▼
[AU Extractor]                      [POSTER V2 Classifier]
 geometric distances                 IR-50 visual features
 → active AUs                        + landmark query tokens
 → AU descriptions (text)            → emotion logits
                                      → probabilities
                                      │
                                      ▼
                                  [Grad-ECLIP]
                                   backward pass
                                   channel × spatial weights
                                   → attention heatmap (2D)
                                      │
                                      ▼
                                  [RegionParser]
                                   heatmap intensity per region
                                   → active region names (text)
   │
   └──────────────────────────────────────────────►
                                                   │
                                                   ▼
                                         [PromptBuilder]
                                          emotion + AUs
                                          + regions + FACS
                                          → structured prompt (text)
                                                   │
                                         [VRAM swap here]
                                                   │
                                                   ▼
                                         [Qwen2.5-0.5B]
                                          greedy decode
                                          → explanation (text)
                                                   │
                                                   ▼
                                         [Visualization]
                                          XAI panel PNG
                                          result.json
```

---

## Why This Is Genuinely XAI (Not Just AI)

| Property | What we do |
|---|---|
| **Faithfulness** | The attention map is computed directly from the model's gradients — it reflects what the model actually used |
| **Groundedness** | Every AU mentioned in the explanation has a measured numeric value from Step 2; no hallucination possible |
| **Traceability** | The result JSON records *every* intermediate value — AUs, attention regions, confidence, alternatives |
| **No human loop** | The system generates, evaluates, and explains its own prediction end-to-end |
| **Contrastive** | The prompt includes alternatives (sad: 4%) so the LLM can explain *why happy and not sad* |
