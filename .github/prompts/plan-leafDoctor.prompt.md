# Plan: Full Hackathon Optimization for Leaf Doctor

## TL;DR
Implement all improvements in parallel-friendly phases: (1) Extract training into reproducible benchmark script with comprehensive metrics, (2) Retrain model with class-balanced augmentation and synthetic data, (3) Integrate Grad-CAM visualization into prediction pipeline, (4) Add robustness stress tests, (5) Polish UI with confidence warnings and clarity. Target: judges see hard numbers + visual explanations + robustness proof.

---

## Steps

### Phase 1: Benchmark Foundation (Days 1-2, parallel with Phase 2)
*Goal: Generate publishable metrics judges can immediately verify and compare.*

1. **Extract training into reproducible script** (`Model/train.py`)
   - Convert `Plant Disease Detection Code.ipynb` logic to standalone script
   - Load dataset, split by disease (train/val/test, consistent seeding)
   - Save dataset split metadata (counts, class distribution) for reproducibility
   - Report: Train/val/test size, class counts, class balance %

2. **Generate comprehensive benchmark metrics** (`Model/benchmark.py`)
   - Calculate per-class: precision, recall, F1, support
   - Macro F1, weighted F1, micro accuracy
   - Confusion matrix (39×39) exported as CSV
   - Per-class breakdown table (39 rows × 5 cols: disease name, precision, recall, F1, support)
   - Top-5 hardest classes (lowest F1)
   - Multi-image vs single-image gain (ensemble accuracy lift vs best single)
   - Latency: CNN inference time, end-to-end /predict time, fallback success %
   - Save all metrics to JSON for repeatability

3. **Create benchmark report** (`Model/BENCHMARK.md`)
   - Table format: | Disease | Precision | Recall | F1 | Support |
   - Summary section with macro/weighted F1 and overall accuracy
   - Confusion matrix heatmap (matplotlib, saved as PNG)
   - Multi-image ensemble gain (e.g., "3-image ensemble: +7.2% accuracy vs single")
   - Latency profile (all components in ms)

### Phase 2: Model Improvement via Augmentation (Days 1-3, parallel with Phase 1)
*Goal: Improve accuracy/F1 with class-targeted augmentation and measure gains.*

1. **Implement augmentation strategy** (`Model/augmentation.py`)
   - Use `torchvision.transforms`: RandomRotation (±15°), ColorJitter (brightness/contrast), RandomAffine (perspective), RandomHorizontalFlip, GaussianBlur
   - Class-balanced augmentation: double augmentation probability for minority classes (<2% of dataset)
   - Synthetic data via mixup (blend two random samples, interpolate labels) for minority classes only
   - Save augmentation pipeline as reusable `create_train_transforms()` function

2. **Retrain model** (`Model/train_v2.py`)
   - Use same CNN architecture (4 conv blocks, 39 classes)
   - Adam optimizer, initial LR 1e-3, ReduceLROnPlateau decay
   - Epochs: train until val F1 plateaus (typically 20-30 epochs)
   - Log: train/val loss, train/val accuracy, train/val F1 per epoch
   - Save best model checkpoint (by val F1) as `plant_disease_model_v2.pt`
   - Save training curves (loss + F1 plots) as PNG

3. **Run ablation study** (`Model/ablation.py`)
   - Baseline: No augmentation (current model or quick retrain)
   - +Augmentation: With transforms only
   - +Synthetic: Augmentation + mixup for minority classes
   - Compare: F1 lift for each minority class, overall F1 gain, per-class precision/recall
   - Report: Table showing before/after per-class F1 for bottom-5 classes, overall macro F1

4. **Update app.py to test both models** (optional, for judge demo)
   - Add `model_version` query param: `?model=v1` (current, baseline) or `?model=v2` (improved)
   - Default to v2 if available
   - Both models return same JSON structure
   - In dev, judges can compare side-by-side results on same images

### Phase 3: Visual Explainability (Days 2-4, depends on trained model)
*Goal: Show judges exactly which image regions the model relied on for each prediction.*

1. **Implement Grad-CAM** (`Leaf-Doctor/grad_cam.py`)
   - Class: `GradCAM(model, target_layer='layer4')` 
   - Input: image tensor + predicted class index
   - Output: heatmap (224×224) showing activation for predicted disease
   - Normalize heatmap to [0,1] range for visual clarity
   - Reference: PyTorch CAM library or torch `register_full_backward_hook`

2. **Integrate into /predict endpoint** (`Leaf-Doctor/app.py:625`)
   - After CNN prediction, if confidence > 0.5 (meaningful enough to explain):
     - Generate Grad-CAM for top-1 prediction
     - Save heatmap PNG to temp directory
     - Include `heatmap_url` in JSON response
   - For multi-image: generate + return heatmap for main ensemble image (averaged logits)
   - Skip Grad-CAM if model_version='v1' (legacy, optional)

3. **Update UI to display heatmaps** (`Leaf-Doctor/templates/base.html`, `chat.js:180`)
   - In prediction card, add new section: "Model Focus Map"
   - Show original leaf image side-by-side with Grad-CAM heatmap overlay
   - Add brief explanation: "Red areas highlight regions the model used for diagnosis"
   - Make heatmap clickable/toggleable if space constrained

4. **Benchmark Grad-CAM impact**
   - No accuracy impact (explanation only)
   - Latency: <50ms per image (should not slow down /predict)
   - Measure in benchmark script

### Phase 4: Robustness Validation (Days 3-5, can run in parallel with Phase 3)
*Goal: Prove the model works reliably in real field conditions (blur, poor light, occlusion).*

1. **Build stress test suite** (`Model/stress_tests.py`)
   - Blur test: `cv2.GaussianBlur(img, ksize=(11,11))` variants
   - Low-light test: multiply pixel values by [0.3, 0.5, 0.7]
   - Occlusion test: random 30-50px patches set to 0
   - Noisy-background test: add Gaussian noise (σ=0.1 normalized images)
   - Each corruption level: 2-3 variants
   - Target: 120-150 corrupted test images total

2. **Run stress test evaluation** (`Model/stress_eval.py`)
   - For each corruption type/level, run ensemble-aggregated predictions on subset
   - Metrics: accuracy, macro F1, per-class F1 deltas vs clean baseline
   - Report: "Model maintains 75% accuracy under 3x blur" e.g.
   - Identify which classes degrade fastest (hardest in the field)

3. **Create robustness report** (`Model/ROBUSTNESS.md`)
   - Table format: | Corruption | Validation Acc | Acc Drop | Macro F1 | Critical Classes |
   - Show accuracy drop-off curve (clean → increasing corruption)
   - Highlight 3-5 hardest cases
   - Recommend field-use conditions: "Best used on in-focus leaf images from green backgrounds"

4. **Latency under load** (optional, if time permits)
   - Send 10 concurrent requests to /predict with 2-image payload
   - Measure response time P50, P95, P99
   - Verify queue doesn't back up (Flask default threading handles, but document)

### Phase 5: UI Polish & Clarity (Days 4-5, parallel with Phase 4)
*Goal: Ensure judges immediately understand what model is running and why they should trust it.*

1. **Add confidence warning policy** (`templates/home.html`, `app.py`)
   - Home hero section: "Uses ensemble of multiple leaf images for higher accuracy"
   - /predict response flow:
     - confidence >= 0.75: "High confidence" (green pill, proceed)
     - 0.5-0.75: "Moderate confidence" + "Consider uploading side/back view" (yellow pill)
     - <0.5: "Low confidence" + "Result may be unreliable, consult expert" (red pill, in quality warnings)
   - Implement in `agent.py:quality_check()` logic (already exists, just refine messaging)

2. **Clarify implementation** (README, no code changes needed)
   - Update root README.md: "Active implementation: Leaf-Doctor/ (with vector search, agent reasoning, Grad-CAM). Reference: old-ui/"
   - State: "Model: 39-class CNN with multi-image ensemble. Training: see Model/train_v2.py"
   - Avoid ambiguity judges may have about which code to look at

3. **Update /predict response schema documentation** (`app.py` docstring)
   - Document all fields: disease, confidence, alternatives, per_image, agentic, reasoning_chain, heatmap_url (new), quality_check
   - Include example JSON with all fields populated
   - Clarify: "Confidence is softmax of ensemble logits across all images"

4. **Add model version to response** (`app.py:655`)
   - Include `"model_version": "v2"` in JSON (or v1 if fallback)
   - Judges can immediately see which model generated the result

---

## Relevant Files

### Training & Metrics
- `Model/train.py` — **Create from notebook** (reproducible training script with metrics)
- `Model/train_v2.py` — **Create** (improved training with augmentation)
- `Model/augmentation.py` — **Create** (class-balanced augmentation pipeline)
- `Model/benchmark.py` — **Create** (comprehensive metrics generation)
- `Model/ablation.py` — **Create** (ablation study: baseline vs augmentation vs synthetic)
- `Model/BENCHMARK.md` — **Create** (publishable metrics report)
- `Model/ROBUSTNESS.md` — **Create** (stress test results)
- `Model/Plant Disease Detection Code.ipynb` — Reference only (source logic to extract)

### Explainability
- `Leaf-Doctor/grad_cam.py` — **Create** (Grad-CAM implementation)
- `Leaf-Doctor/app.py:625` — **Modify** (integrate Grad-CAM into /predict, save heatmap)
- `Leaf-Doctor/app.py` docstring — **Modify** (add heatmap_url to response schema)
- `Leaf-Doctor/templates/base.html:150` — **Modify** (add heatmap display section)
- `Leaf-Doctor/chat.js:180` — **Modify** (render heatmap card in prediction display)

### Robustness
- `Model/stress_tests.py` — **Create** (corruption pipeline: blur, low-light, occlusion, noise)
- `Model/stress_eval.py` — **Create** (evaluate model on corrupted images)

### UI & Clarity
- `templates/home.html:50` — **Modify** (update hero section with ensemble emphasis)
- `Leaf-Doctor/agent.py:quality_check()` — **Enhance** (refine confidence warning messages, already exists)
- `Leaf-Doctor/app.py:655` — **Modify** (add model_version field to response JSON)
- `README.md` — **Modify** (clarify active implementation, model details)

---

## Verification

1. **Benchmark metrics generated & published**
   - Run: `python Model/benchmark.py --model plant_disease_model_v1.pt --dataset [path]`
   - Verify: `Model/benchmark_report.json` and `Model/BENCHMARK.md` generated
   - Check: All 39 classes present in per-class table, confusion matrix 39×39
   - Compare: Multi-image lift ≥ 3% accuracy (realistic for ensemble)

2. **Improved model retraining**
   - Run: `python Model/train_v2.py --augment --epochs 30`
   - Verify: `plant_disease_model_v2.pt` saved
   - Check: New model F1 > old model F1 (target +2-5% macro F1 from augmentation)
   - Ablation report shows per-class gains for minority classes

3. **Grad-CAM integration** 
   - Upload image to `/predict`, verify `heatmap_url` field in response
   - Check: Heatmap PNG generated and visually sensible (red areas on disease region)
   - Latency: /predict endpoint <200ms with Grad-CAM enabled
   - UI: Heatmap displays correctly in chat prediction card

4. **Robustness testing**
   - Run: `python Model/stress_eval.py --model v2 --corruption blur,low-light,occlusion`
   - Verify: Accuracy drop curves generated for each corruption type
   - Check: Report shows classification accuracy at 2-3 severity levels per corruption
   - Highlight: Identify which classes fail first (e.g., "thin-leaf diseases blur-sensitive")

5. **UI warnings functional**
   - Test confidence thresholds: Upload image with high confidence > 0.75 (green pill), low confidence < 0.5 (red pill)
   - Verify: Warning messages display correctly ("Consider multiple images")
   - Check: Heatmap renders without errors

6. **Implementation clarity clarity**
   - Open root README & verify: "Active implementation: Leaf-Doctor/" is stated within first 3 paragraphs
   - Verify: /predict docstring includes full schema with heatmap_url example
   - Model version in response JSON visible in chat (judges can inspect)

---

## Decisions

**Model Strategy:** Keep 39-class CNN architecture, improve via augmentation (faster than retraining from scratch). Publish both baseline (v1) and augmented (v2) metrics for ablation story.

**Explainability Focus:** Grad-CAM only (not SHAP/LIME) — judges prefer simple, visual explanations. CAM is industry-standard for CNN diagnostics.

**Robustness Scope:** Synthetic corruptions on test set (not retraining robust models), focus on reporting realistic accuracy drop under field conditions. Demonstrates practical understanding.

**Timeline Parallelization:** Phase 1 & 2 run simultaneously (training/metrics independent). Phase 3 & 4 start day 2-3 (not blocking). UI polish (Phase 5) concurrent with Phase 4.

**Backward Compatibility:** Keep app.py working with v1 model as fallback. Judges won't penalize if v2 training incomplete, but v1 fallback means no broken demo.

---

## Further Considerations

1. **Dataset availability** — Do you have the training dataset locally? If not, I'll need to reconstruct from the notebook or ask where it's stored. Block: can't train without it.

2. **Hardware for training** — GPU available? V2 retraining on large dataset on CPU may take 1-2 hours. If CPU-only, may need to subsample dataset for speed. Not a blocker but affects timeline.

3. **Heatmap temp storage** — Grad-CAM heatmaps saved to `static/heatmaps/` temp directory or in-memory? If disk, need cleanup strategy to avoid bloat. Usually fine for hackathon demo.
