# Brain Tumor Classification — Benchmarking CLIP vs EfficientNet

Comparing three deep learning architectures for 4-class brain tumor classification from MRI scans: a fine-tuned **EfficientNet-B3**, a **CLIP ViT-B/32** with a trained MLP classifier head, and **BiomedCLIP-PubMedBERT-256-ViT-B/16** with the same classifier design. The central question is whether a biomedical vision-language model pretrained on PubMed literature can rival or surpass a purpose-built CNN for a medical imaging task.

---

## Architectures

### 1. EfficientNet-B3 (full fine-tune)

```
MRI image → EfficientNet-B3 (ImageNet pretrained)
         → Dropout → Linear(1536, 512) → SiLU → Dropout → Linear(512, 4)
         → 4-class softmax
```

The backbone is loaded with ImageNet weights. For the first 5 epochs the backbone is frozen and only the classification head is trained (head LR = 1e-3). After epoch 5 the entire network is unfrozen with differential learning rates: backbone at 1e-4, head at 1e-3. A cosine decay schedule with a 3-epoch linear warmup is applied throughout. Native resolution is 300×300.

### 2. CLIP ViT-B/32 + MLP classifier

```
MRI image → ViT-B/32 (OpenAI CLIP, frozen)
         → 512-d L2-normalised embedding
         → LayerNorm → Dropout → Linear(512, 256) → GELU → Dropout → Linear(256, 4)
         → 4-class softmax
```

The CLIP image encoder is kept fully frozen. Only the lightweight MLP classifier head is trained, making this extremely parameter-efficient. The hypothesis is that CLIP's general visual representations transfer to MRI without any backbone fine-tuning.

### 3. BiomedCLIP-PubMedBERT-256-ViT-B/16 + MLP classifier

```
MRI image → ViT-B/16 (BiomedCLIP, frozen)
         → 512-d L2-normalised embedding
         → LayerNorm → Dropout → Linear(512, 256) → GELU → Dropout → Linear(256, 4)
         → 4-class softmax
```

Uses Microsoft's [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), a CLIP model pretrained on 15 million biomedical image-text pairs from PubMed. Same frozen-encoder + trainable-head design as above. The key difference from CLIP ViT-B/32 is the domain-specific pretraining, which should provide richer medical visual representations.

All three models share the same training recipe: AdamW optimiser, class-weighted cross-entropy loss, cosine decay with linear warmup, and gradient clipping at 1.0.

---

## Datasets

The training and validation data is merged from two public sources:

**Figshare Brain Tumor Dataset** — MRI scans across three tumor types (glioma, meningioma, pituitary) sourced from the Figshare repository. A subset covering the brain tumor classes was used.

**Kaggle Brain Tumor MRI Dataset** — A widely-used 4-class dataset (glioma, meningioma, pituitary, no tumor) with pre-split Training and Testing folders.

Both are merged into a single pool and re-split:

```
data/
├── merged_train.csv       # 8,211 images
│   ├── glioma        2,581
│   ├── pituitary     2,184
│   ├── meningioma    2,006
│   └── no tumor      1,440
└── merged_val.csv         # 2,053 images
    ├── glioma          645
    ├── pituitary       546
    ├── meningioma      502
    └── no tumor        360
```

Each CSV has two columns: `filepath` (absolute path to the image) and `title` (class label). Update the paths in the config section at the top of each training script to match your local directory layout.

---

## Installation

Python 3.11 was used throughout. First configure your PyTorch installation for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/), then install the rest:

```bash
# Example for CUDA 12.1
pip install torch==2.10.0 torchvision==0.25.0 --extra-index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt`:
```
open_clip_torch==3.2.0
transformers==5.1.0
tokenizers==0.22.2
huggingface_hub==1.4.1
safetensors==0.7.0
sentencepiece==0.2.1
timm==1.0.24
numpy==2.4.2
pandas==3.0.0
scikit-learn==1.8.0
scipy==1.17.0
pillow==12.1.1
h5py==3.16.0
tqdm==4.67.1
tensorboard==2.20.0
kaggle==2.0.0
requests==2.32.5
pyyaml==6.0.3
regex==2026.1.15
ftfy==6.3.1
packaging==26.0
```

---

## Scripts

### `train_classifier.py` — CLIP ViT-B/32 and BiomedCLIP head training

This script is used to train the MLP classification head on top of either CLIP backbone. The backbone (`CLIP_MODEL` config variable) can be switched between `"ViT-B-32"` (standard CLIP) and `"hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"` (BiomedCLIP) without any other changes.

The reason a separate classifier head is trained on top of CLIP rather than fine-tuning the whole encoder is twofold: CLIP encoders are large and prone to overfitting on a domain-specific dataset of this size, and a frozen encoder with a small trainable head converges much faster while still transferring well. The head architecture (LayerNorm → Dropout → Linear → GELU → Dropout → Linear) is lightweight enough to train in under 20 epochs.

Key config variables at the top of the file:

| Variable | Description |
|---|---|
| `CLIP_MODEL` | Backbone — swap between CLIP and BiomedCLIP |
| `FREEZE_CLIP` | Whether to freeze the encoder (recommended: `True`) |
| `EPOCHS` | Training epochs (default 20) |
| `CHECKPOINT_DIR` | Where to save `.pt` checkpoints |

Checkpoints are saved as `classifier_best.pt` and `classifier_latest.pt`, each containing the full model state dict, optimizer state, epoch, val accuracy, and the `id_to_label` mapping needed for inference.

### `train_efficientnet.py` — EfficientNet-B3 fine-tuning

Trains EfficientNet-B3 with a two-phase strategy: the backbone is frozen for the first `FREEZE_EPOCHS` (default 5), training only the new classification head. After that, the full network is unfrozen and trained end-to-end with differential learning rates — the backbone receives a 10× lower LR than the head to avoid destroying the pretrained features.

Key training details:

- Input resolution: 300×300 (EfficientNet-B3's native size)
- Heavy augmentation during training: random crop, horizontal/vertical flip, rotation ±15°, colour jitter, random affine
- Class-weighted loss to handle the imbalance between `no tumor` and the other classes
- Cosine decay schedule restarts after the phase switch at epoch `FREEZE_EPOCHS + 1`

### `evaluate_all_models.py` — unified evaluation

Runs all three trained models against the same validation CSV in a single script and produces a consolidated report. For each model it outputs:

- Per-image prediction CSV with class probabilities
- Full classification report (precision, recall, F1 per class)
- Confusion matrix
- Summary CSV ranked by macro F1

Usage:

```bash
python evaluate_all_models.py \
  --csv            /path/to/merged_val.csv \
  --effnet-ckpt    /path/to/efficientnet_best.pt \
  --clip-ckpt      /path/to/classifier_ViT-B32_best.pt \
  --biomedclip-ckpt /path/to/classifier_biomedclip_best.pt \
  --output-dir     ./eval_results
```

The script normalises label strings at evaluation time so that checkpoint-internal label names (e.g. `"pituitary tumor"`) are matched correctly against the CSV ground truth labels (e.g. `"pituitary"`).

---

## Results

Evaluated on 2,053 validation images across all four classes.

### Overall benchmark

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| **EfficientNet-B3** | **0.9898** | **0.9896** | **0.9898** | 0.9889 | 0.9904 |
| BiomedCLIP-Finetuned | 0.9323 | 0.9353 | 0.9323 | 0.9375 | 0.9338 |
| CLIP-ViT-B32-Finetuned | 0.8685 | 0.8723 | 0.8680 | 0.8698 | 0.8755 |

### Per-class F1

| Class | EfficientNet-B3 | BiomedCLIP | CLIP ViT-B/32 |
|---|---|---|---|
| glioma | 0.9915 | 0.9322 | 0.8665 |
| meningioma | 0.9832 | 0.8945 | 0.7759 |
| no tumor | 0.9903 | 0.9764 | 0.9396 |
| **pituitary** | **0.9935** | 0.9380 | 0.9072 |

Meningioma is the hardest class across all models — it is most frequently confused with glioma, which makes clinical sense as both are parenchymal tumors with overlapping MRI appearance.

### EfficientNet-B3 confusion matrix

|  | pred: glioma | pred: meningioma | pred: no tumor | pred: pituitary |
|---|---|---|---|---|
| **glioma** | 638 | 4 | 3 | 0 |
| **meningioma** | 3 | 496 | 3 | 0 |
| **no tumor** | 1 | 0 | 359 | 0 |
| **pituitary** | 0 | 7 | 0 | 539 |

### Key takeaways

EfficientNet-B3 with full fine-tuning at native resolution (300×300) substantially outperforms both CLIP-based models despite the CLIP encoders being frozen. BiomedCLIP's domain-specific pretraining gives it a clear advantage over general CLIP ViT-B/32 (+6.3 pp macro F1), validating the importance of biomedical pretraining data even when the encoder is not fine-tuned. All three models struggle most with meningioma vs glioma disambiguation.

> **Note on data leakage**: the merged dataset includes images from the Kaggle `Testing/` folder mixed into both training and validation splits (~1,600 images total). The reported numbers are therefore slightly optimistic compared to what would be seen on a fully clean held-out test set.

---

## Checkpoints

After training, checkpoints are saved to the directories specified in each script's config. Each `.pt` file contains:

```python
{
  "epoch":       int,
  "model":       state_dict,
  "optimizer":   state_dict,
  "val_acc":     float,
  "val_loss":    float,
  "id_to_label": {0: "glioma", 1: "meningioma", 2: "pituitary", 3: "no tumor"},
  "num_classes": 4,
  # CLIP checkpoints also include:
  "clip_model":  str,
  "embed_dim":   int,
  # EfficientNet checkpoint also includes:
  "img_size":    300,
}
```

---

## License

Dataset licenses apply to their respective sources (Figshare CC BY 4.0, Kaggle dataset terms). Code in this repository is MIT licensed.
