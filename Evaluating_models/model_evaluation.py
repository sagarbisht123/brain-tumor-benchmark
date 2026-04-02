"""
evaluate_all_models.py
======================
Runs inference on three brain-tumor classifiers:
  1. EfficientNet-B3      (your fine-tuned checkpoint)
  2. CLIP ViT-B/32        (your fine-tuned CLIP+linear-head checkpoint)
  3. BiomedCLIP           (hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

Usage
-----
python evaluate_all_models.py \
    --csv          /path/to/labels.csv \
    --effnet-ckpt  /path/to/efficientnet_best.pt \
    --clip-ckpt    /path/to/classifier_best.pt \
    --output-dir   ./eval_results

The labels CSV must have columns: filepath, title
(title holds the ground-truth class name, e.g. "glioma", "meningioma", etc.)

Outputs (all written to --output-dir)
--------------------------------------
  results_effnet.csv         per-image predictions + all class probs
  results_clip.csv
  results_biomedclip.csv
  metrics_all_models.csv     macro/weighted F1, accuracy, per-class precision/recall
  evaluation.log             full timestamped log of the run
"""

import os
import sys
import csv
import json
import logging
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── Vision / model libraries ──────────────────────────────────────────────────
from torchvision import transforms
from torchvision.models import efficientnet_b3
import open_clip


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # file handler — full DEBUG log
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# BiomedCLIP prompt templates  ─ one per class
# We score each image against every template and pick the highest-probability class.
"""BIOMEDCLIP_TEMPLATES = {
    "glioma":          "a brain MRI scan showing glioma",
    "meningioma":      "a brain MRI scan showing meningioma",
    "pituitary tumor": "a brain MRI scan showing pituitary tumor",
    "no tumor":        "a brain MRI scan with no tumor",
    # aliases used by some datasets
    "notumor":         "a brain MRI scan with no tumor",
    "pituitary":       "a brain MRI scan showing pituitary tumor",
}"""


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS  (must match training scripts exactly)
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetClassifier(nn.Module):
    """Matches train_efficientnet.py."""
    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class CLIPClassifier(nn.Module):
    """Matches train_classifier.py (CLIP + linear head)."""
    def __init__(self, clip_model, embed_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            feat = self.clip.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return self.classifier(feat.float())


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_effnet(ckpt_path: str, device: torch.device, logger: logging.Logger):
    logger.info(f"[EfficientNet-B3] Loading checkpoint: {ckpt_path}")
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes   = ckpt.get("num_classes", 4)
    img_size    = ckpt.get("img_size",    300)
    id_to_label = {int(k): v for k, v in ckpt["id_to_label"].items()}

    model = EfficientNetClassifier(n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    logger.info(f"[EfficientNet-B3] epoch={ckpt.get('epoch','?')}  "
                f"val_acc={ckpt.get('val_acc', 0):.4f}  "
                f"classes={list(id_to_label.values())}  img_size={img_size}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, transform, id_to_label


def load_clip(ckpt_path: str, device: torch.device, logger: logging.Logger):
    logger.info(f"[CLIP ViT-B/32] Loading checkpoint: {ckpt_path}")
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    clip_name = ckpt.get("clip_model",  "ViT-B-32")
    embed_dim = ckpt.get("embed_dim",   512)
    n_classes = ckpt.get("num_classes", 4)
    id_to_label = {int(k): v for k, v in ckpt.get("id_to_label", {
        "0": "glioma", "1": "meningioma",
        "2": "pituitary tumor", "3": "no tumor"
    }).items()}

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_name, pretrained="openai"
    )
    model = CLIPClassifier(clip_model, embed_dim, n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    logger.info(f"[CLIP ViT-B/32] epoch={ckpt.get('epoch','?')}  "
                f"val_acc={ckpt.get('val_acc', 0):.4f}  "
                f"backbone={clip_name}  classes={list(id_to_label.values())}")
    return model, preprocess, id_to_label


def load_biomedclip(ckpt_path: str, device: torch.device, logger: logging.Logger):
    logger.info(f"[BiomedCLIP] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    clip_name = ckpt.get(
        "clip_model",
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    embed_dim = ckpt.get("embed_dim", 512)
    n_classes = ckpt.get("num_classes", 4)

    id_to_label = {
        int(k): v for k, v in ckpt.get("id_to_label", {
            "0": "glioma",
            "1": "meningioma",
            "2": "pituitary tumor",
            "3": "no tumor"
        }).items()
    }

  
    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_name)

    model = CLIPClassifier(clip_model, embed_dim, n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    logger.info(
        f"[BiomedCLIP-Finetuned] epoch={ckpt.get('epoch','?')}  "
        f"val_acc={ckpt.get('val_acc', 0):.4f}  "
        f"classes={list(id_to_label.values())}"
    )

    return model, preprocess, id_to_label


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_effnet_inference(model, transform, image_paths, id_to_label,
                         device, batch_size, logger):
    logger.info(f"[EfficientNet-B3] Running inference on {len(image_paths)} images ...")
    results = []
    n_classes = len(id_to_label)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        tensors, valid = [], []
        for p in batch_paths:
            try:
                tensors.append(transform(Image.open(p).convert("RGB")))
                valid.append(p)
            except Exception as e:
                logger.warning(f"  [EfficientNet-B3] Skipping {p}: {e}")

        if not tensors:
            continue

        logits = model(torch.stack(tensors).to(device))
        probs  = torch.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)

        for path, pred_id, prob_row in zip(valid, preds, probs):
            results.append({
                "filepath":   path,
                "prediction": id_to_label[pred_id.item()],
                "confidence": round(prob_row[pred_id].item(), 4),
                **{f"prob_{id_to_label[j]}": round(prob_row[j].item(), 4)
                   for j in range(n_classes)},
            })

        if (i // batch_size) % 10 == 0:
            logger.debug(f"  [EfficientNet-B3] processed {min(i+batch_size, len(image_paths))}/{len(image_paths)}")

    logger.info(f"[EfficientNet-B3] Inference complete. {len(results)} results.")
    return results


@torch.no_grad()
def run_clip_inference(model, preprocess, image_paths, id_to_label,
                       device, batch_size, logger):
    logger.info(f"[CLIP ViT-B/32] Running inference on {len(image_paths)} images ...")
    results = []
    n_classes = len(id_to_label)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        tensors, valid = [], []
        for p in batch_paths:
            try:
                tensors.append(preprocess(Image.open(p).convert("RGB")))
                valid.append(p)
            except Exception as e:
                logger.warning(f"  [CLIP] Skipping {p}: {e}")

        if not tensors:
            continue

        logits = model(torch.stack(tensors).to(device))
        probs  = torch.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)

        for path, pred_id, prob_row in zip(valid, preds, probs):
            results.append({
                "filepath":   path,
                "prediction": id_to_label[pred_id.item()],
                "confidence": round(prob_row[pred_id].item(), 4),
                **{f"prob_{id_to_label[j]}": round(prob_row[j].item(), 4)
                   for j in range(n_classes)},
            })

        if (i // batch_size) % 10 == 0:
            logger.debug(f"  [CLIP] processed {min(i+batch_size, len(image_paths))}/{len(image_paths)}")

    logger.info(f"[CLIP ViT-B/32] Inference complete. {len(results)} results.")
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list[str], ground_truth: list[str],
                    model_name: str, logger: logging.Logger) -> dict:
    """
    Computes and logs comprehensive classification metrics.
    Returns a flat dict suitable for a summary CSV row.
    """
    classes = sorted(set(ground_truth))

    acc      = accuracy_score(ground_truth, predictions)
    macro_f1 = f1_score(ground_truth, predictions, average="macro",    zero_division=0)
    wt_f1    = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
    macro_p  = precision_score(ground_truth, predictions, average="macro",    zero_division=0)
    macro_r  = recall_score(ground_truth, predictions,    average="macro",    zero_division=0)

    cm = confusion_matrix(ground_truth, predictions, labels=classes)
    report = classification_report(ground_truth, predictions, labels=classes,
                                   zero_division=0, digits=4)

    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} — METRICS")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy        : {acc:.4f}")
    logger.info(f"  Macro F1        : {macro_f1:.4f}   ← primary metric")
    logger.info(f"  Weighted F1     : {wt_f1:.4f}")
    logger.info(f"  Macro Precision : {macro_p:.4f}")
    logger.info(f"  Macro Recall    : {macro_r:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix (rows=true, cols=pred):")
    logger.info(f"  Classes: {classes}")
    for row_cls, row in zip(classes, cm):
        logger.info(f"  {row_cls:20s}: {row.tolist()}")
    logger.info(f"{'='*60}\n")

    # Per-class metrics
    per_class_p = precision_score(ground_truth, predictions, labels=classes,
                                  average=None, zero_division=0)
    per_class_r = recall_score(ground_truth, predictions, labels=classes,
                               average=None, zero_division=0)
    per_class_f = f1_score(ground_truth, predictions, labels=classes,
                           average=None, zero_division=0)

    row = {
        "model":            model_name,
        "accuracy":         round(acc,      4),
        "macro_f1":         round(macro_f1, 4),
        "weighted_f1":      round(wt_f1,    4),
        "macro_precision":  round(macro_p,  4),
        "macro_recall":     round(macro_r,  4),
        "n_samples":        len(ground_truth),
    }
    for cls, p, r, f in zip(classes, per_class_p, per_class_r, per_class_f):
        safe = cls.replace(" ", "_")
        row[f"precision_{safe}"] = round(p, 4)
        row[f"recall_{safe}"]    = round(r, 4)
        row[f"f1_{safe}"]        = round(f, 4)

    return row


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate EfficientNet-B3, CLIP ViT-B/32, and BiomedCLIP "
                    "on a brain tumor dataset."
    )
    p.add_argument("--csv",        required=True,
                   help="Labels CSV with columns: filepath, title")
    p.add_argument("--effnet-ckpt", required=True,
                   help="Path to efficientnet_best.pt")
    p.add_argument("--clip-ckpt",   required=True,
                   help="Path to classifier_best.pt  (CLIP + linear head)")
    p.add_argument("--biomedclip-ckpt", required=True,
               help="Path to BiomedCLIP checkpoint")
    p.add_argument("--output-dir",  default="./eval_results",
                   help="Directory to save all outputs (default: ./eval_results)")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-effnet",      action="store_true", help="Skip EfficientNet-B3")
    p.add_argument("--skip-clip",        action="store_true", help="Skip CLIP ViT-B/32")
    p.add_argument("--skip-biomedclip",  action="store_true", help="Skip BiomedCLIP")
    return p.parse_args()
"""
python model_evaluation.py \
    --csv /mnt/d/iit_h_internship/open_clip/merged_val.csv \
    --effnet-ckpt /mnt/d/iit_h_internship/open_clip/Efficient_NET_b3_classifier/efficientnet_checkpoints/efficientnet_best.pt \
    --clip-ckpt  /mnt/d/iit_h_internship/open_clip/CLIP_classifier/classifier_ViT-B32_checkpoints/classifier_best.pt \
    --biomedclip-ckpt /mnt/d/iit_h_internship/open_clip/CLIP_classifier/classifier_bio_med_clip_checkpoints/classifier_best.pt \
"""

def main():
    args = parse_args()

    # ── Output directory ──────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = out_dir / f"evaluation_{timestamp}.log"
    logger    = setup_logger(str(log_path))

    logger.info("=" * 60)
    logger.info("  BRAIN TUMOR MODEL EVALUATION")
    logger.info(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Device  : {args.device}")
    logger.info(f"  CSV     : {args.csv}")
    logger.info(f"  Output  : {out_dir}")
    logger.info("=" * 60)

    device = torch.device(args.device)

    # ── Load dataset CSV ──────────────────────────────────────────
    logger.info(f"Loading dataset from: {args.csv}")
    df = pd.read_csv(args.csv)

    # Normalise column names — accept 'title', 'label', 'class', 'ground_truth'
    col_map = {}
    for col in df.columns:
        if col.lower() in ("title", "label", "class", "ground_truth", "tumor_type"):
            col_map[col] = "label"
        elif col.lower() in ("filepath", "path", "image_path", "filename"):
            col_map[col] = "filepath"
    df.rename(columns=col_map, inplace=True)

    if "filepath" not in df.columns or "label" not in df.columns:
        logger.error(f"CSV must have filepath and label columns. Found: {list(df.columns)}")
        sys.exit(1)

    # Drop rows with missing values
    before = len(df)
    df.dropna(subset=["filepath", "label"], inplace=True)
    df["label"] = df["label"].str.strip().str.lower()
    logger.info(f"Dataset: {len(df)} valid rows (dropped {before - len(df)} with NaN)")

    # Show class distribution
    dist = df["label"].value_counts().to_dict()
    logger.info(f"Class distribution: {dist}")

    image_paths = df["filepath"].tolist()
    ground_truth_raw = df["label"].tolist()

    all_classes = sorted(df["label"].unique().tolist())
    # Add this after you build all_classes (around line where logger.info(f"All classes found..."))
    LABEL_NORM = {"pituitary tumor": "pituitary"}
    logger.info(f"All classes found: {all_classes}")

    # ── Collect all metrics rows ──────────────────────────────────
    summary_rows = []

    # ─────────────────────────────────────────────────────────────
    #  1. EfficientNet-B3
    # ─────────────────────────────────────────────────────────────
    if not args.skip_effnet:
        try:
            model_e, transform_e, id2l_e = load_effnet(args.effnet_ckpt, device, logger)
            results_e = run_effnet_inference(
                model_e, transform_e, image_paths, id2l_e,
                device, args.batch_size, logger
            )

            # Align predictions with ground truth using filepath as key
            pred_map_e = {r["filepath"]: LABEL_NORM.get(r["prediction"].lower().strip(), r["prediction"].lower().strip())
                         for r in results_e}
            preds_e = [pred_map_e.get(p, "MISSING") for p in image_paths]

            # Save per-image results
            df_e = pd.DataFrame(results_e)
            df_e["ground_truth"] = df_e["filepath"].map(
                dict(zip(image_paths, ground_truth_raw))
            )
            df_e["correct"] = df_e["prediction"].str.lower().str.strip() == df_e["ground_truth"]
            out_csv_e = out_dir / "results_effnet.csv"
            df_e.to_csv(out_csv_e, index=False)
            logger.info(f"[EfficientNet-B3] Per-image results saved → {out_csv_e}")

            metrics_e = compute_metrics(preds_e, ground_truth_raw, "EfficientNet-B3", logger)
            summary_rows.append(metrics_e)

            # Free GPU memory
            del model_e
            torch.cuda.empty_cache()

        except Exception as exc:
            logger.error(f"[EfficientNet-B3] FAILED: {exc}", exc_info=True)

    # ─────────────────────────────────────────────────────────────
    #  2. CLIP ViT-B/32 (fine-tuned)
    # ─────────────────────────────────────────────────────────────
    if not args.skip_clip:
        try:
            model_c, preproc_c, id2l_c = load_clip(args.clip_ckpt, device, logger)
            results_c = run_clip_inference(
                model_c, preproc_c, image_paths, id2l_c,
                device, args.batch_size, logger
            )

            pred_map_c = {r["filepath"]: LABEL_NORM.get(r["prediction"].lower().strip(), r["prediction"].lower().strip())
              for r in results_c}
            preds_c = [pred_map_c.get(p, "MISSING") for p in image_paths]

            df_c = pd.DataFrame(results_c)
            df_c["ground_truth"] = df_c["filepath"].map(
                dict(zip(image_paths, ground_truth_raw))
            )
            df_c["correct"] = df_c["prediction"].str.lower().str.strip() == df_c["ground_truth"]
            out_csv_c = out_dir / "results_clip_vitb32.csv"
            df_c.to_csv(out_csv_c, index=False)
            logger.info(f"[CLIP ViT-B/32] Per-image results saved → {out_csv_c}")

            metrics_c = compute_metrics(preds_c, ground_truth_raw, "CLIP-ViT-B32-finetuned", logger)
            summary_rows.append(metrics_c)

            del model_c
            torch.cuda.empty_cache()

        except Exception as exc:
            logger.error(f"[CLIP ViT-B/32] FAILED: {exc}", exc_info=True)


# ─────────────────────────────────────────────────────────────
#  3. BiomedCLIP (fine-tuned classifier)
# ─────────────────────────────────────────────────────────────
    if not args.skip_biomedclip:
        try:
            model_b, preproc_b, id2l_b = load_biomedclip(
                args.biomedclip_ckpt, device, logger
            )

            results_b = run_clip_inference(
                model_b, preproc_b, image_paths, id2l_b,
                device, args.batch_size, logger
            )

            pred_map_b = {r["filepath"]: LABEL_NORM.get(r["prediction"].lower().strip(), r["prediction"].lower().strip())
              for r in results_b}
            preds_b = [pred_map_b.get(p, "MISSING") for p in image_paths]

            df_b = pd.DataFrame(results_b)
            df_b["ground_truth"] = df_b["filepath"].map(
                dict(zip(image_paths, ground_truth_raw))
            )
            df_b["correct"] = df_b["prediction"].str.lower().str.strip() == df_b["ground_truth"]

            out_csv_b = out_dir / "results_biomedclip_finetuned.csv"
            df_b.to_csv(out_csv_b, index=False)
            logger.info(f"[BiomedCLIP-Finetuned] Per-image results saved → {out_csv_b}")

            metrics_b = compute_metrics(preds_b, ground_truth_raw, "BiomedCLIP-Finetuned", logger)
            summary_rows.append(metrics_b)

            del model_b
            torch.cuda.empty_cache()

        except Exception as exc:
            logger.error(f"[BiomedCLIP] FAILED: {exc}", exc_info=True)
    # ─────────────────────────────────────────────────────────────
    #  SUMMARY CSV
    # ─────────────────────────────────────────────────────────────
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)

        # Sort by macro_f1 descending — the primary metric
        df_summary.sort_values("macro_f1", ascending=False, inplace=True)

        out_summary = out_dir / "metrics_all_models.csv"
        df_summary.to_csv(out_summary, index=False)
        logger.info(f"\nSummary metrics saved → {out_summary}")

        # Pretty-print the summary table
        logger.info("\n" + "=" * 60)
        logger.info("  FINAL COMPARISON (sorted by Macro F1)")
        logger.info("=" * 60)
        cols_show = ["model", "accuracy", "macro_f1", "weighted_f1",
                     "macro_precision", "macro_recall"]
        available = [c for c in cols_show if c in df_summary.columns]
        logger.info("\n" + df_summary[available].to_string(index=False))
        logger.info("=" * 60)

        # Announce winner
        best = df_summary.iloc[0]["model"]
        best_f1 = df_summary.iloc[0]["macro_f1"]
        logger.info(f"\n  Best model: {best}  (Macro F1 = {best_f1:.4f})")

    else:
        logger.warning("No models produced results — check errors above.")

    logger.info(f"\nAll outputs saved to: {out_dir}")
    logger.info(f"Log file: {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()