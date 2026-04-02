import os
import sys
import torch
import torch.nn as nn
import open_clip
from PIL import Image
import argparse
import json

# ── Model (must match train_classifier.py) ────────────────────────
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, num_classes, dropout=0.3):
        super().__init__()
        self.clip       = clip_model
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


# ── Load checkpoint ───────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    clip_name = ckpt.get("clip_model")    # ,  "ViT-B-32"
    embed_dim = ckpt.get("embed_dim")   # ,   512
    n_classes = ckpt.get("num_classes")    # , 4

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_name, pretrained="openai"
    )
    model = CLIPClassifier(clip_model, embed_dim, n_classes).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    id_to_label = ckpt.get("id_to_label", {
        0: "glioma", 1: "meningioma", 2: "pituitary tumor", 3: "no tumor"
    })
    # keys may be strings if saved from JSON-round-tripped dict
    id_to_label = {int(k): v for k, v in id_to_label.items()}

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_acc={ckpt.get('val_acc', 0):.4f})")
    return model, preprocess, id_to_label


# ── Single image inference ────────────────────────────────────────
@torch.no_grad()
def predict(model, preprocess, img_path, id_to_label, device, topk=4):
    img    = Image.open(img_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=-1)[0]

    topk_probs, topk_ids = probs.topk(min(topk, len(id_to_label)))
    results = [
        {"label": id_to_label[idx.item()], "prob": round(prob.item(), 4)}
        for idx, prob in zip(topk_ids, topk_probs)
    ]
    return results


# ── Batch inference from a text file of paths ─────────────────────
@torch.no_grad()
def predict_batch(model, preprocess, image_paths, id_to_label, device, batch_size=32):
    import torch
    from tqdm import tqdm

    all_results = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inferring"):
        batch_paths = image_paths[i: i + batch_size]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                tensors.append(preprocess(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"  ⚠ Skipping {p}: {e}")

        if not tensors:
            continue

        batch  = torch.stack(tensors).to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)

        for path, pred_id, prob_row in zip(valid_paths, preds, probs):
            all_results.append({
                "filepath":   path,
                "prediction": id_to_label[pred_id.item()],
                "confidence": round(prob_row[pred_id].item(), 4),
                "all_probs":  {id_to_label[j]: round(prob_row[j].item(), 4)
                               for j in range(len(id_to_label))},
            })

    return all_results


# ── CLI ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Brain tumor CLIP classifier inference")
    p.add_argument("--checkpoint", required=True,
                   help="Path to classifier_best.pt")
    p.add_argument("--image",      default=None,
                   help="Single image path")
    p.add_argument("--image-list", default=None,
                   help="Text file with one image path per line")
    p.add_argument("--output-csv", default=None,
                   help="Save batch results to this CSV path")
    p.add_argument("--topk",  type=int, default=4,
                   help="Top-k predictions to show (single image mode)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    model, preprocess, id_to_label = load_model(args.checkpoint, device)

    # ── Single image ──────────────────────────────────────────────
    if args.image:
        results = predict(model, preprocess, args.image,
                          id_to_label, device, topk=args.topk)
        print(f"\nImage : {args.image}")
        print(f"{'─'*40}")
        for r in results:
            bar = "█" * int(r['prob'] * 30)
            print(f"  {r['label']:20s} {r['prob']:.4f}  {bar}")
        print(f"\nPrediction: {results[0]['label']} ({results[0]['prob']*100:.1f}%)")

    # ── Batch from list file ──────────────────────────────────────
    elif args.image_list:
        with open(args.image_list) as f:
            paths = [line.strip() for line in f if line.strip()]
        print(f"Running batch inference on {len(paths)} images...")
        results = predict_batch(model, preprocess, paths, id_to_label, device)

        for r in results:
            print(f"{r['filepath']:60s} → {r['prediction']:20s} ({r['confidence']:.4f})")

        if args.output_csv:
            import pandas as pd
            rows = [{
                "filepath":   r["filepath"],
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                **{f"prob_{k}": v for k, v in r["all_probs"].items()}
            } for r in results]
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
            print(f"\n✓ Results saved to {args.output_csv}")

    else:
        print("Provide --image or --image-list")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
# make a text file with one path per line
python inference_classifier.py \
    --checkpoint classifier_bio_med_clip_checkpoints/classifier_best.pt \
    --image  /mnt/d/iit_h_internship/pitu.png\
    --output-csv results.csv
"""