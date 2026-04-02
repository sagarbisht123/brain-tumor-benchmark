import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3
from PIL import Image
import argparse
import pandas as pd

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Model (must match train_efficientnet.py) ──────────────────────
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ── Load checkpoint ───────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt        = torch.load(ckpt_path, map_location=device)
    n_classes   = ckpt.get("num_classes", 4)
    img_size    = ckpt.get("img_size",    300)
    id_to_label = {int(k): v for k, v in ckpt["id_to_label"].items()}

    model = EfficientNetClassifier(n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint: epoch {ckpt.get('epoch','?')} "
          f"(val_acc={ckpt.get('val_acc', 0):.4f})")
    return model, img_size, id_to_label


# ── Inference ─────────────────────────────────────────────────────
def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def predict_single(model, transform, img_path, id_to_label, device, topk=4):
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=-1)[0]

    topk_probs, topk_ids = probs.topk(min(topk, len(id_to_label)))
    return [
        {"label": id_to_label[i.item()], "prob": round(p.item(), 4)}
        for i, p in zip(topk_ids, topk_probs)
    ]


@torch.no_grad()
def predict_batch(model, transform, image_paths, id_to_label, device, batch_size=32):
    from tqdm import tqdm
    results = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inferring"):
        batch_paths = image_paths[i: i + batch_size]
        tensors, valid = [], []
        for p in batch_paths:
            try:
                tensors.append(transform(Image.open(p).convert('RGB')))
                valid.append(p)
            except Exception as e:
                print(f"  ⚠ Skipping {p}: {e}")

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
                "all_probs":  {id_to_label[j]: round(prob_row[j].item(), 4)
                               for j in range(len(id_to_label))},
            })

    return results


# ── CLI ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="EfficientNet-B3 brain tumor inference")
    p.add_argument("--checkpoint", required=True,
                   help="Path to efficientnet_best.pt")
    p.add_argument("--image",      default=None,
                   help="Single image path")
    p.add_argument("--image-list", default=None,
                   help="Text file with one image path per line")
    p.add_argument("--output-csv", default=None,
                   help="Save batch results to CSV")
    p.add_argument("--topk",  type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args      = parse_args()
    device    = torch.device(args.device)
    print(f"Device: {device}")

    model, img_size, id_to_label = load_model(args.checkpoint, device)
    transform = get_transform(img_size)

    if args.image:
        results = predict_single(model, transform, args.image,
                                 id_to_label, device, args.topk)
        print(f"\nImage : {args.image}")
        print(f"{'─'*45}")
        for r in results:
            bar = "█" * int(r['prob'] * 30)
            print(f"  {r['label']:20s}  {r['prob']:.4f}  {bar}")
        print(f"\nPrediction: {results[0]['label']} ({results[0]['prob']*100:.1f}%)")

    elif args.image_list:
        with open(args.image_list) as f:
            paths = [l.strip() for l in f if l.strip()]
        print(f"Batch inference on {len(paths)} images...")
        results = predict_batch(model, transform, paths, id_to_label, device)

        for r in results:
            print(f"{r['filepath']:60s} → {r['prediction']:20s} ({r['confidence']:.4f})")

        if args.output_csv:
            rows = [{
                "filepath":   r["filepath"],
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                **{f"prob_{k}": v for k, v in r["all_probs"].items()}
            } for r in results]
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
            print(f"\n✓ Saved to {args.output_csv}")

    else:
        print("Provide --image or --image-list")
        sys.exit(1)


if __name__ == "__main__":
    main()