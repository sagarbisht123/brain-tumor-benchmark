import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.metrics import classification_report, confusion_matrix
import logging
import warnings
warnings.filterwarnings('ignore')

# ----------------------------- CONFIG -----------------------------
TRAIN_CSV       = "/mnt/d/iit_h_internship/open_clip/merged_train.csv"
VAL_CSV         = "/mnt/d/iit_h_internship/open_clip/merged_val.csv"
CLIP_MODEL      = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" # or "ViT-B-32"
CLIP_PRETRAINED = "openai"           # swap to your fine-tuned .pt path if needed
FREEZE_CLIP     = True
EPOCHS          = 20
BATCH_SIZE      = 64
LR              = 1e-3
WEIGHT_DECAY    = 0.01
WARMUP_EPOCHS   = 2
DROPOUT         = 0.3
CHECKPOINT_DIR  = "classifier_bio_med_clip_checkpoints"
LOG_FILE        = "classifier_train.log"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED     = 42
# -----------------------------------------------------------------

torch.manual_seed(RANDOM_SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Labels ────────────────────────────────────────────────────────
LABEL_MAP = {
    "glioma":          0,
    "meningioma":      1,
    "pituitary":       2,
    "pituitary tumor": 2,
    "no tumor":        3,
    "notumor":         3,
}
ID_TO_LABEL = {0: "glioma", 1: "meningioma", 2: "pituitary tumor", 3: "no tumor"}
NUM_CLASSES = 4
LABEL_NAMES = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]


# ── Dataset ───────────────────────────────────────────────────────
class BrainTumorDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)                    # header: filepath, title
        df = df.rename(columns={"title": "label"})
        df['label'] = df['label'].str.strip().str.lower()
        df['label_id'] = df['label'].map(LABEL_MAP)

        unknown = df['label_id'].isna()
        if unknown.sum() > 0:
            log.warning(f"Dropping {unknown.sum()} rows with unrecognised labels")
            log.warning(df[unknown]['label'].value_counts().to_string())
        df = df[~unknown].reset_index(drop=True)
        df['label_id'] = df['label_id'].astype(int)

        self.df        = df
        self.transform = transform
        log.info(f"  {csv_path}: {len(df)} samples")
        log.info(f"  Distribution:\n{df['label'].value_counts().to_string()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['filepath']).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            log.warning(f"Load failed {row['filepath']}: {e}")
            img = torch.zeros(3, 224, 224)
        return img, int(row['label_id'])


# ── Model ─────────────────────────────────────────────────────────
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
        ctx = torch.no_grad() if not self.clip.training else torch.enable_grad()
        with ctx:
            feat = self.clip.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return self.classifier(feat.float())


# ── Scheduler ─────────────────────────────────────────────────────
def make_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train / eval ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.detach().argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Checkpoint ────────────────────────────────────────────────────
def save_ckpt(model, optimizer, epoch, val_acc, val_loss, tag):
    path = os.path.join(CHECKPOINT_DIR, f"classifier_{tag}.pt")
    
    device    = next(model.clip.parameters()).device
    dummy     = torch.zeros(1, 3, 224 ,224).to(device)
    with torch.no_grad():
        embed_dim = model.clip.encode_image(dummy).shape[-1]
    torch.save({
        "epoch":       epoch,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "val_acc":     val_acc,
        "val_loss":    val_loss,
        "label_map":   LABEL_MAP,
        "id_to_label": ID_TO_LABEL,
        "clip_model":  CLIP_MODEL,
        "embed_dim":   embed_dim,
        "num_classes": NUM_CLASSES,
    }, path)
    return path


# ── Main ──────────────────────────────────────────────────────────
def main():
    log.info(f"Device        : {DEVICE}")
    log.info(f"CLIP          : {CLIP_MODEL} / {CLIP_PRETRAINED}")
    log.info(f"Freeze CLIP   : {FREEZE_CLIP}")
    log.info(f"Epochs/BS/LR  : {EPOCHS} / {BATCH_SIZE} / {LR}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )


    dummy     = torch.zeros(1, 3, 224 ,224)
    with torch.no_grad():
        embed_dim = clip_model.encode_image(dummy).shape[-1]

    # embed_dim = clip_model.visual.output_dim

    if FREEZE_CLIP:
        for p in clip_model.visual.parameters():
            p.requires_grad = False
        log.info("CLIP image encoder frozen — training head only")

    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275,  0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])

    log.info("Loading datasets...")
    train_ds = BrainTumorDataset(TRAIN_CSV, train_tfm)
    val_ds   = BrainTumorDataset(VAL_CSV,   preprocess)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = CLIPClassifier(clip_model, embed_dim, NUM_CLASSES, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters() if FREEZE_CLIP else model.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, len(train_loader))

    counts    = train_ds.df['label_id'].value_counts().sort_index().values
    weights   = torch.tensor(1.0 / counts, dtype=torch.float32)
    weights   = weights / weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    best_val_acc = 0.0
    best_epoch   = 0
    history      = []

    log.info(f"\n{'─'*68}")
    log.info(f"{'Epoch':>6} | {'TrLoss':>8} | {'TrAcc':>7} | {'VlLoss':>8} | {'VlAcc':>7} |")
    log.info(f"{'─'*68}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE)
        vl_loss, vl_acc, preds, labels = eval_epoch(
            model, val_loader, criterion, DEVICE)

        log.info(f"{epoch:>6} | {tr_loss:>8.4f} | {tr_acc:>7.4f} | "
                 f"{vl_loss:>8.4f} | {vl_acc:>7.4f} |")

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            vl_loss=vl_loss, vl_acc=vl_acc))

        # always save latest
        save_ckpt(model, optimizer, epoch, vl_acc, vl_loss, tag="latest")

        # save best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            path = save_ckpt(model, optimizer, epoch, vl_acc, vl_loss, tag="best")
            log.info(f"  ★ Best checkpoint saved → {path}")

    pd.DataFrame(history).to_csv(
        os.path.join(CHECKPOINT_DIR, "history.csv"), index=False)

    log.info(f"\n{'─'*68}")
    log.info(f"Done. Best epoch: {best_epoch} | Val acc: {best_val_acc:.4f}")

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "classifier_best.pt"))
    model.load_state_dict(ckpt['model'])
    _, _, preds, labels = eval_epoch(model, val_loader, criterion, DEVICE)

    log.info("\nClassification Report:\n" +
             classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))
    cm    = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    log.info(f"\nConfusion Matrix:\n{cm_df.to_string()}")


if __name__ == "__main__":
    main()