import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import logging
import warnings
warnings.filterwarnings('ignore')

# ----------------------------- CONFIG -----------------------------
TRAIN_CSV      = "/mnt/d/iit_h_internship/open_clip/merged_train.csv"
VAL_CSV        = "/mnt/d/iit_h_internship/open_clip/merged_val.csv"
EPOCHS         = 30
BATCH_SIZE     = 32               # B3 is heavier than ViT-B-32, keep lower
LR             = 1e-4
LR_HEAD        = 1e-3             # classifier head gets higher LR
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 3
DROPOUT        = 0.4              # B3 already has dropout, this is extra
FREEZE_EPOCHS  = 5                # freeze backbone for first N epochs, then unfreeze
IMG_SIZE       = 300              # EfficientNet-B3 native resolution
CHECKPOINT_DIR = "efficientnet_checkpoints"
LOG_FILE       = "efficientnet_train.log"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED    = 42
# -----------------------------------------------------------------

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
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

# ImageNet normalization (EfficientNet was pretrained on ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────────────────
class BrainTumorDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"title": "label"})
        df['label'] = df['label'].str.strip().str.lower()
        df['label_id'] = df['label'].map(LABEL_MAP)

        unknown = df['label_id'].isna()
        if unknown.sum() > 0:
            log.warning(f"Dropping {unknown.sum()} unrecognised label rows")
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
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return img, int(row['label_id'])


# ── Model ─────────────────────────────────────────────────────────
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        # Load pretrained B3
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.SiLU(),                           # same activation as B3 internals
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        log.info("Backbone frozen — training head only")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        log.info("Backbone unfrozen — full fine-tune")


# ── Scheduler: warmup + cosine decay ─────────────────────────────
def make_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_optimizer(model, freeze=True):
    if freeze:
        return torch.optim.AdamW(
            model.backbone.classifier.parameters(),
            lr=LR_HEAD, weight_decay=WEIGHT_DECAY
        )
    else:
        # Differential LR: lower for backbone, higher for head
        backbone_params = [p for n, p in model.backbone.named_parameters()
                           if 'classifier' not in n]
        head_params     = list(model.backbone.classifier.parameters())
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": LR},
            {"params": head_params,     "lr": LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)


# ── Train / eval loops ────────────────────────────────────────────
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
    path = os.path.join(CHECKPOINT_DIR, f"efficientnet_{tag}.pt")
    torch.save({
        "epoch":       epoch,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "val_acc":     val_acc,
        "val_loss":    val_loss,
        "id_to_label": ID_TO_LABEL,
        "num_classes": NUM_CLASSES,
        "img_size":    IMG_SIZE,
    }, path)
    return path


# ── Main ──────────────────────────────────────────────────────────
def main():
    log.info(f"Device        : {DEVICE}")
    log.info(f"Model         : EfficientNet-B3 (ImageNet pretrained)")
    log.info(f"Epochs/BS/LR  : {EPOCHS} / {BATCH_SIZE} / backbone={LR} head={LR_HEAD}")
    log.info(f"Freeze epochs : {FREEZE_EPOCHS} then full fine-tune")
    log.info(f"Image size    : {IMG_SIZE}x{IMG_SIZE}")

    # Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Datasets
    log.info("Loading datasets...")
    train_ds = BrainTumorDataset(TRAIN_CSV, train_tfm)
    val_ds   = BrainTumorDataset(VAL_CSV,   val_tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model
    model = EfficientNetClassifier(NUM_CLASSES, DROPOUT).to(DEVICE)
    model.freeze_backbone()

    # Class-weighted loss
    counts    = train_ds.df['label_id'].value_counts().sort_index().values
    weights   = torch.tensor(1.0 / counts, dtype=torch.float32)
    weights   = weights / weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    optimizer = make_optimizer(model, freeze=True)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, len(train_loader))

    best_val_acc = 0.0
    best_epoch   = 0
    history      = []

    log.info(f"\n{'─'*70}")
    log.info(f"{'Epoch':>6} | {'Phase':>8} | {'TrLoss':>8} | {'TrAcc':>7} | {'VlLoss':>8} | {'VlAcc':>7}")
    log.info(f"{'─'*70}")

    for epoch in range(1, EPOCHS + 1):

        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            model.unfreeze_backbone()
            optimizer = make_optimizer(model, freeze=False)
            scheduler = make_scheduler(optimizer, 0, EPOCHS - FREEZE_EPOCHS,
                                       len(train_loader))
            log.info(f"  Epoch {epoch}: switching to full fine-tune with differential LR")

        phase = "frozen" if epoch <= FREEZE_EPOCHS else "full"

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE)
        vl_loss, vl_acc, preds, labels = eval_epoch(
            model, val_loader, criterion, DEVICE)

        log.info(f"{epoch:>6} | {phase:>8} | {tr_loss:>8.4f} | {tr_acc:>7.4f} | "
                 f"{vl_loss:>8.4f} | {vl_acc:>7.4f}")

        history.append(dict(epoch=epoch, phase=phase,
                            tr_loss=tr_loss, tr_acc=tr_acc,
                            vl_loss=vl_loss, vl_acc=vl_acc))

        # Always save latest
        save_ckpt(model, optimizer, epoch, vl_acc, vl_loss, tag="latest")

        # Save best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            path = save_ckpt(model, optimizer, epoch, vl_acc, vl_loss, tag="best")
            log.info(f"  ★ Best checkpoint → {path}")

    pd.DataFrame(history).to_csv(
        os.path.join(CHECKPOINT_DIR, "history.csv"), index=False)

    # Final report
    log.info(f"\n{'─'*70}")
    log.info(f"Training complete. Best epoch: {best_epoch} | Val acc: {best_val_acc:.4f}")

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "efficientnet_best.pt"), weights_only=False)
    model.load_state_dict(ckpt['model'])
    _, _, preds, labels = eval_epoch(model, val_loader, criterion, DEVICE)

    log.info("\nClassification Report:\n" +
             classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))
    cm    = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    log.info(f"\nConfusion Matrix:\n{cm_df.to_string()}")


if __name__ == "__main__":
    main()