"""
LSTM training — standalone verze train_lstm_pytorch.ipynb cells 0-12.
Trénuje BiLSTM+Attention model na RWF-2000 datasetu (binary NonFight/Fight).

Použití:
    python run_train_lstm.py                 # full train (100 epoch)
    python run_train_lstm.py --epochs 50     # custom epochs
    python run_train_lstm.py --no-augment    # vypnout augmentaci
"""
import argparse
import glob
import json
import logging
import os
import pickle
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===== Configuration =====
N_KEYPOINTS = 14
SEQUENCE_LENGTH = 15
STEP_SIZE = 5
CLASS_NAMES = ['NonFight', 'Fight']
N_CLASSES = 2

BATCH_SIZE = 64
DEFAULT_EPOCHS = 50           # v2: max 50, early stop usually triggers ~15-25
LR = 1e-3
WEIGHT_DECAY = 5e-4           # v2: increased from 1e-4 for stronger reg
GRADIENT_CLIP = 1.0

USE_FOCAL_LOSS = False        # v2: switched off, balanced data doesn't need it
FOCAL_GAMMA = 2.0

# v2: smaller model (~250k params vs v1 ~963k) — víc generalizuje na 7568 samples
HIDDEN_DIM = 64               # v2: 128 → 64
N_LSTM_LAYERS = 1             # v2: 2 → 1
ATTENTION_HEADS = 2           # v2: 4 → 2
DROPOUT = 0.5                 # v2: 0.3 → 0.5

# v2: early stopping
EARLY_STOPPING_PATIENCE = 10  # zastavit pokud val_acc neroste 10 epoch
KEYPOINT_DROPOUT_PROB = 0.1   # 10% šance na maskování každého keypointu (simulace occlusion)

RWF_SKELETONS_DIR = 'data/RWF-2000/skeletons'
MODEL_SAVE_PATH = 'best_lstm_pytorch.pt'
HISTORY_SAVE_PATH = 'training_history.json'
LOG_DIR = 'logs'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LEFT_RIGHT_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]


# ===== Logging =====
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(LOG_DIR, f'train_lstm_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_file


# ===== Skeleton preprocessing =====
def normalize_skeleton(skeleton, eps=1e-6):
    sk = skeleton.copy()
    hip_mid = (sk[:, 6, :2] + sk[:, 7, :2]) / 2
    sh_mid = (sk[:, 0, :2] + sk[:, 1, :2]) / 2
    torso = np.maximum(np.linalg.norm(sh_mid - hip_mid, axis=-1, keepdims=True), eps)
    sk[..., :2] = (sk[..., :2] - hip_mid[:, None, :]) / torso[:, :, None]
    return sk


def add_velocity_features(skeleton):
    sk = skeleton[..., :2]
    velocity = np.zeros_like(sk)
    velocity[1:] = sk[1:] - sk[:-1]
    return np.concatenate([sk, velocity], axis=-1)


def augment_skeleton(skeleton):
    sk = skeleton.copy()
    # Mirror flip (50%)
    if random.random() < 0.5:
        sk[..., 0] = -sk[..., 0]
        for l, r in LEFT_RIGHT_PAIRS:
            sk[:, [l, r]] = sk[:, [r, l]]
    # Random rotation (±8°)
    angle = np.random.uniform(-0.14, 0.14)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=sk.dtype)
    sk[..., :2] = sk[..., :2] @ R.T
    # Random scale
    sk[..., :2] *= np.random.uniform(0.9, 1.1)
    # Coordinate jitter
    sk[..., :2] += np.random.normal(0, 0.02, sk[..., :2].shape).astype(sk.dtype)
    # v2: Keypoint dropout — náhodně zerouj 10% keypointů (simulace occlusion v CCTV)
    if KEYPOINT_DROPOUT_PROB > 0:
        T, K = sk.shape[:2]
        mask = np.random.random((T, K)) < KEYPOINT_DROPOUT_PROB
        sk[mask, :2] = 0
    return sk


# ===== Dataset =====
class SkeletonDataset(Dataset):
    def __init__(self, rwf_dir, split='train', sequence_length=SEQUENCE_LENGTH,
                 step_size=STEP_SIZE, augment=False):
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.augment = augment
        self.samples = []

        for cls_dir, label in [('NonFight', 0), ('Fight', 1)]:
            d = os.path.join(rwf_dir, split, cls_dir)
            if not os.path.exists(d):
                logging.warning(f'{d} not found')
                continue
            for pkl_path in sorted(glob.glob(os.path.join(d, '*.pkl'))):
                try:
                    with open(pkl_path, 'rb') as f:
                        result = pickle.load(f)
                except Exception as e:
                    logging.warning(f'Skip {pkl_path}: {e}')
                    continue
                for track in result.get('person_tracks', []):
                    sk = track['skeletons'][..., :2]
                    if len(sk) < self.sequence_length:
                        continue
                    for i in range(0, len(sk) - self.sequence_length + 1, self.step_size):
                        self.samples.append((sk[i:i + self.sequence_length], label))

        labels = [s[1] for s in self.samples]
        logging.info(f'Dataset {split}: {len(self.samples)} samples '
                     f'(NonFight={labels.count(0)}, Fight={labels.count(1)})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sk, lbl = self.samples[idx]
        sk = np.nan_to_num(sk.copy().astype(np.float32))
        sk = normalize_skeleton(sk)
        if self.augment:
            sk = augment_skeleton(sk)
        sk = add_velocity_features(sk).reshape(self.sequence_length, -1)
        return torch.from_numpy(sk).float(), torch.tensor(lbl, dtype=torch.long)


# ===== Model =====
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, n_layers=N_LSTM_LAYERS,
                 n_classes=N_CLASSES, dropout=DROPOUT, attention_heads=ATTENTION_HEADS):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, attention_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out).mean(dim=1)
        return self.classifier(x)


# ===== Loss =====
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ===== Training utilities =====
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses, correct, total = [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        losses.append(loss.item())
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return float(np.mean(losses)), correct / max(total, 1)


@torch.no_grad()
def eval_model(model, loader, loss_fn, device):
    model.eval()
    losses, correct, total = [], 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return float(np.mean(losses)), correct / max(total, 1), all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    log_file = setup_logging()
    logging.info('=' * 70)
    logging.info('LSTM training started')
    logging.info(f'  Device: {DEVICE}')
    if torch.cuda.is_available():
        logging.info(f'  GPU: {torch.cuda.get_device_name(0)}')
    logging.info(f'  Epochs: {args.epochs}')
    logging.info(f'  Batch size: {args.batch_size}')
    logging.info(f'  Augmentation: {not args.no_augment}')
    logging.info(f'  Sequence length: {SEQUENCE_LENGTH}')
    logging.info(f'  Step size: {STEP_SIZE}')
    logging.info(f'  Log file: {log_file}')
    logging.info('=' * 70)

    # Datasets
    logging.info('Loading datasets...')
    train_ds = SkeletonDataset(RWF_SKELETONS_DIR, split='train', augment=not args.no_augment)
    val_ds = SkeletonDataset(RWF_SKELETONS_DIR, split='val', augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weights
    labels = np.array([s[1] for s in train_ds.samples])
    class_counts = np.bincount(labels, minlength=N_CLASSES)
    class_counts = np.maximum(class_counts, 1)
    class_weights = len(labels) / (N_CLASSES * class_counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    logging.info(f'Class counts: {class_counts}, weights: {class_weights}')

    # Model
    model = BiLSTMAttention(input_dim=N_KEYPOINTS * 4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Model: {n_params:,} params')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if USE_FOCAL_LOSS:
        loss_fn = FocalLoss(alpha=class_weights_t, gamma=FOCAL_GAMMA)
        logging.info(f'Loss: FocalLoss (gamma={FOCAL_GAMMA})')
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
        logging.info('Loss: weighted CrossEntropyLoss')

    # Train
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        va_loss, va_acc, va_preds, va_labels = eval_model(model, val_loader, loss_fn, DEVICE)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': va_acc,
            }, MODEL_SAVE_PATH)
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - t_start
        logging.info(
            f'Epoch {epoch+1:3d}/{args.epochs} | '
            f'tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | '
            f'val_loss={va_loss:.4f} val_acc={va_acc:.3f}'
            f'{"  *BEST*" if is_best else ""} | '
            f'patience={epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} | '
            f'elapsed={elapsed/60:.1f}min'
        )

        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logging.info(f'Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)')
            break

    # Save history
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t_start
    logging.info('=' * 70)
    logging.info(f'TRAINING DONE in {elapsed/60:.1f} min')
    logging.info(f'  Best val accuracy: {best_val_acc:.3f}')
    logging.info(f'  Model saved: {MODEL_SAVE_PATH}')
    logging.info(f'  History saved: {HISTORY_SAVE_PATH}')

    # Final eval with best checkpoint
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    _, _, va_preds, va_labels_arr = eval_model(model, val_loader, loss_fn, DEVICE)

    from sklearn.metrics import classification_report, confusion_matrix
    logging.info('')
    logging.info('Classification Report (best checkpoint, val split):')
    report = classification_report(va_labels_arr, va_preds, target_names=CLASS_NAMES,
                                    digits=3, zero_division=0)
    for line in report.split('\n'):
        logging.info(f'  {line}')

    cm = confusion_matrix(va_labels_arr, va_preds, labels=list(range(N_CLASSES)))
    logging.info('Confusion Matrix:')
    logging.info(f'  Predicted →')
    logging.info(f'  True ↓     {CLASS_NAMES[0]:>10} {CLASS_NAMES[1]:>10}')
    for i in range(N_CLASSES):
        logging.info(f'  {CLASS_NAMES[i]:<10} {cm[i, 0]:>10} {cm[i, 1]:>10}')
    logging.info('=' * 70)


if __name__ == '__main__':
    main()
