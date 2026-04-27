import os
import argparse
import torch
from torch.amp import autocast, GradScaler
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np

from model.model import YOLOv4
from bdd100k import BDD100k
from loss import DetectionLoss
from evaluate import DetectionMetric

ANCHORS = [
    [(142, 110), (192, 243), (459, 401)],
    [(36, 75), (76, 55), (72, 146)],
    [(12, 16), (19, 36), (40, 28)],
]

IMG_HEIGHT = 384
IMG_WIDTH = 640


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--root', type=str, required=True, help='path to bdd100k root')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--subset', type=int, default=0, help='use only N images (0 = full dataset)')
    return parser.parse_args()


def save_checkpoint(model, optimizer, scheduler, epoch, map50, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'map50': map50,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['map50']


def train_one_epoch(model, loader, optimizer, loss_fn, device, writer, epoch, scaler):
    model.train()
    running_loss = 0.0

    for imgs, targets in tqdm.tqdm(loader, desc=f'Epoch {epoch} [train]', leave=True):
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        with autocast('cuda'):
            preds = model(imgs)
            loss = loss_fn(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss


def validate(model, loader, loss_fn, metric, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in tqdm.tqdm(loader, desc=f'Epoch {epoch} [val]', leave=True):
            imgs = imgs.to(device)
            targets_dev = [t.to(device) for t in targets]

            with autocast('cuda'):
                preds = model(imgs)
                loss = loss_fn(preds, targets)
            running_loss += loss.item()

            # Accumulate for mAP — keep on CPU
            all_preds.append([p.cpu() for p in preds])
            all_targets.append([t.cpu() for t in targets])

    avg_loss = running_loss / len(loader)
    map50, per_class_ap = metric.compute_map(
        all_preds, all_targets, IMG_HEIGHT, IMG_WIDTH
    )
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('mAP/val@0.5', map50, epoch)
    for cls_idx, ap in per_class_ap.items():
        writer.add_scalar(f'AP/class_{cls_idx}', ap, epoch)

    return avg_loss, map50, per_class_ap


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    writer = SummaryWriter('runs/yolov4_detection')
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH),
                          interpolation=transforms.InterpolationMode.NEAREST),
    ])
    
    S = [(48, 80), (24, 40), (12, 20)]

    train_dataset = BDD100k(root=args.root, train=True, transform=transform, anchors=ANCHORS, S=S)
    
    if args.subset > 0:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(args.subset))
        print(f'Using subset of {args.subset} train images')

    full_val_dataset = BDD100k(root=args.root, train=False, transform=transform, anchors=ANCHORS, S=S)
    val_size = int(0.8 * len(full_val_dataset))
    test_size = len(full_val_dataset) - val_size
    val_dataset, test_dataset = random_split(
        full_val_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    print(f'Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}')

    # ── Model, loss, optimizer, scheduler ────────────────────────────────────
    model = YOLOv4(num_classes=13, num_anchors=3).to(device)
    loss_fn = DetectionLoss(n_classes=13, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    metric = DetectionMetric()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_map50 = 0.0

    if args.checkpoint:
        start_epoch, best_map50 = load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)
        print(f'Resumed from epoch {start_epoch}, best mAP@0.5: {best_map50:.4f}')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        scaler = GradScaler('cuda')
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, writer, epoch, scaler)
        val_loss, map50, per_class_ap = validate(model, val_loader, loss_fn, metric, device, writer, epoch)

        scheduler.step()

        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mAP@0.5: {map50:.4f}')
        for cls_idx, ap in per_class_ap.items():
            print(f'  Class {cls_idx:02d} AP: {ap:.4f}')

        # Save best checkpoint
        if map50 > best_map50:
            best_map50 = map50
            save_checkpoint(
                model, optimizer, scheduler, epoch, map50,
                os.path.join(args.save_dir, 'best.pt')
            )
            print(f'  -> New best mAP@0.5: {best_map50:.4f}, checkpoint saved')

        # Save latest checkpoint every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch, map50,
            os.path.join(args.save_dir, 'latest.pt')
        )

    writer.close()
    print(f'Training complete. Best mAP@0.5: {best_map50:.4f}')


if __name__ == '__main__':
    main()