"""
MNIST DCGAN Training Script

This script implements the complete training pipeline for a DCGAN model
on the MNIST handwritten digit dataset. It includes:

- Command-line interface for configuration
- Automatic device detection (CPU/CUDA/MPS)
- Training loop with progress tracking
- Sample generation and saving
- Checkpoint management
- Reproducible training with fixed seeds

Usage:
    python train.py --epochs 20 --batch-size 128 --subset-size 60000

The script automatically downloads MNIST data and saves generated samples
and model checkpoints to the specified output directory.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as vutils

from mnist_gan_project.models import (
    DCGANDiscriminator,
    DCGANGenerator,
    weights_init,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    beta1: float
    beta2: float
    latent_dim: int
    save_dir: str
    subset_size: Optional[int]
    seed: int
    workers: int
    device: str


def get_dataloaders(batch_size: int, subset_size: Optional[int], workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # scale to [-1, 1]
        ]
    )
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    if subset_size is not None and subset_size < len(train_dataset):
        indices = np.random.RandomState(42).permutation(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices.tolist())
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def save_samples(generator: DCGANGenerator, fixed_noise: torch.Tensor, save_path: str) -> None:
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise).cpu()
        grid = vutils.make_grid(samples, nrow=10, normalize=True, value_range=(-1, 1))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vutils.save_image(grid, save_path)
    generator.train()


def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # Data
    loader = get_dataloaders(cfg.batch_size, cfg.subset_size, cfg.workers)

    # Models
    netG = DCGANGenerator(latent_dim=cfg.latent_dim).to(device)
    netD = DCGANDiscriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    fixed_noise = torch.randn(100, cfg.latent_dim, device=device)

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        running_d_loss = 0.0
        running_g_loss = 0.0

        for real, _ in loader:
            real = real.to(device)
            bsz = real.size(0)

            # Labels: use smoothing for real labels
            real_labels = torch.empty(bsz, device=device).uniform_(0.8, 1.0)
            fake_labels = torch.zeros(bsz, device=device)

            # ---------------------------
            # Train Discriminator
            # ---------------------------
            optimizerD.zero_grad(set_to_none=True)

            logits_real = netD(real)
            loss_real = criterion(logits_real, real_labels)

            noise = torch.randn(bsz, cfg.latent_dim, device=device)
            fake = netG(noise)
            logits_fake = netD(fake.detach())
            loss_fake = criterion(logits_fake, fake_labels)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizerD.step()

            # ---------------------------
            # Train Generator
            # ---------------------------
            optimizerG.zero_grad(set_to_none=True)
            logits_fake_for_g = netD(fake)
            target_real_for_g = torch.ones(bsz, device=device)
            g_loss = criterion(logits_fake_for_g, target_real_for_g)
            g_loss.backward()
            optimizerG.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            step += 1

        avg_d = running_d_loss / len(loader)
        avg_g = running_g_loss / len(loader)
        print(f"Epoch {epoch:03d}/{cfg.epochs} | D: {avg_d:.4f} | G: {avg_g:.4f}")

        # Save periodic samples and checkpoints
        samples_path = os.path.join(cfg.save_dir, "samples", f"epoch_{epoch:04d}.png")
        save_samples(netG, fixed_noise, samples_path)
        torch.save({
            "epoch": epoch,
            "netG": netG.state_dict(),
            "netD": netD.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
            "cfg": cfg.__dict__,
        }, os.path.join(cfg.save_dir, f"checkpoint_{epoch:04d}.pt"))

    # Final sample
    final_path = os.path.join(cfg.save_dir, "samples", "final.png")
    save_samples(netG, fixed_noise, final_path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="runs/mnist_dcgan")
    parser.add_argument("--subset-size", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    subset_size = None if args.subset_size >= 60000 else int(args.subset_size)

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        latent_dim=args.latent_dim,
        save_dir=args.save_dir,
        subset_size=subset_size,
        seed=args.seed,
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)


