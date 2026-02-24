"""
Training with PyTorch Lightning: ColorContrastNet, AdamW, CosineAnnealing, grad clip.
Checkpoints saved under save_dir/ModelClassName/; TensorBoard logs for loss and pred_L/C mean.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from color.oklch import oklch_to_srgb
from color.apca import apca_contrast
from model.net import ColorContrastNet
from data.sampler import ContrastDataset
from loss import weighted_loss


class ContrastDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        resample_each_epoch: bool,
        num_workers: int = 0,
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.resample_each_epoch = resample_each_epoch
        self.num_workers = num_workers

    def train_dataloader(self):
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        seed = self.seed + epoch if self.resample_each_epoch else self.seed
        ds = ContrastDataset(self.dataset_size, seed=seed)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


LOG_DEBUG_EVERY_N_STEPS = 50


class ColorContrastLightningModule(pl.LightningModule):
    def __init__(
        self,
        hidden: list[int] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        hidden = hidden or [128, 128, 64]
        self.save_hyperparameters()
        self.hidden = hidden
        self.grad_clip = grad_clip
        self.net = ColorContrastNet(hidden=hidden)

    def forward(self, bg_oklch, weights_4, want_lchc_4):
        return self.net(bg_oklch, weights_4, want_lchc_4)

    def training_step(self, batch, batch_idx):
        bg_oklch, weights_4, want_lchc_4 = batch
        pred_lch = self.net(bg_oklch, weights_4, want_lchc_4)
        pred_lch.retain_grad()
        self._last_pred_lch = pred_lch
        pred_srgb = oklch_to_srgb(pred_lch)
        bg_srgb = oklch_to_srgb(bg_oklch)
        pred_contrast = apca_contrast(pred_srgb, bg_srgb).abs()
        want_lch = want_lchc_4[:, :3]
        want_contrast = want_lchc_4[:, 3]
        loss, components = weighted_loss(pred_lch, want_lch, want_contrast, pred_contrast, weights_4)
        with torch.no_grad():
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_pred_L_mean", pred_lch[:, 0].mean(), on_step=False, on_epoch=True)
            self.log("train_pred_C_mean", pred_lch[:, 1].mean(), on_step=False, on_epoch=True)
            self.log("train_L_contrast", components["L_contrast"], on_step=True, on_epoch=True)
            self.log("train_L_L", components["L_L"], on_step=True, on_epoch=True)
            self.log("train_L_C", components["L_C"], on_step=True, on_epoch=True)
            self.log("train_L_H", components["L_H"], on_step=True, on_epoch=True)
            if self.global_step % LOG_DEBUG_EVERY_N_STEPS == 0:
                self.log("train_pred_contrast_mean", pred_contrast.mean(), on_step=True, on_epoch=False)
                self.log("train_want_contrast_mean", want_contrast.mean(), on_step=True, on_epoch=False)
                self.log("train_pred_L_mean_step", pred_lch[:, 0].mean(), on_step=True, on_epoch=False)
                self.log("train_pred_C_mean_step", pred_lch[:, 1].mean(), on_step=True, on_epoch=False)
                self.log("train_pred_H_mean_step", pred_lch[:, 2].mean(), on_step=True, on_epoch=False)
        return loss

    def on_after_backward(self):
        if self.global_step % LOG_DEBUG_EVERY_N_STEPS != 0:
            return
        if hasattr(self, "_last_pred_lch") and self._last_pred_lch.grad is not None:
            g = self._last_pred_lch.grad
            self.log("train_pred_lch_grad_norm", g.norm().item(), on_step=True, on_epoch=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def on_before_optimizer_step(self, optimizer):
        if self.grad_clip > 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip)


class SaveCompatCheckpointCallback(ModelCheckpoint):
    """Save checkpoint in the dict format expected by webui/inference (epoch, model_state_dict, hidden)."""

    def __init__(self, model_dir: Path, save_every: int, hidden: list[int], **kwargs):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.save_every = save_every
        self.hidden = hidden

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.save_every == 0:
            pth = self.model_dir / f"ckpt_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": pl_module.net.state_dict(),
                "hidden": self.hidden,
            }, pth)
            print(f"  saved {pth}")
        if epoch == trainer.max_epochs:
            pth = self.model_dir / "ckpt_final.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": pl_module.net.state_dict(),
                "hidden": self.hidden,
            }, pth)
            print(f"  saved {pth}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train ColorContrastNet (Lightning)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--dataset_size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_resample_each_epoch", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers，Windows 建议保持 0")
    p.add_argument("--log_dir", type=str, default="logs")
    args = p.parse_args()

    pl.seed_everything(args.seed)
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    hidden = [128, 128, 64]
    model = ColorContrastLightningModule(
        hidden=hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    model_dir = Path(args.save_dir) / ColorContrastNet.__name__
    model_dir.mkdir(parents=True, exist_ok=True)

    datamodule = ContrastDataModule(
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        seed=args.seed,
        resample_each_epoch=not args.no_resample_each_epoch,
        num_workers=args.num_workers,
    )

    save_callback = SaveCompatCheckpointCallback(
        model_dir=model_dir,
        save_every=args.save_every,
        hidden=hidden,
    )
    log_dir = Path(args.log_dir)
    logger = TensorBoardLogger(str(log_dir), name="color_contrast")
    csv_logger = CSVLogger(str(log_dir), name="color_contrast")

    if args.device.startswith("cuda:"):
        try:
            gpu_id = int(args.device.split(":")[1])
            devices = [gpu_id]
            accelerator = "gpu"
        except (IndexError, ValueError):
            devices = 1
            accelerator = "gpu" if "cuda" in args.device else "cpu"
    else:
        devices = 1
        accelerator = "gpu" if args.device == "cuda" else "cpu"

    print(f"Device: {args.device} (accelerator={accelerator}, devices={devices}), torch.cuda.is_available()={torch.cuda.is_available()}")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[save_callback],
        logger=[logger, csv_logger],
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=datamodule)
    print("Done.")


if __name__ == "__main__":
    main()
