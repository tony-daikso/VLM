"""
在 PadChest-GR train split（3,185 張胸腔 X-ray，unlabeled）上重新跑一次 DINO self-supervised
pretraining，架構完全沿用學長 DINO_LLM/prior/dino/vision_transformer.py 裡現成的
vit_large（embed_dim=1280, depth=12, num_heads=20, patch_size=16）—— 這是使用者選定要忠實
重現的部分，不改架構。

單一 GPU 版本（沒有 distributed training，跟官方 facebookresearch/dino 的 main_dino.py 比少了
多機多卡那段，其餘：multi-crop augmentation、student/teacher EMA、DINO loss（centering +
sharpening）、cosine schedule（lr/wd/momentum/teacher_temp）都照官方邏輯實作，方便跟官方版本
對照。

已知風險（見規劃文件，這裡也留一份提醒）：DINO self-supervised pretraining 官方是在 ImageNet
（百萬級圖片）上做的，這裡只有 3,185 張，訓練出來的表徵品質預期會明顯弱化，這是「忠實重現土炮
流程」的必然代價，不是這支程式碼寫錯。

用法：
    python3 main_dino.py --manifest /root/Desktop/VLM/data/X-ray/PadChest-GR/processed/manifest.jsonl \
        --data_root /root/Desktop/VLM/data/X-ray/PadChest-GR/processed \
        --output_dir /datadrive/VLM/DINO_LLM/X-ray/stage_dino_ssl/checkpoints
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dino.utils as dino_utils
import dino.vision_transformer as vits
from stage_dino_ssl.dataset import DataAugmentationDINO, PadChestSSLDataset


def get_args_parser():
    p = argparse.ArgumentParser("DINO SSL pretraining on PadChest-GR (chest X-ray)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--out_dim", type=int, default=65536)
    p.add_argument("--norm_last_layer", type=dino_utils.bool_flag, default=True)
    p.add_argument("--momentum_teacher", type=float, default=0.996)
    p.add_argument("--use_bn_in_head", type=dino_utils.bool_flag, default=False)

    p.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--freeze_last_layer", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--weight_decay_end", type=float, default=0.4)
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--use_fp16", type=dino_utils.bool_flag, default=True)

    p.add_argument("--global_crops_scale", type=float, nargs="+", default=(0.4, 1.0))
    p.add_argument("--local_crops_number", type=int, default=6)
    p.add_argument("--local_crops_scale", type=float, nargs="+", default=(0.05, 0.4))
    # 官方 DINO 在 ImageNet 上用 224/96 這組解析度（不是 U-VLM 那邊的 512x512 —— DINO 的
    # global/local crop 尺寸是自監督訓練的超參數，跟下游影像原始解析度無關，pos_embed 靠
    # interpolate_pos_encoding 自動適應任意 crop 尺寸）。實測 512 global crop 在 ViT-L +
    # batch_size 16 + 6 local crops 會直接把 46GB GPU OOM，所以維持官方預設值。
    p.add_argument("--global_crop_size", type=int, default=224)
    p.add_argument("--local_crop_size", type=int, default=96)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--saveckp_freq", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p


class DINOLoss(nn.Module):
    """跟官方 main_dino.py 裡的 DINOLoss 完全一致：teacher centering + sharpening 避免崩塌。"""

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss, n_loss_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def main(args):
    dino_utils.fix_random_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = DataAugmentationDINO(
        args.global_crops_scale, args.local_crops_scale, args.local_crops_number,
        global_crop_size=args.global_crop_size, local_crop_size=args.local_crop_size,
    )
    dataset = PadChestSSLDataset(args.manifest, args.data_root, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    print(f"train images: {len(dataset)}, iters/epoch: {len(loader)}")

    student = vits.vit_large(patch_size=args.patch_size, drop_path_rate=0.1)
    teacher = vits.vit_large(patch_size=args.patch_size)
    embed_dim = student.embed_dim

    student = dino_utils.MultiCropWrapper(student, vits.DINOHead(
        embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer,
    ))
    teacher = dino_utils.MultiCropWrapper(
        teacher, vits.DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head),
    )
    student, teacher = student.to(device), teacher.to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"student/teacher vit_large built, embed_dim={embed_dim}")

    dino_loss = DINOLoss(
        args.out_dim, args.local_crops_number + 2, args.warmup_teacher_temp, args.teacher_temp,
        args.warmup_teacher_temp_epochs, args.epochs,
    ).to(device)

    params_groups = dino_utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    lr_schedule = dino_utils.cosine_scheduler(
        args.lr * args.batch_size / 256, args.min_lr, args.epochs, len(loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = dino_utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(loader),
    )
    momentum_schedule = dino_utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(loader))

    start_epoch = 0
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.exists(ckpt_path):
        to_restore = {"epoch": 0}
        dino_utils.restart_from_checkpoint(
            ckpt_path, run_variables=to_restore, student=student, teacher=teacher,
            optimizer=optimizer, dino_loss=dino_loss,
        )
        start_epoch = to_restore["epoch"]
        print(f"resumed from {ckpt_path} at epoch {start_epoch}")

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    print(f"starting DINO SSL training from epoch {start_epoch}")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = train_one_epoch(
            student, teacher, dino_loss, loader, optimizer, lr_schedule, wd_schedule,
            momentum_schedule, epoch, args, device, fp16_scaler,
        )
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
            "dino_loss": dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        torch.save(save_dict, ckpt_path)
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))
        elapsed = time.time() - start_time
        print(f"epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} elapsed={elapsed/60:.1f}min")

    print(f"done, final checkpoint at {ckpt_path}")


def train_one_epoch(student, teacher, dino_loss, loader, optimizer, lr_schedule, wd_schedule,
                     momentum_schedule, epoch, args, device, fp16_scaler=None):
    student.train()
    total_loss, n_batches = 0.0, 0
    n_iters_per_epoch = len(loader)
    for it, images in enumerate(loader):
        global_it = n_iters_per_epoch * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[global_it]

        images = [im.to(device, non_blocking=True) for im in images]

        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                dino_utils.clip_gradients(student, args.clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                dino_utils.clip_gradients(student, args.clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[global_it]
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        total_loss += loss.item()
        n_batches += 1
        if it % 20 == 0:
            print(f"  epoch {epoch} iter {it}/{n_iters_per_epoch} loss={loss.item():.4f} lr={lr_schedule[global_it]:.6f}")

    return total_loss / max(n_batches, 1)


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
