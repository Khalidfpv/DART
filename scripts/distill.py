#!/usr/bin/env python3
"""
SAM3 backbone distillation: train a lightweight FPN adapter to replace ViT-H,
or self-distill a pruned SAM3 backbone.

Phase 1 (adapter-only): Frozen teacher backbone + frozen student backbone,
    train only ~5M adapter params via feature MSE.

Phase 2 (optional): Fine-tune student backbone with lower lr.
    Use --lora-rank to apply LoRA instead of full fine-tuning.

Prune mode (--phase prune): Self-distillation for sub-block pruning.
    Loads the full SAM3 backbone as teacher, deep-copies it, masks sub-blocks,
    and fine-tunes the remaining blocks to recover quality. Saves a pruned
    checkpoint with masked block weights removed.

Usage (single GPU, adapter distillation):
    python scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --epochs 5 --batch-size 2 --lr 1e-3

Usage (8xH100 via SLURM srun):
    salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=12
    srun python scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --epochs 5 --batch-size 16 --lr 1e-3

Usage (8xH100 via torchrun, if preferred):
    torchrun --nproc_per_node=8 scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --epochs 5 --batch-size 16 --lr 1e-3

Distill from pruned teacher to lightweight student (SBP-8):
    python scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone repvit_m2_3 \
        --mask-blocks "25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp" \
        --epochs 5 --batch-size 2 --lr 1e-3

Self-distill pruned SAM3 backbone (SBP-8, 8xH100 via salloc):
    srun --ntasks=1 torchrun --nproc_per_node=8 scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --phase prune \
        --mask-blocks "25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp" \
        --epochs 5 --batch-size 4 --lr 1e-4

Self-distill pruned SAM3 backbone (SBP-16, 8xH100 via salloc):
    srun --ntasks=1 torchrun --nproc_per_node=8 scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --phase prune \
        --mask-blocks "25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp,26:attn,22:mlp,24:attn,18:attn,20:mlp,21:attn,25:mlp,18:mlp" \
        --epochs 5 --batch-size 4 --lr 1e-4

Self-distill with full block removal (skip-8, 8xH100 via salloc):
    srun --ntasks=1 torchrun --nproc_per_node=8 scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --phase prune \
        --skip-blocks "25,28,27,22,20,30,26,24" \
        --epochs 5 --batch-size 4 --lr 1e-4

Phase 2 (backbone fine-tuning with LoRA, after phase 1):
    python scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --adapter-checkpoint distill_checkpoints/adapter_final.pt \
        --phase 2 --lora-rank 4 --epochs 3 --lr 1e-4

Test (single GPU, no srun needed):
    python scripts/distill.py \
        --checkpoint /path/to/sam3.pt \
        --adapter-checkpoint distill_checkpoints/adapter_final.pt \
        --test-image /path/to/image.jpg --test-prompt "car"
"""

import argparse
import os
import time

import torch


def parse_mask_blocks(mask_str):
    """Parse mask-blocks string like '25:attn,28:mlp' into list of (block_idx, sub_type)."""
    if not mask_str:
        return []
    masks = []
    for item in mask_str.split(","):
        item = item.strip()
        if not item:
            continue
        block_str, sub_type = item.split(":")
        block_idx = int(block_str)
        assert sub_type in ("attn", "mlp"), f"Invalid sub-type: {sub_type}"
        masks.append((block_idx, sub_type))
    return masks


def apply_mask_blocks(model, mask_blocks):
    """Apply sub-block masks to the teacher ViT backbone.

    Args:
        model: SAM3 image model (has .backbone.vision_backbone.trunk.blocks)
        mask_blocks: list of (block_idx, sub_type) from parse_mask_blocks()
    """
    if not mask_blocks:
        return
    trunk = model.backbone.vision_backbone.trunk
    for block_idx, sub_type in mask_blocks:
        block = trunk.blocks[block_idx]
        if sub_type == "attn":
            block.mask_attn = True
        elif sub_type == "mlp":
            block.mask_mlp = True
    masked_attn = sum(1 for _, t in mask_blocks if t == "attn")
    masked_mlp = sum(1 for _, t in mask_blocks if t == "mlp")
    print(f"  Masked {len(mask_blocks)} sub-blocks: {masked_attn} attn, {masked_mlp} mlp")


def run_phase1(args):
    """Phase 1: Adapter-only distillation."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.distillation.student_backbone import build_student_backbone
    from sam3.distillation.distill_trainer import (
        DistillationTrainer,
        dist_print,
        is_main_process,
        setup_distributed,
    )

    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.distributed.is_initialized() else args.device

    dist_print("=" * 60)
    dist_print("Phase 1: Adapter-Only Feature Distillation")
    dist_print("=" * 60)

    # Build teacher model (full SAM3 — we only use its backbone)
    # Each rank loads independently; weights are identical across ranks.
    dist_print("\nLoading teacher SAM3 model...")
    teacher = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        load_from_HF=args.checkpoint is None,
        enable_inst_interactivity=False,
    )
    teacher.eval()

    # Apply sub-block pruning masks to teacher backbone
    mask_blocks = parse_mask_blocks(args.mask_blocks)
    if mask_blocks:
        dist_print(f"\nApplying sub-block pruning ({len(mask_blocks)} sub-blocks):")
        apply_mask_blocks(teacher, mask_blocks)

    # Build student backbone (same init on all ranks — timm pretrained)
    dist_print(f"\nBuilding student backbone: {args.backbone}")
    student_bb = build_student_backbone(
        config_name=args.backbone,
        pretrained=True,
        freeze_backbone=True,
    )
    student_bb = student_bb.to(device)
    dist_print(f"  Trainable adapter params: {student_bb.trainable_params:,}")

    # Train
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_backbone=student_bb,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
        device=device,
    )
    # Store mask-blocks string so checkpoint records it
    trainer.mask_blocks_str = args.mask_blocks

    # Free teacher model components we don't need (save VRAM)
    del teacher.transformer
    del teacher.dot_prod_scoring
    del teacher.segmentation_head
    del teacher.geometry_encoder
    del teacher.backbone.language_backbone
    torch.cuda.empty_cache()
    dist_print(f"\nFreed non-backbone teacher components to save VRAM")

    trainer.train()


def run_phase2(args):
    """Phase 2: Fine-tune student backbone (full or LoRA) + FPN adapter."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.distillation.student_backbone import build_student_backbone
    from sam3.distillation.distill_trainer import (
        DistillationTrainer,
        dist_print,
        setup_distributed,
    )

    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.distributed.is_initialized() else args.device

    dist_print("=" * 60)
    dist_print("Phase 2: Student Backbone Fine-Tuning")
    dist_print("=" * 60)

    # Build teacher model (full SAM3 — we only use its backbone)
    dist_print("\nLoading teacher SAM3 model...")
    teacher = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        load_from_HF=args.checkpoint is None,
        enable_inst_interactivity=False,
    )
    teacher.eval()

    # Apply sub-block pruning masks to teacher backbone
    mask_blocks = parse_mask_blocks(args.mask_blocks)
    if mask_blocks:
        dist_print(f"\nApplying sub-block pruning ({len(mask_blocks)} sub-blocks):")
        apply_mask_blocks(teacher, mask_blocks)

    use_lora = args.lora_rank > 0

    # Build student backbone — frozen if using LoRA, unfrozen otherwise
    dist_print(f"\nBuilding student backbone: {args.backbone}")
    student_bb = build_student_backbone(
        config_name=args.backbone,
        pretrained=True,
        freeze_backbone=use_lora,
    )

    # Apply LoRA to the timm backbone (not the FPN adapters)
    if use_lora:
        from sam3.distillation.lora import apply_lora, lora_param_count
        n_wrapped = apply_lora(
            student_bb.backbone, rank=args.lora_rank, alpha=args.lora_rank
        )
        dist_print(
            f"  LoRA rank={args.lora_rank}: wrapped {n_wrapped} layers, "
            f"{lora_param_count(student_bb):,} LoRA params"
        )
    else:
        dist_print("  Full fine-tuning (no LoRA)")

    student_bb = student_bb.to(device)

    # Load adapter weights from phase 1
    if args.adapter_checkpoint:
        dist_print(f"Loading phase 1 weights from {args.adapter_checkpoint}")
        ckpt = torch.load(args.adapter_checkpoint, map_location=device)
        student_bb.load_state_dict(ckpt["student_state_dict"], strict=False)
    else:
        dist_print("WARNING: No --adapter-checkpoint provided. Starting from scratch.")

    total_params = sum(p.numel() for p in student_bb.parameters())
    trainable = sum(p.numel() for p in student_bb.parameters() if p.requires_grad)
    dist_print(f"  Total params: {total_params:,}")
    dist_print(f"  Trainable params: {trainable:,}")

    # Train with same DistillationTrainer
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_backbone=student_bb,
        data_dir=args.data_dir,
        output_dir=args.output_dir + "_phase2",
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
        device=device,
    )
    # Store mask-blocks string so checkpoint records it
    trainer.mask_blocks_str = args.mask_blocks

    # Free teacher model components we don't need (save VRAM)
    del teacher.transformer
    del teacher.dot_prod_scoring
    del teacher.segmentation_head
    del teacher.geometry_encoder
    del teacher.backbone.language_backbone
    torch.cuda.empty_cache()
    dist_print(f"\nFreed non-backbone teacher components to save VRAM")

    trainer.train()


def run_prune(args):
    """Self-distillation for block/sub-block pruning.

    Loads the full SAM3 backbone as teacher (frozen), deep-copies it,
    applies block skips and/or sub-block masks, and fine-tunes the
    remaining layers to recover quality. Saves a pruned checkpoint
    with removed block weights stripped.
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.distillation.prune_trainer import PruneDistillTrainer
    from sam3.distillation.distill_trainer import (
        dist_print,
        setup_distributed,
    )

    mask_blocks = parse_mask_blocks(args.mask_blocks)
    skip_blocks = []
    if args.skip_blocks:
        skip_blocks = [int(x.strip()) for x in args.skip_blocks.split(",")]

    if not mask_blocks and not skip_blocks:
        print("ERROR: --mask-blocks and/or --skip-blocks required for --phase prune")
        return

    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.distributed.is_initialized() else args.device

    dist_print("=" * 60)
    dist_print("Pruning Self-Distillation")
    dist_print("=" * 60)

    # Load full SAM3 model (teacher = unpruned backbone)
    dist_print("\nLoading SAM3 model...")
    teacher = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        load_from_HF=args.checkpoint is None,
        enable_inst_interactivity=False,
    )
    teacher.eval()

    if mask_blocks:
        masked_attn = sum(1 for _, t in mask_blocks if t == "attn")
        masked_mlp = sum(1 for _, t in mask_blocks if t == "mlp")
        dist_print(
            f"\nSub-block masks: {len(mask_blocks)} "
            f"({masked_attn} attn, {masked_mlp} mlp)"
        )
    if skip_blocks:
        dist_print(f"Full block skips: {sorted(skip_blocks)}")

    # Free non-backbone components before deep-copy (save VRAM)
    del teacher.transformer
    del teacher.dot_prod_scoring
    del teacher.segmentation_head
    del teacher.geometry_encoder
    del teacher.backbone.language_backbone
    torch.cuda.empty_cache()
    dist_print("Freed non-backbone components to save VRAM")

    # Train
    trainer = PruneDistillTrainer(
        teacher_model=teacher,
        mask_blocks=mask_blocks,
        mask_blocks_str=args.mask_blocks or "",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
        device=device,
        skip_blocks=skip_blocks,
    )

    trainer.train()


def run_test(args):
    """Test student model with a single image + text prompt (single GPU only)."""
    from sam3.distillation.sam3_student import build_sam3_student_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image

    print("=" * 60)
    print("Testing Student Model")
    print("=" * 60)

    # Build student model
    print(f"\nBuilding student model with {args.backbone} backbone...")
    model = build_sam3_student_model(
        backbone_config=args.backbone,
        teacher_checkpoint=args.checkpoint,
        load_from_HF=args.checkpoint is None,
        device=args.device,
        freeze_teacher=True,
    )

    # Load adapter weights
    if args.adapter_checkpoint:
        print(f"Loading adapter weights from {args.adapter_checkpoint}")
        ckpt = torch.load(args.adapter_checkpoint, map_location=args.device)
        model.backbone.student_backbone.load_state_dict(
            ckpt["student_state_dict"]
        )

    model.eval()

    # Create processor
    processor = Sam3Processor(model, device=args.device)

    # Run inference
    image = Image.open(args.test_image)
    print(f"Image: {args.test_image} ({image.size[0]}x{image.size[1]})")
    print(f"Prompt: '{args.test_prompt}'")

    t0 = time.perf_counter()
    state = processor.set_image(image)
    t_backbone = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = processor.set_text_prompt(args.test_prompt, state)
    t_head = time.perf_counter() - t0

    num_dets = len(state["scores"])
    print(f"\nResults: {num_dets} detections")
    print(f"  Backbone: {t_backbone*1000:.1f}ms")
    print(f"  Head: {t_head*1000:.1f}ms")
    print(f"  Total: {(t_backbone+t_head)*1000:.1f}ms")

    for i in range(min(num_dets, 10)):
        score = state["scores"][i].item()
        box = state["boxes"][i].tolist()
        print(
            f"  [{i}] score={score:.3f} "
            f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
        )


def main():
    parser = argparse.ArgumentParser(description="SAM3 backbone distillation")

    parser.add_argument(
        "--phase", type=str, default="1", choices=["1", "2", "prune"],
        help="Training phase: 1=adapter-only, 2=encoder fine-tuning, prune=self-distill pruned backbone"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to image directory (e.g., COCO train2017)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to teacher SAM3 checkpoint (default: download from HF)"
    )
    parser.add_argument(
        "--adapter-checkpoint", type=str, default=None,
        help="Path to adapter checkpoint (for phase 2 or testing)"
    )
    parser.add_argument(
        "--backbone", type=str, default="efficientvit_l1",
        choices=["efficientvit_l1", "efficientvit_l2", "repvit_m2_3", "tiny_vit_21m", "vit_base", "vit_base_dinov3"],
        help="Student backbone architecture (phases 1/2 only)"
    )
    parser.add_argument("--output-dir", type=str, default="distill_checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Sub-block pruning
    parser.add_argument(
        "--mask-blocks", type=str, default=None,
        help=(
            "Sub-blocks to mask, e.g. '25:attn,28:mlp,27:attn'. "
            "In phases 1/2: masks teacher backbone before distilling to student. "
            "In prune mode: defines which sub-blocks to remove and self-distill."
        ),
    )

    # Full block removal
    parser.add_argument(
        "--skip-blocks", type=str, default=None,
        help=(
            "Entire blocks to remove, e.g. '25,28,27'. "
            "In prune mode: defines which full blocks to skip and self-distill."
        ),
    )

    # LoRA (phase 2 only)
    parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank for phase 2 backbone fine-tuning (0=full fine-tune)"
    )

    # Test mode
    parser.add_argument("--test-image", type=str, default=None)
    parser.add_argument("--test-prompt", type=str, default="object")

    args = parser.parse_args()

    if args.test_image:
        run_test(args)
    elif args.data_dir is None:
        parser.print_help()
        print("\nProvide --data-dir for training or --test-image for testing.")
    elif args.phase == "prune":
        run_prune(args)
    elif args.phase == "1":
        run_phase1(args)
    elif args.phase == "2":
        run_phase2(args)


if __name__ == "__main__":
    main()
