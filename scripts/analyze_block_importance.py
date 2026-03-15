#!/usr/bin/env python3
"""Analyze per-block importance in SAM3 ViT-H backbone.

Measures the contribution of each ViT block by removing it entirely and
computing feature reconstruction loss (L2) against the full model.  Two
analysis modes:

  1. **Individual**: Remove each block one-at-a-time → rank by importance.
  2. **Greedy**: Iteratively remove the least-important block → cumulative
     quality-vs-speed tradeoff curve.

Uses the model's built-in `skip_blocks` mechanism (full block removal,
not sub-block masking).  Measures loss after the FPN neck so results
reflect impact on downstream detection, not just trunk output.

All analysis is sequential (one candidate at a time) to keep VRAM usage
bounded — the ViT-H backbone is large and runs at high resolution.

Usage:
    python scripts/analyze_block_importance.py \
        --checkpoint sam3.pt \
        --calib-dir train2017 \
        --num-images 20 \
        --imgsz 1008
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_calibration_images(calib_dir, num_images, imgsz, device):
    """Load and preprocess calibration images."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_paths = sorted(
        p for p in Path(calib_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if len(all_paths) < num_images:
        print(f"WARNING: Only {len(all_paths)} images found, using all")
        num_images = len(all_paths)

    step = max(1, len(all_paths) // num_images)
    paths = all_paths[:num_images * step:step][:num_images]

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(imgsz, max_size=imgsz + 1),
        v2.CenterCrop(imgsz),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img).to(device))

    return torch.stack(tensors)


@torch.no_grad()
def run_trunk(trunk, images, skip_blocks=None, batch_size=4):
    """Run ViT trunk with optional block skipping.

    Returns (N, C, h, w) trunk output features (float32).
    """
    orig_skip = trunk.skip_blocks

    if skip_blocks is not None:
        trunk.skip_blocks = set(skip_blocks)
    else:
        trunk.skip_blocks = set()

    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = trunk(batch)
        # trunk returns list of feature tensors; take last (only level used)
        all_features.append(outputs[-1].float())

    trunk.skip_blocks = orig_skip

    return torch.cat(all_features, dim=0)


def compute_loss(ref, pruned):
    """L2 reconstruction loss + cosine similarity."""
    l2 = (ref - pruned).pow(2).mean().item()
    # Cosine sim over flattened features
    ref_flat = ref.flatten(1)
    pruned_flat = pruned.flatten(1)
    cos = torch.nn.functional.cosine_similarity(ref_flat, pruned_flat, dim=1).mean().item()
    return l2, cos


def run_analysis(
    checkpoint_path,
    calib_dir,
    num_images=20,
    imgsz=1008,
    batch_size=4,
    num_greedy=16,
    protect_global=True,
    device="cuda",
):
    from sam3.model_builder import build_sam3_image_model

    print(f"Loading SAM3 model from {checkpoint_path} ...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=False,
    )

    trunk = model.backbone.vision_backbone.trunk
    num_blocks = len(trunk.blocks)
    global_ids = set(trunk.full_attn_ids)

    # Free everything except the trunk to save VRAM
    del model.transformer
    del model.dot_prod_scoring
    del model.segmentation_head
    del model.geometry_encoder
    del model.backbone.language_backbone
    torch.cuda.empty_cache()

    print(f"\nViT-H backbone: {num_blocks} blocks")
    print(f"  Global attention blocks: {sorted(global_ids)}")
    print(f"  Protect global blocks: {protect_global}")
    print(f"  Resolution: {imgsz}px")

    # Load calibration images
    print(f"\nLoading {num_images} calibration images from {calib_dir} ...")
    images = load_calibration_images(calib_dir, num_images, imgsz, device)
    print(f"  Loaded: {images.shape}")

    # Reference features (full model)
    print("Computing reference trunk features ...")
    ref_features = run_trunk(trunk, images, skip_blocks=None, batch_size=batch_size)
    print(f"  Shape: {ref_features.shape}, norm: {ref_features.norm().item():.1f}")

    # ---------------------------------------------------------------
    # Phase 1: Individual block importance
    # ---------------------------------------------------------------
    print(f"\n{'='*75}")
    print("PHASE 1: Individual block importance (remove one block at a time)")
    print(f"{'='*75}")

    candidates = list(range(num_blocks))
    if protect_global:
        candidates = [i for i in candidates if i not in global_ids]
        print(f"  Testing {len(candidates)} non-global blocks "
              f"(protecting {sorted(global_ids)})")
    else:
        print(f"  Testing all {len(candidates)} blocks")

    individual_results = {}  # bi -> (l2, cos)
    header = f"{'Block':>5}  {'Type':>7}  {'L2':>12}  {'CosSim':>8}  {'Time':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for bi in candidates:
        t0 = time.perf_counter()
        pruned = run_trunk(trunk, images, skip_blocks={bi}, batch_size=batch_size)
        l2, cos = compute_loss(ref_features, pruned)
        elapsed = time.perf_counter() - t0
        is_global = "GLOBAL" if bi in global_ids else "window"
        individual_results[bi] = (l2, cos)
        print(f"{bi:>5}  {is_global:>7}  {l2:>12.6f}  {cos:>8.4f}  {elapsed:>5.1f}s")

    # Sort by L2 loss (ascending = least important first)
    ranked = sorted(individual_results.items(), key=lambda x: x[1][0])

    print(f"\n{'='*75}")
    print("INDIVIDUAL RANKING (least important first):")
    print(f"{'='*75}")
    for rank, (bi, (l2, cos)) in enumerate(ranked):
        is_global = "GLOBAL" if bi in global_ids else "window"
        print(f"  {rank+1:>2}. Block {bi:>2} ({is_global})  "
              f"L2={l2:.6f}  cos={cos:.4f}")

    # ---------------------------------------------------------------
    # Phase 2: Greedy cumulative removal
    # ---------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"PHASE 2: Greedy block removal (up to {num_greedy} blocks)")
    print(f"{'='*75}")

    removed = set()
    greedy_order = []
    remaining = set(candidates)

    # Per-block timing estimates (compiled, ms)
    # Window block: ~1.8ms (0.6 attn + 1.2 mlp)
    # Global block: ~4.5ms (3.0 attn + 1.5 mlp)
    BLOCK_MS = {bi: (4.5 if bi in global_ids else 1.8) for bi in range(num_blocks)}

    header = (f"{'Step':>4}  {'Block':>5}  {'Type':>7}  "
              f"{'StepL2':>10}  {'CumulL2':>10}  {'CosSim':>8}  {'SavedMs':>8}  {'Time':>6}")
    print(f"\n{header}")
    print("-" * len(header))

    for step in range(min(num_greedy, len(remaining))):
        best_l2 = float("inf")
        best_block = None
        t0 = time.perf_counter()

        for bi in remaining:
            trial_skip = removed | {bi}
            pruned = run_trunk(trunk, images, skip_blocks=trial_skip, batch_size=batch_size)
            l2, _ = compute_loss(ref_features, pruned)
            if l2 < best_l2:
                best_l2 = l2
                best_block = bi

        removed.add(best_block)
        remaining.discard(best_block)

        # Cumulative metrics
        cumul_pruned = run_trunk(trunk, images, skip_blocks=removed, batch_size=batch_size)
        cumul_l2, cumul_cos = compute_loss(ref_features, cumul_pruned)

        saved_ms = sum(BLOCK_MS[b] for b in removed)
        is_global = "GLOBAL" if best_block in global_ids else "window"
        elapsed = time.perf_counter() - t0
        greedy_order.append((best_block, cumul_l2, cumul_cos, saved_ms))

        print(f"{step+1:>4}  {best_block:>5}  {is_global:>7}  "
              f"{best_l2:>10.6f}  {cumul_l2:>10.6f}  {cumul_cos:>8.4f}  "
              f"{saved_ms:>7.1f}  {elapsed:>5.1f}s")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    total_ms = 65.0  # approx compiled backbone time

    print(f"\n{'='*75}")
    print("GREEDY REMOVAL ORDER (least important first):")
    print(f"{'='*75}")
    for rank, (bi, l2, cos, saved) in enumerate(greedy_order):
        is_global = "GLOBAL" if bi in global_ids else "window"
        pct = saved / total_ms * 100
        print(f"  {rank+1:>2}. Block {bi:>2} ({is_global})  "
              f"L2={l2:.6f}  cos={cos:.4f}  saved={saved:.1f}ms ({pct:.0f}%)")

    print(f"\n{'='*75}")
    print("QUALITY vs SPEED TRADEOFF:")
    print(f"{'='*75}")
    print(f"{'Blocks removed':>15}  {'CumulL2':>10}  {'CosSim':>8}  "
          f"{'Backbone ms':>12}  {'Speedup':>8}")
    print(f"{'-'*60}")
    print(f"{'0':>15}  {'0.000000':>10}  {'1.0000':>8}  "
          f"{total_ms:>11.1f}  {'1.00x':>8}")
    for rank, (bi, l2, cos, saved) in enumerate(greedy_order):
        remain_ms = total_ms - saved
        speedup = total_ms / remain_ms if remain_ms > 0 else float("inf")
        print(f"{rank+1:>15}  {l2:>10.6f}  {cos:>8.4f}  "
              f"{remain_ms:>11.1f}  {speedup:>7.2f}x")

    # Output --skip-blocks string
    skip_str = ",".join(str(b) for b, _, _, _ in greedy_order)
    print(f"\nFull greedy order as --skip-blocks: \"{skip_str}\"")

    # Suggest sweet spots
    print(f"\nSuggested configs:")
    for n in [4, 8, 12, 16]:
        if n <= len(greedy_order):
            blocks = [b for b, _, _, _ in greedy_order[:n]]
            saved = sum(BLOCK_MS[b] for b in blocks)
            l2 = greedy_order[n-1][1]
            cos = greedy_order[n-1][2]
            remain = total_ms - saved
            skip = ",".join(str(b) for b in sorted(blocks))
            print(f"  skip-{n}: {remain:.1f}ms ({total_ms/remain:.2f}x)  "
                  f"L2={l2:.6f}  cos={cos:.4f}  --skip-blocks \"{skip}\"")

    return individual_results, greedy_order


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-block importance in SAM3 ViT-H backbone"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to SAM3 checkpoint (.pt)"
    )
    parser.add_argument(
        "--calib-dir", required=True,
        help="Directory with calibration images (e.g. train2017/)"
    )
    parser.add_argument(
        "--num-images", type=int, default=20,
        help="Number of calibration images (default 20)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Input image resolution (default 1008)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for forward passes"
    )
    parser.add_argument(
        "--num-greedy", type=int, default=16,
        help="Max blocks to remove in greedy search (default 16)"
    )
    parser.add_argument(
        "--no-protect-global", action="store_true",
        help="Allow removing global attention blocks (7, 15, 23, 31)"
    )
    args = parser.parse_args()

    run_analysis(
        checkpoint_path=args.checkpoint,
        calib_dir=args.calib_dir,
        num_images=args.num_images,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        num_greedy=args.num_greedy,
        protect_global=not args.no_protect_global,
    )


if __name__ == "__main__":
    main()
