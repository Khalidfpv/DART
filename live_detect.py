#!/usr/bin/env python3
"""
DART Live Detection — Real-time open-vocabulary object detection.

Supports webcam input and pre-recorded video files with live display.
Built on top of DART (Detect Anything in Real Time) using SAM3.

Usage:
    # Webcam with custom classes:
    python live_detect.py --source 0 --classes person chair laptop cup

    # Pre-recorded video:
    python live_detect.py --source office.mp4 --classes person desk chair

    # All 80 COCO classes:
    python live_detect.py --source 0 --coco

    # With TensorRT acceleration (requires built engines):
    python live_detect.py --source 0 --classes person car chair \
        --trt hf_backbone_1008_fp16.engine \
        --trt-enc-dec enc_dec_fp16.engine

    # With ByteTrack tracking:
    python live_detect.py --source 0 --classes person car --track

    # Save annotated output:
    python live_detect.py --source 0 --classes person chair --save output.mp4

Controls:
    q / ESC  — Quit
    p        — Pause / Resume
    s        — Screenshot (saves current frame)
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import (
    build_pruned_sam3_image_model,
    build_sam3_image_model,
    load_pruned_config,
)
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from demo_multiclass import CLASS_COLOURS


def parse_source(source_str):
    """Parse source string: integer for webcam index, string for file path."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


def draw_detections(frame_bgr, results, class_names, tracks=None):
    """Draw bounding boxes, labels, and confidence scores on a BGR frame.

    Args:
        frame_bgr: OpenCV BGR frame (modified in-place).
        results: Detection results dict from predictor.predict().
        class_names: Ordered list of class names for colour assignment.
        tracks: Optional list of STrack objects from ByteTrack.

    Returns:
        Annotated BGR frame.
    """
    n_colours = len(CLASS_COLOURS)
    class_to_colour = {
        name: CLASS_COLOURS[i % n_colours] for i, name in enumerate(class_names)
    }

    if tracks is not None:
        for track in tracks:
            cls_idx = track.class_id
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else "?"
            colour_rgb = class_to_colour.get(cls_name, CLASS_COLOURS[0])
            colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])

            box = track.box_xyxy
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour_bgr, 2)

            label = f"#{track.track_id} {cls_name} {track.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                frame_bgr, (x1, max(y1 - th - 8, 0)),
                (x1 + tw + 6, max(y1, th + 8)), colour_bgr, -1,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 3, max(y1 - 4, th + 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
    else:
        for i in range(len(results["scores"])):
            cls_name = results["class_names"][i]
            score = results["scores"][i].item()
            box = results["boxes"][i].cpu().tolist()
            colour_rgb = class_to_colour.get(cls_name, CLASS_COLOURS[0])
            colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour_bgr, 2)

            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                frame_bgr, (x1, max(y1 - th - 8, 0)),
                (x1 + tw + 6, max(y1, th + 8)), colour_bgr, -1,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 3, max(y1 - 4, th + 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return frame_bgr


def draw_hud(frame_bgr, fps, n_dets, n_classes, frame_idx, paused=False):
    """Draw heads-up display with FPS, detection count, and status."""
    h, w = frame_bgr.shape[:2]

    # Semi-transparent background bar
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

    # FPS
    fps_color = (0, 255, 0) if fps >= 10 else (0, 255, 255) if fps >= 5 else (0, 0, 255)
    cv2.putText(
        frame_bgr, f"FPS: {fps:.1f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA,
    )

    # Detection count
    cv2.putText(
        frame_bgr, f"Detections: {n_dets}", (180, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # Classes count
    cv2.putText(
        frame_bgr, f"Classes: {n_classes}", (420, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
    )

    # Frame counter
    cv2.putText(
        frame_bgr, f"Frame: {frame_idx}", (w - 180, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
    )

    # Pause indicator
    if paused:
        cv2.putText(
            frame_bgr, "PAUSED", (w // 2 - 60, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA,
        )

    return frame_bgr


def main():
    parser = argparse.ArgumentParser(
        description="DART Live Detection — Real-time open-vocabulary object detection"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Webcam index (0, 1, ...) or path to video file",
    )
    parser.add_argument(
        "--classes", nargs="+", type=str,
        default=["person", "chair", "laptop", "cup", "bottle"],
        help="Target class names to detect",
    )
    parser.add_argument(
        "--coco", action="store_true",
        help="Use all 80 COCO classes (overrides --classes)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to SAM3 checkpoint (default: auto-download from HuggingFace)",
    )
    parser.add_argument(
        "--trt", type=str, default=None, metavar="ENGINE",
        help="Path to TensorRT backbone engine",
    )
    parser.add_argument(
        "--trt-enc-dec", type=str, default=None, metavar="ENGINE",
        help="Path to TensorRT encoder-decoder engine",
    )
    parser.add_argument(
        "--trt-max-classes", type=int, default=4,
        help="Max classes for enc-dec TRT engine (must match export)",
    )
    parser.add_argument(
        "--compile", type=str, default=None, metavar="MODE",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for backbone (alternative to --trt)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--nms", type=float, default=0.7,
        help="NMS IoU threshold (default: 0.7)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Model input resolution (default: 1008, must be divisible by 14)",
    )
    parser.add_argument(
        "--text-cache", type=str, default=None, metavar="PATH",
        help="Path to cached text embeddings (.pt) for faster startup",
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Enable ByteTrack multi-object tracking",
    )
    parser.add_argument(
        "--track-thresh", type=float, default=0.5,
        help="ByteTrack high score threshold",
    )
    parser.add_argument(
        "--match-thresh", type=float, default=0.5,
        help="ByteTrack IoU matching threshold",
    )
    parser.add_argument(
        "--max-time-lost", type=int, default=30,
        help="ByteTrack frames before removing lost track",
    )
    parser.add_argument(
        "--class-agnostic-nms", type=float, default=None,
        help="Class-agnostic NMS threshold before tracking (e.g. 0.7)",
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="Save annotated output to video file",
    )
    parser.add_argument(
        "--no-hud", action="store_true",
        help="Disable the heads-up display (FPS, detection count, etc.)",
    )
    parser.add_argument(
        "--skip-blocks", type=str, default=None,
        help="Comma-separated ViT block indices to skip (pruning)",
    )
    parser.add_argument(
        "--mask-blocks", type=str, default=None,
        help="Fine-grained sub-block pruning spec",
    )
    args = parser.parse_args()

    # Validate
    if args.imgsz % 14 != 0:
        print(f"ERROR: --imgsz must be divisible by 14, got {args.imgsz}")
        sys.exit(1)

    if args.trt is None and args.compile is None:
        print("NOTE: No --trt or --compile specified. Using --compile default")
        args.compile = "default"

    if args.coco:
        from sam3.coco_classes import COCO_CLASSES
        args.classes = COCO_CLASSES

    # Parse pruning options
    skip_blocks = None
    if args.skip_blocks:
        skip_blocks = set(int(x.strip()) for x in args.skip_blocks.split(","))

    mask_blocks = None
    if args.mask_blocks:
        mask_blocks = [s.strip() for s in args.mask_blocks.split(",")]

    device = "cuda"
    print(f"{'='*60}")
    print(f"  DART Live Detection")
    print(f"{'='*60}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Source: {args.source}")
    print(f"  Classes: {len(args.classes)} ({', '.join(args.classes[:5])}{'...' if len(args.classes) > 5 else ''})")
    print(f"  Resolution: {args.imgsz}")
    print(f"  Backbone: {'TRT' if args.trt else f'torch.compile({args.compile})'}")
    print(f"  Enc-dec: {'TRT' if args.trt_enc_dec else 'PyTorch'}")
    print(f"  Tracking: {'ByteTrack' if args.track else 'Off'}")
    print(f"{'='*60}")

    # --- Load model ---
    text_cache_exists = args.text_cache and os.path.exists(args.text_cache)
    use_trt_only = (
        text_cache_exists
        and args.trt
        and args.trt_enc_dec
        and args.checkpoint is None
    )

    if use_trt_only:
        from sam3.model.sam3_multiclass_fast import _TRTModelStub
        print("Loading: TRT-only mode (no checkpoint — text from cache)")
        model = _TRTModelStub(device=device)
    else:
        skip_msg = f", skip_blocks={sorted(skip_blocks)}" if skip_blocks else ""
        print(f"Loading SAM3 model...{skip_msg}")
        pruned_config = (
            load_pruned_config(args.checkpoint) if args.checkpoint else None
        )
        if pruned_config is not None:
            print(f"  Detected pruned checkpoint: {pruned_config}")
            model = build_pruned_sam3_image_model(
                checkpoint_path=args.checkpoint,
                pruning_config=pruned_config,
                device=device,
                eval_mode=True,
                skip_blocks=skip_blocks,
            )
            if model.transformer.decoder.presence_token is not None:
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=args.checkpoint,
                eval_mode=True,
                skip_blocks=skip_blocks,
                mask_blocks=mask_blocks,
            )

    # Precompute position encoding for non-default resolution
    if args.imgsz != 1008:
        pos_enc = model.backbone.vision_backbone.position_encoding
        pos_enc.precompute_for_resolution(args.imgsz)

    # --- Create predictor ---
    predictor = Sam3MultiClassPredictorFast(
        model,
        device=device,
        resolution=args.imgsz,
        use_fp16=True,
        detection_only=True,
        trt_engine_path=args.trt,
        compile_mode=args.compile if not args.trt else None,
        trt_enc_dec_engine_path=args.trt_enc_dec,
        trt_max_classes=args.trt_max_classes,
    )

    print(f"Setting {len(args.classes)} classes...")
    predictor.set_classes(args.classes, text_cache=args.text_cache)

    # --- Warmup ---
    print("Running warmup passes...")
    dummy_img = Image.new("RGB", (args.imgsz, args.imgsz))
    with torch.inference_mode():
        for i in range(3):
            state = predictor.set_image(dummy_img)
            predictor.predict(state, confidence_threshold=args.confidence)
    torch.cuda.synchronize()
    print("Warmup complete.")

    # --- Create tracker ---
    tracker = None
    if args.track:
        from sam3.tracking import BYTETracker
        ca_nms = args.class_agnostic_nms if args.class_agnostic_nms is not None else 1.0
        tracker = BYTETracker(
            track_thresh=args.track_thresh,
            match_thresh=args.match_thresh,
            max_time_lost=args.max_time_lost,
            class_agnostic_nms_thresh=ca_nms,
        )

    # --- Open video source ---
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {args.source}")
        sys.exit(1)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_webcam = isinstance(source, int)
    print(f"Source opened: {src_w}x{src_h} @ {src_fps:.1f} FPS {'(webcam)' if is_webcam else '(file)'}")

    # --- Video writer ---
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps, (src_w, src_h))
        print(f"Recording to: {args.save}")

    # --- Detection loop ---
    print(f"\nStarting detection... Press 'q' or ESC to quit, 'p' to pause, 's' to screenshot.\n")

    fps_history = deque(maxlen=30)
    frame_idx = 0
    paused = False
    screenshot_count = 0

    try:
        while True:
            # Handle pause
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("p"):
                    paused = False
                elif key == ord("q") or key == 27:
                    break
                continue

            ret, frame_bgr = cap.read()
            if not ret:
                if is_webcam:
                    continue
                else:
                    print("End of video.")
                    break

            t_start = time.perf_counter()

            # Convert BGR → RGB → PIL for DART predictor
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run detection
            with torch.inference_mode():
                state = predictor.set_image(pil_image)
                results = predictor.predict(
                    state,
                    confidence_threshold=args.confidence,
                    nms_threshold=args.nms,
                )

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            inference_ms = (t_end - t_start) * 1000
            current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0
            fps_history.append(current_fps)
            avg_fps = sum(fps_history) / len(fps_history)

            n_dets = len(results["scores"])

            # Run tracker if enabled
            tracks = None
            if tracker is not None and n_dets > 0:
                boxes_np = results["boxes"].cpu().numpy()
                scores_np = results["scores"].cpu().numpy()
                class_ids_np = results["class_ids"].cpu().numpy()
                tracks = tracker.update(boxes_np, scores_np, class_ids_np)
            elif tracker is not None:
                tracks = tracker.update(
                    np.empty((0, 4), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int64),
                )

            # Draw detections and HUD
            annotated = draw_detections(
                frame_bgr.copy(), results, args.classes, tracks=tracks
            )
            if not args.no_hud:
                annotated = draw_hud(
                    annotated, avg_fps, n_dets, len(args.classes), frame_idx
                )

            # Display
            cv2.imshow("DART Live Detection", annotated)

            # Save to output video
            if writer is not None:
                writer.write(annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("p"):
                paused = True
            elif key == ord("s"):
                screenshot_name = f"screenshot_{screenshot_count:04d}.jpg"
                cv2.imwrite(screenshot_name, annotated)
                print(f"  Screenshot saved: {screenshot_name}")
                screenshot_count += 1

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")

    # --- Cleanup ---
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # --- Print summary ---
    if frame_idx > 0 and len(fps_history) > 0:
        print(f"\n{'='*50}")
        print(f"  Session Summary")
        print(f"{'='*50}")
        print(f"  Frames processed: {frame_idx}")
        print(f"  Average FPS: {sum(fps_history) / len(fps_history):.1f}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        if args.save:
            print(f"  Output saved: {args.save}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
