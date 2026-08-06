# ============================================================
# face_prep.py
# STORYPHY — Step 1: Prep whole photo for cartoonify
# ============================================================
# This script takes the child's photo as input and:
#   1. Normalizes input image size
#   2. Detects the face using MediaPipe (for tilt only)
#   3. Straightens tilted faces
#   4. Removes the background using rembg (whole photo, no crop)
#   5. Fits the whole subject onto a FIXED 800x800 transparent
#      canvas, bottom-anchored + horizontally centered
#
# NOTE: This version does NOT crop to the face. The whole photo
# is kept so nothing gets clipped (hair/neck). Face-size
# consistency across different kids' photos is enforced by the
# cartoonify prompt instead of by this script.
# ============================================================

print("▶ Running face_prep.py — Face Detection & Background Removal")

import cv2
import mediapipe as mp
from rembg import remove
from PIL import Image
import numpy as np
import os
import math

# ── Configuration ────────────────────────────────────────────
# NOTE: No crop/padding step anymore. Whole photo is kept as-is
# (background removed, not cropped) to avoid clipping hair/neck.
# Face-size consistency is now enforced by the cartoonify prompt
# instead of by this script. See README/notes if this changes.

TARGET_LONG_SIDE = 1500  # normalize input image to this size
STANDARD_SIZE    = 800   # every face_ready.png is this square
BOTTOM_MARGIN    = 0     # flush at canvas bottom, no margin (must match cartoonify.py's ANCHOR_BOTTOM_MARGIN)
TARGET_SUBJECT_HEIGHT_PCT = 0.68  # subject bbox height as % of canvas — makes every output the same scale


# ── Helper: Straighten tilted face ───────────────────────────
def straighten_face(image_bgr, detection):
    """
    Detects eye positions and rotates image so eyes are horizontal.
    """
    img_h, img_w = image_bgr.shape[:2]

    keypoints = detection.keypoints
    right_eye = keypoints[0]
    left_eye  = keypoints[1]

    rx = int(right_eye.x * img_w)
    ry = int(right_eye.y * img_h)
    lx = int(left_eye.x  * img_w)
    ly = int(left_eye.y  * img_h)

    dx    = lx - rx
    dy    = ly - ry
    angle = math.degrees(math.atan2(dy, dx))

    print(f"  → Face tilt: {angle:.1f}°")

    if abs(angle) < 1.0:
        print("  → Tilt within tolerance — no rotation needed")
        return image_bgr

    center = (img_w // 2, img_h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    straightened = cv2.warpAffine(
        image_bgr, M, (img_w, img_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )
    print(f"  ✔ Face straightened by {angle:.1f}°")
    return straightened


# ── Main Function ─────────────────────────────────────────────
def prepare_face(input_image_path, output_path):
    """
    Takes a child photo, straightens tilt, removes background
    (whole photo, no crop), and saves a standardized 800x800
    transparent PNG with the subject bottom-anchored + centered.
    """

    print(f"  → Loading image from: {input_image_path}")

    # ── Step 1: Load image ────────────────────────────────────
    image_bgr = cv2.imread(input_image_path)

    if image_bgr is None:
        print(f"  ✖ ERROR: Could not load image at {input_image_path}")
        return None

    orig_h, orig_w = image_bgr.shape[:2]
    print(f"  → Original size: {orig_w}x{orig_h}")

    # ── Step 2: Normalize input image size ────────────────────
    # Photos from different phones have wildly different
    # resolutions. Normalize so face detection is consistent.
    longest_side = max(orig_w, orig_h)
    if longest_side != TARGET_LONG_SIDE:
        scale     = TARGET_LONG_SIDE / longest_side
        new_w     = int(orig_w * scale)
        new_h     = int(orig_h * scale)
        interp    = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
        print(f"  ✔ Normalized to: {new_w}x{new_h}")

    img_h, img_w = image_bgr.shape[:2]
    image_rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Step 3: Detect face using MediaPipe (for straightening only) ──
    # NOTE: detection is used only to find eye positions for tilt
    # correction. It is NOT used for cropping/padding anymore —
    # the whole photo is kept, to avoid clipping hair/neck.
    print("  → Detecting face with MediaPipe (for tilt correction)...")

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    model_path = "config/blaze_face_short_range.tflite"
    if not os.path.exists(model_path):
        print("  → Downloading face detection model (one-time)...")
        import urllib.request
        os.makedirs("config", exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            model_path
        )
        print("  ✔ Model downloaded!")

    base_options = BaseOptions(model_asset_path=model_path)
    options      = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )

    with mp_vision.FaceDetector.create_from_options(options) as detector:

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        results = detector.detect(mp_image)

        if not results.detections:
            print("  ⚠ No face detected — skipping tilt correction, using photo as-is")
        else:
            detection = results.detections[0]
            print(f"  ✔ Face detected! Confidence: {detection.categories[0].score:.2f}")

            # ── Step 4: Straighten tilted face ────────────────
            print("  → Checking face tilt...")
            image_bgr = straighten_face(image_bgr, detection)

    img_h, img_w = image_bgr.shape[:2]

    # ── Step 5: Remove background (whole image, no crop) ─────
    print("  → Removing background from full photo...")
    full_image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    subject_no_bg   = remove(full_image_pil)
    print("  ✔ Background removed")

    # ── Step 6: Scale subject to a fixed % of canvas, then place ──
    # Since background is transparent, we measure the actual subject
    # bounding box (not just fit-to-canvas) and scale so its height
    # always equals a fixed target % of the canvas. This makes every
    # child's face_ready.png the same face scale, regardless of how
    # close/far they were in the original photo — assumes headshot-
    # framed input photos (internal use).
    print(f"  → Scaling subject to {int(TARGET_SUBJECT_HEIGHT_PCT*100)}% of canvas height...")

    alpha = subject_no_bg.split()[-1]
    mask  = alpha.point(lambda a: 255 if a > 25 else 0)
    bbox  = mask.getbbox()

    if not bbox:
        print("  ✖ ERROR: No subject found after background removal.")
        return None

    x1, y1, x2, y2 = bbox
    subject_h = y2 - y1

    target_h = int(STANDARD_SIZE * TARGET_SUBJECT_HEIGHT_PCT)
    scale    = target_h / subject_h

    current_w, current_h = subject_no_bg.size
    scaled_w = int(current_w * scale)
    scaled_h = int(current_h * scale)

    subject_scaled = subject_no_bg.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Re-measure bbox on the scaled image to anchor precisely
    alpha_scaled = subject_scaled.split()[-1]
    mask_scaled  = alpha_scaled.point(lambda a: 255 if a > 25 else 0)
    bbox_scaled  = mask_scaled.getbbox()
    sx1, sy1, sx2, sy2 = bbox_scaled
    subject_center_x   = (sx1 + sx2) // 2
    subject_bottom_y   = sy2

    canvas        = Image.new("RGBA", (STANDARD_SIZE, STANDARD_SIZE), (0, 0, 0, 0))
    bottom_margin = int(STANDARD_SIZE * BOTTOM_MARGIN)
    target_x      = STANDARD_SIZE // 2
    target_y      = STANDARD_SIZE - bottom_margin
    offset_x      = target_x - subject_center_x
    offset_y      = target_y - subject_bottom_y
    canvas.paste(subject_scaled, (offset_x, offset_y), subject_scaled)

    print(f"  ✔ Scaled subject height {subject_h}px → {target_h}px "
          f"({scale:.2f}x), bottom-anchored at y={target_y}")

    # ── Step 8: Save ──────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path, format="PNG")
    print(f"  ✔ Face saved to: {output_path}")
    print("  ✔ face_prep.py complete!\n")

    return output_path


# ── Run directly for testing ──────────────────────────────────
if __name__ == "__main__":

    print("\n  Enter path to child photo")
    print("  (or press ENTER to use input/test_child.jpg)")
    user_input = input("  Image path: ").strip().strip('"').strip("'")

    test_input  = user_input if user_input else "input/test_child.jpg"
    test_output = "output/face_ready_test.png"

    if not os.path.exists(test_input):
        print(f"❌ File not found: {test_input}")
        exit(1)

    result = prepare_face(test_input, test_output)

    if result:
        print(f"✅ Success! Open '{test_output}' to check the result.")
    else:
        print("❌ Something went wrong. Check errors above.")