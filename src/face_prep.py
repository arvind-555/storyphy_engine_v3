# ============================================================
# face_prep.py
# STORYPHY — Step 1: Face Detection, Crop & Background Removal
# ============================================================
# This script takes the child's photo as input and:
#   1. Normalizes input image size
#   2. Detects the face using MediaPipe
#   3. Straightens tilted faces
#   4. Crops the face with padding
#   5. Removes the background using rembg
#   6. Saves to a FIXED 800x800 square (consistent every time)
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
PADDING_TOP    = 0.8   # space above face (forehead + hair)
PADDING_BOTTOM = 0.1   # space below face (chin only, no neck)
PADDING_SIDES  = 0.3   # space left and right

TARGET_LONG_SIDE = 1500  # normalize input image to this size
STANDARD_SIZE    = 800   # every face_ready.png is this square


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
    Takes a child photo, detects face, crops, removes background,
    and saves a standardized 800x800 transparent PNG.
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

    # ── Step 3: Detect face using MediaPipe ───────────────────
    print("  → Detecting face with MediaPipe...")

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
            print("  ✖ ERROR: No face detected.")
            return None

        detection = results.detections[0]
        print(f"  ✔ Face detected! Confidence: {detection.categories[0].score:.2f}")

        # ── Step 4: Straighten tilted face ────────────────────
        print("  → Checking face tilt...")
        image_bgr = straighten_face(image_bgr, detection)

        # Re-detect on straightened image
        mp_image_straight = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        )
        results_straight = detector.detect(mp_image_straight)

        if results_straight.detections:
            detection = results_straight.detections[0]
            print("  ✔ Face re-detected on straightened image")
        else:
            print("  ⚠ Re-detection failed — using original detection")

        # Get bounding box
        bbox = detection.bounding_box
        x    = bbox.origin_x
        y    = bbox.origin_y
        w    = bbox.width
        h    = bbox.height

    # ── Step 5: Crop face with padding ───────────────────────
    print(f"  → Cropping with padding...")

    pad_top    = int(h * PADDING_TOP)
    pad_bottom = int(h * PADDING_BOTTOM)
    pad_left   = int(w * PADDING_SIDES)
    pad_right  = int(w * PADDING_SIDES)

    x1 = max(0, x - pad_left)
    y1 = max(0, y - pad_top)
    x2 = min(img_w, x + w + pad_right)
    y2 = min(img_h, y + h + pad_bottom)

    face_crop_bgr = image_bgr[y1:y2, x1:x2]
    print(f"  → Cropped face size: {x2-x1}x{y2-y1} pixels")

    # ── Step 6: Remove background ─────────────────────────────
    print("  → Removing background...")
    face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))
    face_no_bg    = remove(face_crop_pil)
    print("  ✔ Background removed")

    # ── Step 7: Standardize to fixed 800x800 square ──────────
    # This is the KEY step — every face_ready.png is EXACTLY
    # 800x800 regardless of input photo size or distance.
    # This guarantees the AI and compositor get consistent input.
    print(f"  → Standardizing to {STANDARD_SIZE}x{STANDARD_SIZE}px...")

    current_w, current_h = face_no_bg.size
    scale    = min(STANDARD_SIZE / current_w, STANDARD_SIZE / current_h)
    scaled_w = int(current_w * scale)
    scaled_h = int(current_h * scale)

    face_scaled = face_no_bg.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Place on fixed transparent square canvas — centered
    canvas   = Image.new("RGBA", (STANDARD_SIZE, STANDARD_SIZE), (0, 0, 0, 0))
    offset_x = (STANDARD_SIZE - scaled_w) // 2
    offset_y = (STANDARD_SIZE - scaled_h) // 2
    canvas.paste(face_scaled, (offset_x, offset_y), face_scaled)

    print(f"  ✔ Standardized: {current_w}x{current_h} → {STANDARD_SIZE}x{STANDARD_SIZE}px")

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