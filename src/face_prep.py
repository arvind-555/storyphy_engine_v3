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
#   6. ALSO writes a separate hi-res tight head crop, used only
#      as the cartoonify identity input
#
# NOTE: This version does NOT crop face_ready.png to the face.
# The whole photo is kept so nothing gets clipped (hair/neck).
#
# TWO OUTPUTS, TWO JOBS:
#   face_ready.png      → standardized canvas, for placement/compositing
#   face_crop_hires.png → tight head crop, maximum pixels on the face,
#                         for gpt-image-1 identity transfer only
# These have different requirements, so they get different images.
# ============================================================

print("▶ Running face_prep.py — Face Detection & Background Removal")

import cv2
import mediapipe as mp
from rembg import remove
from PIL import Image
import numpy as np
from scipy import ndimage
import os
import math

# ── Configuration ────────────────────────────────────────────
# NOTE: No crop/padding step for face_ready.png. Whole photo is kept
# as-is (background removed, not cropped) to avoid clipping hair/neck.
# Face-size consistency is enforced by the scale-to-target logic below.

TARGET_LONG_SIDE = 1500  # normalize working image to this size
STANDARD_SIZE    = 800   # every face_ready.png is this square
BOTTOM_MARGIN    = 0     # flush at canvas bottom, no margin (must match cartoonify.py's ANCHOR_BOTTOM_MARGIN)
TARGET_SUBJECT_HEIGHT_PCT = 0.68  # subject bbox height as % of canvas — makes every output the same scale
MIN_ROW_PIXELS   = 5     # legacy, unused (see clean_subject)
FEATHER_PAD      = 4     # dilate kept region so soft hair feathering survives

# ── Hi-res head crop (cartoonify input only) ─────────────────
# Separate branch off the ORIGINAL image, before TARGET_LONG_SIDE
# downscaling. gpt-image-1 needs as many real pixels on the face as
# possible for identity transfer; a crop taken from the already-
# downscaled 800px canvas would just be an upscale of ~350px of face.
HIRES_LONG_SIDE    = 2400   # cap for the crop branch (memory guard)
CARTOON_INPUT_SIZE = 1024   # final size sent to gpt-image-1
CROP_PAD_TOP       = 0.75   # headroom above face box, as multiple of face height
CROP_PAD_BOTTOM    = 1.55   # measured DOWN from face box top (chin + neck)
CROP_PAD_SIDE      = 0.45   # side padding, as multiple of face width


def clean_subject(pil_rgba, alpha_threshold=25, feather_pad=FEATHER_PAD):
    """
    Isolates the child (largest connected region of non-transparent
    pixels) and ERASES everything else by zeroing its alpha. Returns
    (cleaned_image, bbox_of_subject).

    Re-saved / AI-generated PNGs often carry artifacts — stray specks or
    dense edge lines. Measuring around them isn't enough: they stay in
    the saved file, so anything downstream that trims with a plain
    getbbox() re-inherits the same off-centre error. Deleting them here
    means every later step sees a clean subject.
    """
    rgba = np.array(pil_rgba.convert("RGBA"))
    alpha = rgba[:, :, 3]
    mask = alpha > alpha_threshold

    labeled, n = ndimage.label(mask)
    if n == 0:
        return pil_rgba, None

    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_label = int(np.argmax(sizes)) + 1
    keep = labeled == largest_label

    keep_dilated = ndimage.binary_dilation(keep, iterations=feather_pad)
    rgba[:, :, 3] = np.where(keep_dilated, alpha, 0)

    ys, xs = np.where(keep)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Image.fromarray(rgba), bbox


def get_robust_bbox(pil_rgba, alpha_threshold=25, min_pixels=MIN_ROW_PIXELS):
    """Bbox of the largest connected region only (no image modification)."""
    _, bbox = clean_subject(pil_rgba, alpha_threshold)
    return bbox


# ── MediaPipe helpers ─────────────────────────────────────────
def _ensure_model():
    """Downloads the BlazeFace model once, returns its path."""
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
    return model_path


def _detect_face(image_bgr):
    """
    Runs MediaPipe face detection on a BGR image.
    Returns the first detection, or None if no face found.
    """
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    options = mp_vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        min_detection_confidence=0.5
    )

    with mp_vision.FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(mp_image)

    return results.detections[0] if results.detections else None


def _rotate(image_bgr, angle):
    """
    Rotates about the image centre. Used to apply the same tilt
    correction to the hi-res branch that straighten_face applied
    to the working image, so both branches stay in agreement.
    """
    if abs(angle) < 1.0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale=1.0)
    return cv2.warpAffine(
        image_bgr, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )


# ── Helper: Straighten tilted face ───────────────────────────
def straighten_face(image_bgr, detection):
    """
    Detects eye positions and rotates image so eyes are horizontal.
    Returns (image, angle_applied) — the angle is needed so the
    hi-res branch can be rotated by the same amount.
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
        return image_bgr, angle

    center = (img_w // 2, img_h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    straightened = cv2.warpAffine(
        image_bgr, M, (img_w, img_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )
    print(f"  ✔ Face straightened by {angle:.1f}°")
    return straightened, angle


# ── Hi-res head crop for cartoonify ──────────────────────────
def make_hires_head_crop(image_bgr_hires, detection, out_path):
    """
    Tight square head+neck crop at maximum available resolution,
    background removed, saved at CARTOON_INPUT_SIZE.

    This is the cartoonify identity input. It exists purely to put as
    many real pixels on the face as possible — placement and scaling
    are NOT its job, they stay with face_ready.png.

    Returns the face-box height in source px (the QA number that tells
    you whether the source photo had the detail to make this worthwhile).
    """
    h, w = image_bgr_hires.shape[:2]

    # Tasks API BoundingBox is in PIXELS, not relative coords
    bb = detection.bounding_box
    fx, fy = bb.origin_x, bb.origin_y
    fw, fh = bb.width, bb.height

    top    = fy - fh * CROP_PAD_TOP
    bottom = fy + fh * CROP_PAD_BOTTOM
    left   = fx - fw * CROP_PAD_SIDE
    right  = fx + fw * (1 + CROP_PAD_SIDE)

    # square it around the centre
    cx, cy = (left + right) / 2, (top + bottom) / 2
    side   = max(right - left, bottom - top)
    l, t   = int(cx - side / 2), int(cy - side / 2)
    r, b   = int(cx + side / 2), int(cy + side / 2)

    # Paste onto a white canvas so out-of-bounds regions pad cleanly.
    # White rather than edge-replicate: rembg separates a flat white
    # field far more reliably than smeared edge pixels.
    src    = Image.fromarray(cv2.cvtColor(image_bgr_hires, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGB", (r - l, b - t), (255, 255, 255))
    canvas.paste(src, (-l, -t))

    crop = canvas.resize((CARTOON_INPUT_SIZE, CARTOON_INPUT_SIZE), Image.LANCZOS)

    print("  → Removing background from head crop...")
    crop_no_bg = remove(crop)
    crop_no_bg, _ = clean_subject(crop_no_bg)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    crop_no_bg.save(out_path, format="PNG")

    return int(fh)


# ── Main Function ─────────────────────────────────────────────
def prepare_face(input_image_path, output_path):
    """
    Takes a child photo, straightens tilt, removes background
    (whole photo, no crop), and saves a standardized 800x800
    transparent PNG with the subject bottom-anchored + centered.

    Also writes face_crop_hires.png alongside it — the tight head
    crop used as the cartoonify identity input.
    """

    print(f"  → Loading image from: {input_image_path}")

    # ── Step 1: Load image ────────────────────────────────────
    image_bgr = cv2.imread(input_image_path)

    if image_bgr is None:
        print(f"  ✖ ERROR: Could not load image at {input_image_path}")
        return None

    orig_h, orig_w = image_bgr.shape[:2]
    print(f"  → Original size: {orig_w}x{orig_h}")

    # ── Step 1b: Keep a hi-res copy BEFORE normalizing ────────
    # TARGET_LONG_SIDE would throw away detail the crop branch needs.
    image_hires = image_bgr.copy()
    if max(orig_w, orig_h) > HIRES_LONG_SIDE:
        s = HIRES_LONG_SIDE / max(orig_w, orig_h)
        image_hires = cv2.resize(
            image_bgr, (int(orig_w * s), int(orig_h * s)),
            interpolation=cv2.INTER_AREA
        )
    print(f"  → Hi-res branch: {image_hires.shape[1]}x{image_hires.shape[0]}")

    # ── Step 2: Normalize working image size ──────────────────
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

    # ── Step 3: Detect face (for straightening only) ──────────
    # NOTE: detection is used only to find eye positions for tilt
    # correction. It is NOT used for cropping face_ready.png —
    # the whole photo is kept, to avoid clipping hair/neck.
    print("  → Detecting face with MediaPipe (for tilt correction)...")

    detection  = _detect_face(image_bgr)
    tilt_angle = 0.0

    if detection is None:
        print("  ⚠ No face detected — skipping tilt correction, using photo as-is")
    else:
        print(f"  ✔ Face detected! Confidence: {detection.categories[0].score:.2f}")

        # ── Step 4: Straighten tilted face ────────────────────
        print("  → Checking face tilt...")
        image_bgr, tilt_angle = straighten_face(image_bgr, detection)

    img_h, img_w = image_bgr.shape[:2]

    # ── Step 4b: Hi-res head crop for cartoonify ─────────────
    # The same rotation is re-applied to the hi-res copy, then the
    # face is re-detected on it — post-rotation coordinates from the
    # working image would be both stale and wrongly scaled.
    image_hires = _rotate(image_hires, tilt_angle)
    hires_det   = _detect_face(image_hires)

    if hires_det is None:
        print("  ⚠ No face in hi-res branch — skipping cartoonify crop")
    else:
        crop_path = os.path.join(
            os.path.dirname(output_path) or ".", "face_crop_hires.png"
        )
        head_px = make_hires_head_crop(image_hires, hires_det, crop_path)
        print(f"  ✔ Cartoonify crop saved: {crop_path}")
        print(f"  [QA] source face-box height: {head_px}px")

    # ── Step 5: Remove background (whole image, no crop) ─────
    print("  → Removing background from full photo...")
    full_image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    subject_no_bg  = remove(full_image_pil)
    print("  ✔ Background removed")

    # ── Step 6: Scale subject to a fixed % of canvas, then place ──
    # Since background is transparent, we measure the actual subject
    # bounding box (not just fit-to-canvas) and scale so its height
    # always equals a fixed target % of the canvas. This makes every
    # child's face_ready.png the same face scale, regardless of how
    # close/far they were in the original photo — assumes headshot-
    # framed input photos (internal use).
    print(f"  → Scaling subject to {int(TARGET_SUBJECT_HEIGHT_PCT*100)}% of canvas height...")

    subject_no_bg, bbox = clean_subject(subject_no_bg)

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

    # Scale the ORIGINAL bbox coordinates by the same factor instead of
    # re-measuring on the resized image. Re-measuring is unreliable —
    # thin hair strands / soft edges can drop below alpha_threshold
    # after LANCZOS resampling, shrinking the measured subject and
    # throwing off both the scale-to-target and the anchor position.
    # Geometry (scaling known coordinates) doesn't have this problem.
    subject_center_x = int(((x1 + x2) // 2) * scale)
    subject_bottom_y = int(y2 * scale)

    canvas        = Image.new("RGBA", (STANDARD_SIZE, STANDARD_SIZE), (0, 0, 0, 0))
    bottom_margin = int(STANDARD_SIZE * BOTTOM_MARGIN)
    target_x      = STANDARD_SIZE // 2
    target_y      = STANDARD_SIZE - bottom_margin
    offset_x      = target_x - subject_center_x
    offset_y      = target_y - subject_bottom_y
    canvas.paste(subject_scaled, (offset_x, offset_y), subject_scaled)

    print(f"  ✔ Scaled subject height {subject_h}px → {target_h}px "
          f"({scale:.2f}x), bottom-anchored at y={target_y}")

    # ── Step 7: Save ──────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
        print("   Also check 'output/face_crop_hires.png' — that's the cartoonify input.")
    else:
        print("❌ Something went wrong. Check errors above.")