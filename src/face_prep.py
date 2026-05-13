# ============================================================
# face_prep.py
# STORYPHY — Step 1: Face Detection, Crop & Background Removal
# ============================================================
# This script takes the child's photo as input and:
#   1. Detects the face using MediaPipe
#   2. Crops the face with padding (so it doesn't look too tight)
#   3. Removes the background using rembg
#   4. Saves a clean transparent PNG ready for cartoonification
# ============================================================

print("▶ Running face_prep.py — Face Detection & Background Removal")

import cv2
import mediapipe as mp
from rembg import remove
from PIL import Image
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────
# How much extra space to add around the detected face (30% on each side)
# Padding ratios — tuned for children's portraits
PADDING_TOP    = 0.8   # extra space above (forehead + hair)
PADDING_BOTTOM = 0.1   # reduced — just below chin, no neck
PADDING_SIDES  = 0.3   # slightly tighter on sides

# ── Main Function ───────────────────────────────────────────
def prepare_face(input_image_path, output_path):
    """
    Takes a child photo, detects the face, crops it with padding,
    removes the background, and saves a transparent PNG.

    Args:
        input_image_path : path to the original child photo
        output_path      : where to save the final transparent face PNG
    """

    print(f"  → Loading image from: {input_image_path}")

    # ── Step 1: Load the image ──────────────────────────────
    # OpenCV loads images in BGR format by default
    image_bgr = cv2.imread(input_image_path)

    if image_bgr is None:
        print(f"  ✖ ERROR: Could not load image at {input_image_path}")
        print("    Make sure the file exists and is a valid JPG or PNG.")
        return None

    # Get image dimensions
    img_height, img_width = image_bgr.shape[:2]
    print(f"  → Image size: {img_width}x{img_height} pixels")

    # Convert BGR to RGB (MediaPipe works with RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Step 2: Detect face using MediaPipe (new API) ───────
    print("  → Detecting face with MediaPipe...")

    # New MediaPipe API uses FaceDetector instead of solutions.face_detection
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    # Download the face detection model file if not already present
    model_path = "config/blaze_face_short_range.tflite"
    if not os.path.exists(model_path):
        print("  → Downloading face detection model (one-time only)...")
        import urllib.request
        os.makedirs("config", exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            model_path
        )
        print("  ✔ Model downloaded!")

    # Set up the face detector
    base_options = BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )

    with mp_vision.FaceDetector.create_from_options(options) as detector:

        # MediaPipe new API needs its own image format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        results = detector.detect(mp_image)

        # Check if any face was found
        if not results.detections:
            print("  ✖ ERROR: No face detected in the image.")
            print("    Try a clearer photo with the face visible and well-lit.")
            return None

        # Use the first detected face (assuming one child per photo)
        detection = results.detections[0]
        print(f"  ✔ Face detected! Confidence: {detection.categories[0].score:.2f}")

        # Get bounding box in pixel coordinates (new API gives pixels directly)
        bbox = detection.bounding_box
        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height

   # ── Step 3: Add padding around the face ─────────────────
    # Different padding for top vs bottom — children's hair needs more room above
    print(f"  → Cropping face with custom padding (top={int(PADDING_TOP*100)}%, sides={int(PADDING_SIDES*100)}%, bottom={int(PADDING_BOTTOM*100)}%)...")

    pad_top    = int(h * PADDING_TOP)
    pad_bottom = int(h * PADDING_BOTTOM)
    pad_left   = int(w * PADDING_SIDES)
    pad_right  = int(w * PADDING_SIDES)

    # Calculate padded crop coordinates, clamped to image boundaries
    x1 = max(0, x - pad_left)
    y1 = max(0, y - pad_top)
    x2 = min(img_width,  x + w + pad_right)
    y2 = min(img_height, y + h + pad_bottom)

    # Crop the face region from the original BGR image
    face_crop_bgr = image_bgr[y1:y2, x1:x2]

    print(f"  → Cropped face size: {x2-x1}x{y2-y1} pixels")

    # ── Step 4: Remove background using rembg ───────────────
    print("  → Removing background (this may take a few seconds)...")

    # Convert cropped face to PIL Image (rembg works with PIL)
    face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))

    # Remove the background — output is RGBA (transparent background)
    face_no_bg = remove(face_crop_pil)

    # ── Step 5: Save the result ─────────────────────────────
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    face_no_bg.save(output_path, format="PNG")
    print(f"  ✔ Face saved to: {output_path}")
    print("  ✔ face_prep.py complete!\n")

    return output_path


# ── Run directly for testing ────────────────────────────────
# This block only runs when you execute this file directly
# (not when it's imported by main.py)
if __name__ == "__main__":

    # ⚠ Change these paths to test with your own image
    test_input  = "input/test_child.jpg"   # put a photo here
    test_output = "output/face_ready.png"  # result will be saved here

    result = prepare_face(test_input, test_output)

    if result:
        print(f"✅ Success! Open '{test_output}' to check the result.")
    else:
        print("❌ Something went wrong. Check the error messages above.")