# ============================================================
# compositor.py
# STORYPHY — Step 3: Compose Each Page
# ============================================================
# This script takes the cartoon face + zones.json and:
#   1. Loops through all 28 pages
#   2. Opens each template
#   3. Pastes the cartoon face at the correct zone
#   4. Saves each composed page to output/pages/
# ============================================================

print("▶ Running compositor.py — Composing All Pages")

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Configuration ───────────────────────────────────────────
TEMPLATES_DIR      = "templates"
ZONES_FILE         = "config/zones.json"

# Fixed size for the cartoon face before placing in zones.
# Every input photo — regardless of resolution — produces a
# face standardized to this square canvas first.
# Adjust this value if faces consistently look too big or small.
STANDARD_FACE_SIZE = 400  # pixels

# ── Helper: Re-crop cartoon face to consistent framing ───────
def recrop_cartoon_face(cartoon_img):
    """
    The AI returns cartoon faces with inconsistent framing —
    sometimes tight head, sometimes head + long neck.

    This re-detects the actual head in the cartoon and re-crops
    it to a CONSISTENT framing every time, so the face fits the
    same way on every template regardless of AI output variation.

    Returns a re-cropped RGBA image.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    # Convert PIL RGBA to RGB numpy array for detection
    rgb = cartoon_img.convert("RGB")
    rgb_array = np.array(rgb)

    model_path = "config/blaze_face_short_range.tflite"
    if not os.path.exists(model_path):
        print("  ⚠ Face model not found — skipping re-crop")
        return cartoon_img

    base_options = BaseOptions(model_asset_path=model_path)
    options      = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.4
    )

    with mp_vision.FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        results  = detector.detect(mp_image)

        if not results.detections:
            print("  ⚠ No face found in cartoon — using as-is")
            return cartoon_img

        bbox = results.detections[0].bounding_box
        fx, fy = bbox.origin_x, bbox.origin_y
        fw, fh = bbox.width, bbox.height

    # Re-crop with CONSISTENT padding around the detected face
    # These ratios define the fixed framing for every cartoon
    pad_top    = int(fh * 0.85)   # room for hair
    pad_bottom = int(fh * 0.55)   # short neck stub
    pad_side   = int(fw * 0.30)

    img_w, img_h = cartoon_img.size
    x1 = max(0, fx - pad_side)
    y1 = max(0, fy - pad_top)
    x2 = min(img_w, fx + fw + pad_side)
    y2 = min(img_h, fy + fh + pad_bottom)

    cropped = cartoon_img.crop((x1, y1, x2, y2))
    if cropped.size[0] == 0 or cropped.size[1] == 0:
        print("  ⚠ Re-crop gave zero size — using original")
        return cartoon_img
    print(f"  ✔ Cartoon re-cropped to consistent framing: {cropped.size}")
    return cropped

# ── Main Function ─────────────────────────────────────────────
def compose_all_pages(cartoon_face_path, child_name, output_dir):
    """
    Composes all 28 pages by pasting the cartoon face
    onto each template.

    Args:
        cartoon_face_path : path to the cartoon face PNG
        child_name        : child's name string
        output_dir        : where to save composed pages
    """

    print(f"  → Child name   : {child_name}")
    print(f"  → Cartoon face : {cartoon_face_path}")
    print(f"  → Output dir   : {output_dir}")

    # ── Load zones.json ───────────────────────────────────────
    print("\n  → Loading zones.json...")

    if not os.path.exists(ZONES_FILE):
        print(f"  ✖ ERROR: {ZONES_FILE} not found!")
        return None

    with open(ZONES_FILE, "r") as f:
        zones = json.load(f)

    print(f"  ✔ Loaded zones for {len(zones)} pages")

    # ── Load cartoon face ─────────────────────────────────────
    print("  → Loading cartoon face...")

    if not os.path.exists(cartoon_face_path):
        print(f"  ✖ ERROR: Cartoon face not found at {cartoon_face_path}")
        return None

    cartoon_face = Image.open(cartoon_face_path).convert("RGBA")
    print(f"  ✔ Cartoon face loaded: {cartoon_face.size}")

    # Re-crop to consistent framing — the AI returns inconsistent
    # head/neck framing per photo (tight head vs head+long neck).
    # Without this, TARGET_H scaling below standardizes the outer
    # box but not where the actual face sits within it, causing
    # per-image size/position drift.
    cartoon_face = recrop_cartoon_face(cartoon_face)

    # Trim transparent padding
    bbox = cartoon_face.getbbox()
    if bbox:
        cartoon_face = cartoon_face.crop(bbox)
        print(f"  ✔ Trimmed to content: {cartoon_face.size}")

    # Standardize height
    TARGET_H = 600
    face_w, face_h = cartoon_face.size
    scale = TARGET_H / face_h
    new_w = int(face_w * scale)
    face_scaled = cartoon_face.resize((new_w, TARGET_H), Image.LANCZOS)

    # Place on fixed square canvas — CENTERED
    # This corrects any left/right offset the AI introduced
    canvas_size = 600
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - TARGET_H) // 2
    canvas.paste(face_scaled, (offset_x, offset_y), face_scaled)
    cartoon_face = canvas
    print(f"  ✔ Centered on {canvas_size}x{canvas_size} canvas")

    # NOTE: do NOT re-trim to bbox here — that would undo the
    # centering we just did and make face size/position vary
    # per photo (inconsistent hair/tilt/ears in the trimmed box).
    # Keep the full padded, centered canvas going into the page loop.

    # ── Create output directory ───────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # ── Define page order ─────────────────────────────────────
    page_order = (
        ["cover_front"] +
        [f"page_{chr(i)}" for i in range(65, 91)] +
        ["cover_back"]
    )

    composed_pages = []

    # ── Loop through all pages ────────────────────────────────
    print(f"\n  → Composing {len(page_order)} pages...\n")

    for page_num, page_key in enumerate(page_order, start=1):

        print(f"  [{page_num:02d}/28] Processing {page_key}...")

        if page_key not in zones:
            print(f"         ⚠ No zone config found for {page_key} — skipping")
            continue

        template_path = os.path.join(TEMPLATES_DIR, f"{page_key}.png")
        if not os.path.exists(template_path):
            print(f"         ⚠ Template not found: {template_path} — skipping")
            continue

        # Open template as RGBA
        template              = Image.open(template_path).convert("RGBA")
        img_width, img_height = template.size

        # Get zone config for this page
        zone      = zones[page_key]
        face_zone = zone.get("face_zone")

        # ── Paste cartoon face ────────────────────────────────
        if face_zone:
            fz_x = face_zone["x"]
            fz_y = face_zone["y"]
            fz_w = face_zone["w"]
            fz_h = face_zone["h"]

            # Scale standardized face to fit zone maintaining
            # aspect ratio — same math every time since the
            # input face is now always STANDARD_FACE_SIZE square
            face_w, face_h = cartoon_face.size
            scale  = min(fz_w / face_w, fz_h / face_h)
            new_w  = max(1, int(face_w * scale))
            new_h  = max(1, int(face_h * scale))

            face_resized = cartoon_face.resize((new_w, new_h), Image.LANCZOS)
            face_rgba    = face_resized.convert("RGBA")

            # Align face to top of zone, centered horizontally
            # (top-align prevents bottom of face being cut off)
            center_x = fz_x + (fz_w - new_w) // 2
            center_y = fz_y + (fz_h - new_h) // 2

            # Paste using the face's own transparency as the mask
            template.paste(face_rgba, (center_x, center_y), face_rgba)

            print(f"         ✔ Face pasted at ({center_x}, {center_y}) "
                  f"size {new_w}x{new_h}")

        # ── Save composed page ────────────────────────────────
        final_page  = template.convert("RGB")
        output_path = os.path.join(output_dir, f"{page_num:02d}_{page_key}.png")
        final_page.save(output_path, format="PNG", dpi=(300, 300))

        composed_pages.append(output_path)
        print(f"         ✔ Saved: {output_path}")

    print(f"\n  ✔ All {len(composed_pages)} pages composed!")
    print("  ✔ compositor.py complete!\n")

    return composed_pages


# ── Run directly for testing ──────────────────────────────────
if __name__ == "__main__":

    import glob

    # Find most recent cartoon face
    candidates = sorted(
        glob.glob("output/*/face_cartoon.png") +
        glob.glob("output/*/face_ready.png"),
        key=os.path.getmtime,
        reverse=True
    )

    if not candidates:
        print("❌ No face images found. Run main.py first.")
        exit(1)

    test_face  = candidates[0]
    order_dir  = os.path.dirname(test_face)
    child_name = os.path.basename(order_dir).split("_")[0]
    pages_dir  = os.path.join(order_dir, "pages")

    print(f"  → Using face : {test_face}")
    print(f"  → Child name : {child_name}")

    pages = compose_all_pages(test_face, child_name, pages_dir)

    if pages:
        print(f"✅ Success! {len(pages)} pages saved to {pages_dir}")
    else:
        print("❌ Something went wrong. Check errors above.")