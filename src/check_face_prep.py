# ============================================================
# check_face_prep.py
# STORYPHY — Quick visual QA for face_prep.py output
# ============================================================
# Draws a red line at the expected bottom-anchor Y position and
# a center guideline, so misalignment across different photos
# is obvious at a glance instead of eyeballing pixel offsets.
#
# Usage: python check_face_prep.py path/to/face_ready.png
# ============================================================

import sys
from PIL import Image, ImageDraw

STANDARD_SIZE = 800
BOTTOM_MARGIN = 0.05  # must match face_prep.py


def check(path):
    img = Image.open(path).convert("RGBA")
    if img.size != (STANDARD_SIZE, STANDARD_SIZE):
        print(f"⚠ WARNING: expected {STANDARD_SIZE}x{STANDARD_SIZE}, got {img.size}")

    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)

    # Expected bottom-anchor line (where subject's bottom edge should sit)
    anchor_y = STANDARD_SIZE - int(STANDARD_SIZE * BOTTOM_MARGIN)
    draw.line([(0, anchor_y), (STANDARD_SIZE, anchor_y)], fill=(255, 0, 0, 255), width=2)

    # Center vertical guideline
    draw.line([(STANDARD_SIZE // 2, 0), (STANDARD_SIZE // 2, STANDARD_SIZE)],
              fill=(0, 150, 255, 180), width=1)

    # Find actual bounding box of non-transparent pixels
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        draw.rectangle(bbox, outline=(0, 255, 0, 255), width=2)
        print(f"→ Subject bounding box: {bbox}")
        print(f"→ Subject bottom edge at y={bbox[3]} (target line at y={anchor_y})")
        gap = abs(bbox[3] - anchor_y)
        if gap > 10:
            print(f"⚠ Bottom edge is {gap}px off target — check for clipping or bg-removal issue")
        else:
            print("✔ Bottom edge aligned with target anchor")
    else:
        print("✖ No non-transparent pixels found — image may be blank")

    out_path = path.rsplit(".", 1)[0] + "_check.png"
    canvas.save(out_path)
    print(f"✔ Saved annotated check image to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_face_prep.py path/to/face_ready.png")
        sys.exit(1)
    check(sys.argv[1])