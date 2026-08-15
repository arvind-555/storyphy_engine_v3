# ============================================================
# compositor.py
# STORYPHY — Step 3: Compose Each Page
# ============================================================
# This script takes the cartoon face + zones.json and:
#   1. Loops through all 28 pages
#   2. Opens each template
#   3. Pastes the cartoon face at the correct position
#   4. Saves each composed page to output/pages/
#
# ── How positioning works ──────────────────────────────────
# No face detection and NO MEASUREMENT runs here — deliberately.
#
# cartoonify.py guarantees a fixed contract for face_cartoon.png:
#   • Canvas is exactly FACE_CANVAS_SIZE square
#   • Subject height is exactly FACE_SUBJECT_PCT of the canvas
#   • Neck bottom is flush at the canvas bottom edge
#   • Face centre is at canvas mid-x (eye-midpoint anchored)
#
# So this file only has to map two KNOWN canvas points onto each
# page's neck_anchor. Same arithmetic every time, for every child.
#
# ── Why the old trim was removed ───────────────────────────
# This file used to run getbbox() and crop the face before pasting.
# That re-measured what cartoonify.py had already standardized, and
# reintroduced the exact variance the standardization existed to
# remove:
#
#   1. Plain getbbox() uses alpha > 0, not our threshold of 25. Since
#      cutouts now come from rembg (gpt-image-2 has no transparent
#      background option), faint feathered pixels around hair shift
#      the crop unpredictably.
#   2. Centering on the trimmed WIDTH meant asymmetric hair — a
#      side-swept fringe — pushed the face off-centre.
#   3. Scaling on the trimmed size meant a child with voluminous
#      hair got a visibly SMALLER face than one with flat hair.
#
# Trust the contract; don't re-derive it.
# ============================================================

print("▶ Running compositor.py — Composing All Pages")

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from cartoonify import REFERENCE_FACE_CX

# ── Configuration ───────────────────────────────────────────
TEMPLATES_DIR = "templates"
ZONES_FILE    = "config/zones.json"

# ── face_cartoon.png contract (set by cartoonify.py) ────────
# These MUST stay in sync with cartoonify.py. If you change the
# canvas size or subject height there, change them here too.
FACE_CANVAS_SIZE = 800    # must match cartoonify.py ANCHOR_CANVAS_SIZE
FACE_SUBJECT_PCT = 0.68   # must match cartoonify.py TARGET_SUBJECT_HEIGHT_PCT


# Must match cartoonify.py's REFERENCE_FACE_CX. Measured from the
# reference face that composites correctly on all 28 templates —
# it is NOT canvas mid-x. If you recalibrate there, change it here.
REFERENCE_FACE_CX = 385

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

    # Sanity check the contract rather than silently misplacing 28
    # pages. A wrong canvas size here means cartoonify.py didn't run,
    # or an old/hand-edited file was passed in.
    if cartoon_face.size != (FACE_CANVAS_SIZE, FACE_CANVAS_SIZE):
        print(f"  ⚠ WARNING: expected {FACE_CANVAS_SIZE}x{FACE_CANVAS_SIZE}, "
              f"got {cartoon_face.width}x{cartoon_face.height}")
        print("    Placement assumes the cartoonify.py contract. If this "
              "file didn't come from cartoonify.py, positions will be off.")

    # NOTE: no getbbox()/crop here. See the header comment.

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

            # ── SIZE ──────────────────────────────────────────
            # Scale is derived from zone height against the KNOWN
            # subject height, not the measured image.
            #
            # Width is deliberately ignored. The old min(w_fit, h_fit)
            # let hair volume drive the scale, so faces came out
            # different sizes for different children. Height-only
            # scaling keeps every child's face identically sized.
            subject_h = FACE_CANVAS_SIZE * FACE_SUBJECT_PCT
            scale     = fz_h / subject_h

            new_w = max(1, int(cartoon_face.width  * scale))
            new_h = max(1, int(cartoon_face.height * scale))

            face_rgba = cartoon_face.resize((new_w, new_h), Image.LANCZOS)

            # ── POSITION ──────────────────────────────────────
            # Two known canvas points, scaled, mapped onto neck_anchor:
            #   face centre x = canvas mid-x
            #   neck bottom y = canvas bottom edge
            # Nothing is measured, so nothing varies per child.
            neck = zone.get("neck_anchor")
            if neck:
                canvas_face_cx = int(REFERENCE_FACE_CX * scale)
                canvas_neck_y  = int(FACE_CANVAS_SIZE * scale)

                center_x = neck["x"] - canvas_face_cx
                center_y = neck["y"] - canvas_neck_y

                # No clamping. A negative y means the zone genuinely
                # doesn't fit and zones.json needs fixing — clamping
                # would just hide that by shoving the face downward.
                if center_y < 0:
                    print(f"         ⚠ Face overflows top edge (y={center_y}) "
                          f"— check neck_anchor/face_zone for {page_key}")
            else:
                # Fallback for any page where neck_anchor isn't set yet
                center_x = fz_x + (fz_w - new_w) // 2
                center_y = fz_y + (fz_h - new_h) // 2
                print("         ⚠ No neck_anchor — using face_zone centre")

            # Paste using the face's own transparency as the mask
            template.paste(face_rgba, (center_x, center_y), face_rgba)

            print(f"         ✔ Face pasted at ({center_x}, {center_y}) "
                  f"size {new_w}x{new_h} (scale {scale:.3f})")

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

    # Only face_cartoon.png is valid input now — face_ready.png does
    # not satisfy the placement contract (different subject framing),
    # so it's no longer accepted as a stand-in.
    candidates = sorted(
        glob.glob("output/*/face_cartoon.png"),
        key=os.path.getmtime,
        reverse=True
    )

    if not candidates:
        print("❌ No face_cartoon.png found. Run cartoonify.py first.")
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