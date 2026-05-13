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
TEMPLATES_DIR = "templates"
ZONES_FILE    = "config/zones.json"


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
        template       = Image.open(template_path).convert("RGBA")
        img_width, img_height = template.size

        # Get zone config
        zone      = zones[page_key]
        face_zone = zone.get("face_zone")

        # ── Paste cartoon face ────────────────────────────────
        if face_zone:
            fz_x = face_zone["x"]
            fz_y = face_zone["y"]
            fz_w = face_zone["w"]
            fz_h = face_zone["h"]

            # Normalize face size maintaining aspect ratio
            face_w, face_h = cartoon_face.size
            scale  = min(fz_w / face_w, fz_h / face_h)
            new_w  = int(face_w * scale)
            new_h  = int(face_h * scale)

            face_resized = cartoon_face.resize((new_w, new_h), Image.LANCZOS)

            # ── Apply oval mask ───────────────────────────────
            oval_mask  = Image.new("L", (new_w, new_h), 0)
            mask_draw  = ImageDraw.Draw(oval_mask)
            padding    = int(new_w * 0.05)
            mask_draw.ellipse(
                [padding, padding, new_w - padding, new_h - padding],
                fill=255
            )
            oval_mask = oval_mask.filter(ImageFilter.GaussianBlur(radius=8))

            # Combine original alpha with oval mask
            face_rgba      = face_resized.convert("RGBA")
            r, g, b, orig_alpha = face_rgba.split()
            orig_array     = np.array(orig_alpha)
            oval_array     = np.array(oval_mask)
            combined_array = np.minimum(orig_array, oval_array)
            combined_alpha = Image.fromarray(combined_array)
            face_rgba.putalpha(combined_alpha)

            # Center face in zone
            center_x = fz_x + (fz_w - new_w) // 2
            center_y = fz_y + (fz_h - new_h) // 2

            template.paste(face_rgba, (center_x, center_y), face_rgba)
            print(f"         ✔ Face pasted at ({center_x}, {center_y})")

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