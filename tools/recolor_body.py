# ============================================================
# recolor_body.py
# STORYPHY — Recolor template body skin to match child
# ============================================================
# Samples the child's skin tone from face_cartoon.png, then
# shifts skin-colored pixels inside each page's body_zone
# toward that tone. Keeps shading (works in HSV hue/sat only).
# ============================================================

print("> Running recolor_body.py")

import cv2
import json
import os
import glob
import numpy as np
from PIL import Image

ZONES_FILE = "config/zones.json"


# ── Sample average skin tone from the cartoon face ──────────
def get_skin_tone(face_path):
    """Returns average (H, S, V) of the face's skin pixels."""
    img = Image.open(face_path).convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Skin range in HSV — covers light to deep skin tones
    lower = np.array([0, 20, 60],   dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    skin_pixels = hsv[mask > 0]
    if len(skin_pixels) == 0:
        print("  WARNING: no skin pixels found in face")
        return None

    avg = skin_pixels.mean(axis=0)
    print(f"  Child skin tone (HSV): "
          f"H={avg[0]:.0f} S={avg[1]:.0f} V={avg[2]:.0f}")
    return avg


# ── Recolor body skin inside a zone ──────────────────────────
def recolor_page(page_path, body_zone, target_hsv):
    """Shifts skin pixels in body_zone toward target skin tone."""
    img = Image.open(page_path).convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    bx = body_zone["x"]
    by = body_zone["y"]
    bw = body_zone["w"]
    bh = body_zone["h"]

    # Crop the body region only
    region = hsv[by:by+bh, bx:bx+bw]

    # Find skin pixels in the region
    region_u8 = region.astype(np.uint8)
    lower = np.array([0, 20, 60],   dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(region_u8, lower, upper)

    skin = skin_mask > 0
    if skin.sum() == 0:
        print(f"  no skin pixels in body zone — skipped")
        return

    # Shift hue and saturation toward target, keep V (shading)
    region[skin, 0] = target_hsv[0]                       # hue
    region[skin, 1] = np.clip(target_hsv[1], 0, 255)      # saturation

    # Write region back
    hsv[by:by+bh, bx:bx+bw] = region

    # Convert back to RGB and save
    out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(out_rgb).save(page_path, format="PNG", dpi=(300, 300))
    print(f"  recolored body skin ({skin.sum()} pixels)")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":

    with open(ZONES_FILE, "r") as f:
        zones = json.load(f)

    # Find most recent order folder
    orders = sorted(
        glob.glob("output/*/pages/"),
        key=os.path.getmtime,
        reverse=True
    )
    if not orders:
        print("No order folders found. Run main.py first.")
        exit(1)

    print("\n  Available orders:")
    for i, folder in enumerate(orders):
        name = folder.split(os.sep)[-3]
        print(f"  [{i+1}] {name}")

    print("\n  Select order (or ENTER for most recent):")
    choice = input("  Choice: ").strip()
    pages_dir = orders[0] if not choice else orders[int(choice) - 1]

    order_dir = os.path.dirname(os.path.dirname(pages_dir))
    face_path = os.path.join(order_dir, "face_cartoon.png")

    if not os.path.exists(face_path):
        print(f"  ERROR: face_cartoon.png not found in {order_dir}")
        exit(1)

    # Get the child's skin tone
    target = get_skin_tone(face_path)
    if target is None:
        exit(1)

    # Recolor every page that has a body_zone
    page_files = sorted(glob.glob(os.path.join(pages_dir, "*.png")))
    print(f"\n  Processing {len(page_files)} pages...\n")

    for page_path in page_files:
        filename = os.path.basename(page_path)
        parts    = filename.replace(".png", "").split("_", 1)
        page_key = parts[1] if len(parts) > 1 else parts[0]

        body_zone = zones.get(page_key, {}).get("body_zone")
        print(f"  [{filename}]")
        if body_zone:
            recolor_page(page_path, body_zone, target)
        else:
            print(f"  no body_zone — skipped")

    print("\n  Done! Body skin recolored.")
    print("  Rebuild PDF:  python src/pdf_builder.py")