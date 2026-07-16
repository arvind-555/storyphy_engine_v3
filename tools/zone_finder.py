# ============================================================
# zone_finder.py
# STORYPHY — Helper Tool: Find Face Zone Coordinates
# ============================================================
# This tool helps you visually set the face placement zone
# for each template page.
#
# How to use:
#   1. Run this script
#   2. It opens your template image
#   3. Click and drag to draw the face zone rectangle
#   4. Press ENTER to confirm, R to redraw, Q to quit
#   5. It prints the coordinates to copy into zones.json
# ============================================================

print("▶ Running zone_finder.py — Interactive Zone Coordinate Finder")

import cv2
import json
import os
import glob

# ── Configuration ───────────────────────────────────────────
TEMPLATES_DIR = "templates"
OUTPUT_FILE   = "config/zones.json"

# ── Global variables for mouse drawing ──────────────────────
drawing    = False
start_x    = start_y = 0
end_x      = end_y   = 0
rect_drawn = False

def mouse_callback(event, x, y, flags, param):
    """Handles mouse click and drag to draw rectangle."""
    global drawing, start_x, start_y, end_x, end_y, rect_drawn

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing
        drawing  = True
        start_x  = x
        start_y  = y
        end_x    = x
        end_y    = y
        rect_drawn = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x = x
            end_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing
        drawing    = False
        end_x      = x
        end_y      = y
        rect_drawn = True


def get_zone_for_template(template_path, page_key, existing_zone=None, face_path=None):
    """
    Opens a template and lets user draw the face zone.
    Shows live preview of face placed in the drawn zone.
    Returns the zone coordinates as a dict.
    """

    print(f"\n  → Opening template: {template_path}")

    # Load the template image
    image = cv2.imread(template_path)
    if image is None:
        print(f"  ✖ Could not load {template_path}")
        return None

    img_h, img_w = image.shape[:2]

    # ── Load face image for live preview ─────────────────────
    face_img      = None
    face_img_bgra = None

    if face_path and os.path.exists(face_path):
        # Load face with PIL to handle transparency
        from PIL import Image as PILImage
        import numpy as np

        pil_face = PILImage.open(face_path).convert("RGBA")
        face_img_bgra = cv2.cvtColor(
            np.array(pil_face), cv2.COLOR_RGBA2BGRA
        )
        print(f"  → Face loaded for preview: {face_path}")
    else:
        print("  ⚠ No face image found — showing zone outline only")
        print("    Run main.py once first to generate face_ready.png")

    # Scale down for display if image is large
    display_scale = min(800 / img_w, 800 / img_h, 1.0)
    display_w     = int(img_w * display_scale)
    display_h     = int(img_h * display_scale)

    window_name = f"Zone Finder — {page_key}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)
    cv2.setMouseCallback(window_name, mouse_callback)

    global start_x, start_y, end_x, end_y, rect_drawn
    rect_drawn = False

    # Pre-load existing zone if available
    if existing_zone and "face_zone" in existing_zone:
        fz = existing_zone["face_zone"]
        start_x    = int(fz["x"] * display_scale)
        start_y    = int(fz["y"] * display_scale)
        end_x      = int((fz["x"] + fz["w"]) * display_scale)
        end_y      = int((fz["y"] + fz["h"]) * display_scale)
        rect_drawn = True

    print(f"  → Image: {img_w}x{img_h} | Scale: {display_scale:.2f}")
    print("  → DRAG to draw face zone")
    print("  → ENTER to confirm | R to redraw | Q to skip")

    while True:
        # Start with a fresh copy of the template
        display = cv2.resize(image, (display_w, display_h))

        # ── Draw instructions bar at top ──────────────────────
        cv2.rectangle(display, (0, 0), (display_w, 58), (30, 30, 30), -1)
        cv2.putText(display, f"{page_key} — Drag to set face zone",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        cv2.putText(display, "ENTER = confirm  |  R = redraw  |  Q = skip",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1)

        # ── Draw zone rectangle + live face preview ───────────
        if rect_drawn or drawing:
            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            zone_w = x2 - x1
            zone_h = y2 - y1

            if zone_w > 10 and zone_h > 10:

                # ── Live face preview inside zone ─────────────
                if face_img_bgra is not None:
                    import numpy as np

                    # Resize face to fit zone maintaining aspect ratio
                    face_h_orig, face_w_orig = face_img_bgra.shape[:2]
                    scale_face = min(zone_w / face_w_orig,
                                     zone_h / face_h_orig)
                    new_fw = int(face_w_orig * scale_face)
                    new_fh = int(face_h_orig * scale_face)

                    # Center face in zone
                    offset_x = x1 + (zone_w - new_fw) // 2
                    offset_y = y1 + (zone_h - new_fh) // 2

                    # Make sure face fits within display bounds
                    if (offset_x >= 0 and offset_y >= 0 and
                        offset_x + new_fw <= display_w and
                        offset_y + new_fh <= display_h and
                        new_fw > 0 and new_fh > 0):

                        # Resize face
                        face_resized = cv2.resize(
                            face_img_bgra, (new_fw, new_fh)
                        )

                        # Extract BGR and alpha from face
                        face_bgr   = face_resized[:, :, :3]
                        face_alpha = face_resized[:, :, 3:4] / 255.0

                        # Get the region on display where face goes
                        roi = display[
                            offset_y:offset_y + new_fh,
                            offset_x:offset_x + new_fw
                        ]

                        # Blend face onto template using alpha
                        blended = (face_bgr * face_alpha +
                                   roi * (1 - face_alpha)).astype('uint8')

                        # Put blended result back onto display
                        display[
                            offset_y:offset_y + new_fh,
                            offset_x:offset_x + new_fw
                        ] = blended

                # ── Draw zone border ───────────────────────────
                cv2.rectangle(display,
                               (x1, y1), (x2, y2),
                               (0, 255, 0), 2)

                # ── Show zone dimensions ───────────────────────
                actual_w = int(zone_w / display_scale)
                actual_h = int(zone_h / display_scale)
                cv2.putText(display,
                             f"{actual_w} x {actual_h} px",
                             (x1 + 4, y2 - 8),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 255, 0), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and rect_drawn:
            # ENTER — confirm zone
            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            # Convert back to actual image coordinates
            actual_x = int(x1 / display_scale)
            actual_y = int(y1 / display_scale)
            actual_w = int((x2 - x1) / display_scale)
            actual_h = int((y2 - y1) / display_scale)

            print(f"  ✔ Zone confirmed: x={actual_x}, y={actual_y}, "
                  f"w={actual_w}, h={actual_h}")
            cv2.destroyWindow(window_name)
            return {
                "face_zone": {
                    "x": actual_x,
                    "y": actual_y,
                    "w": actual_w,
                    "h": actual_h
                }
            }

        elif key == ord('r'):
            rect_drawn = False
            print("  → Rectangle reset. Draw again.")

        elif key == ord('q'):
            print(f"  → Skipped {page_key}")
            cv2.destroyWindow(window_name)
            return None


def run_zone_finder():
    """
    Loops through all templates and collects face zones.
    Saves everything to zones.json.
    """

    print("\n════════════════════════════════════════")
    print("  STORYPHY — Zone Finder Tool")
    print("════════════════════════════════════════")

    # Load existing zones.json if it exists (so we don't lose progress)
    zones = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            content = f.read().strip()
            if content:
                zones = json.loads(content)
                print(f"  → Loaded existing zones.json ({len(zones)} pages already configured)")
            else:
                print("  → zones.json was empty, starting fresh")
    else:
        print("  → Starting fresh zones.json")

    # ── Ask ONCE which face image to use for preview ─────────
    all_faces = sorted(
        glob.glob("output/*/face_cartoon.png") +
        glob.glob("output/*/face_ready.png"),
        key=os.path.getmtime,
        reverse=True
    )

    face_path = None

    if all_faces:
        print("\n  ── Face image for preview ──────────────────")
        print("  Available face images:")
        for i, path in enumerate(all_faces):
            folder = os.path.basename(os.path.dirname(path))
            file   = os.path.basename(path)
            print(f"  [{i+1}] {folder} — {file}")

        print("\n  Options:")
        print("  → Type a NUMBER to select from list")
        print("  → Type a FILE PATH to use a custom image")
        print("  → Press ENTER to use most recent")
        choice = input("  Choice: ").strip().strip('"').strip("'")

        if not choice:
            face_path = all_faces[0]
            print(f"  → Using most recent: {face_path}")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_faces):
                face_path = all_faces[idx]
                print(f"  → Using: {face_path}")
            else:
                print("  ⚠ Invalid number — using most recent")
                face_path = all_faces[0]
        else:
            if os.path.exists(choice):
                face_path = choice
                print(f"  → Using custom image: {face_path}")
            else:
                print(f"  ⚠ File not found — using most recent")
                face_path = all_faces[0]
    else:
        print("\n  ⚠ No face images found in output folders.")
        print("  Enter path to a face image (or press ENTER to skip):")
        manual = input("  Path: ").strip().strip('"').strip("'")
        if manual and os.path.exists(manual):
            face_path = manual

    # Define the order of pages to process
    page_order = ["cover_front"] + [f"page_{chr(i)}" for i in range(65, 91)] + ["cover_back"]

    print(f"\n  → Found {len(page_order)} pages to configure")
    print("  → You can press Q to skip any page and come back later\n")

    for page_key in page_order:
        template_path = os.path.join(TEMPLATES_DIR, f"{page_key}.png")

        if not os.path.exists(template_path):
            print(f"  ⚠ Template not found: {template_path} — skipping")
            continue

        # Skip if already configured — ask if they want to redo
        if page_key in zones:
            print(f"\n  → {page_key} already configured.")
            redo = input(f"     Redo this page? (y/n): ").strip().lower()
            if redo != "y":
                print(f"  → Skipping {page_key}")
                continue
            else:
                print(f"  → Redoing {page_key}...")

        # Get zone for this page with live preview
        zone = get_zone_for_template(
            template_path, page_key,
            zones.get(page_key),
            face_path
        )

        if zone:
            # Merge into existing entry — do NOT overwrite the whole
            # page dict, or previously-saved rhyme/text_zone data
            # gets wiped out whenever a face zone is redone.
            if page_key not in zones:
                zones[page_key] = {}
            zones[page_key]["face_zone"] = zone["face_zone"]

            # Save after every page so progress is never lost
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(zones, f, indent=2)
            print(f"  ✔ Saved to zones.json ({len(zones)} pages done)")

    print("\n════════════════════════════════════════")
    print(f"  ✔ Zone Finder complete! {len(zones)} pages configured.")
    print(f"  ✔ Saved to: {OUTPUT_FILE}")
    print("════════════════════════════════════════\n")

def run_cloud_zone_finder():
    """
    Separate pass to set cloud/text zone for each page.
    Saves as 'text_zone' in zones.json.
    """

    print("\n════════════════════════════════════════")
    print("  STORYPHY — Cloud Text Zone Finder")
    print("════════════════════════════════════════")
    print("  Draw a rectangle AROUND the cloud shape")
    print("  on each template to set where text goes.")
    print("════════════════════════════════════════\n")

    # Load existing zones
    zones = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            content = f.read().strip()
            if content:
                zones = json.loads(content)

    page_order = [f"page_{chr(i)}" for i in range(65, 91)]

    for page_key in page_order:
        template_path = os.path.join(TEMPLATES_DIR, f"{page_key}.png")

        if not os.path.exists(template_path):
            print(f"  ⚠ Template not found: {template_path} — skipping")
            continue

        # Check if already configured
        if page_key in zones and "text_zone" in zones[page_key]:
            print(f"\n  → {page_key} text zone already set.")
            redo = input("     Redo? (y/n): ").strip().lower()
            if redo != "y":
                print(f"  → Skipping {page_key}")
                continue

        print(f"\n  → Setting cloud zone for: {page_key}")
        print("  → Draw rectangle AROUND the cloud shape")

        # Use zone finder without face preview for text zones
        zone = get_zone_for_template(
            template_path,
            f"{page_key} — DRAW AROUND CLOUD",
            None,
            None
        )

        if zone and "face_zone" in zone:
            if page_key not in zones:
                zones[page_key] = {}

            # Save as text_zone
            zones[page_key]["text_zone"] = zone["face_zone"]

            # Save after every page
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(zones, f, indent=2)

            tz = zone["face_zone"]
            print(f"  ✔ Cloud zone saved: x={tz['x']}, y={tz['y']}, "
                  f"w={tz['w']}, h={tz['h']}")

    print("\n════════════════════════════════════════")
    print("  ✔ Cloud zones configured!")
    print("  → Now run: python tools/add_text.py")
    print("════════════════════════════════════════\n")

if __name__ == "__main__":

    print("\n========================================")
    print("  STORYPHY — Zone Finder")
    print("========================================")
    print("  [1] Face placement zones")
    print("  [2] Cloud/text zones")
    print("========================================")

    mode = input("  Choice (1/2): ").strip()

    if mode == "1":
        run_zone_finder()
    elif mode == "2":
        run_cloud_zone_finder()
    else:
        print("  Invalid choice — enter 1,or 2")