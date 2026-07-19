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


# ── Global variables for point selection (neck anchor) ──────
point_x, point_y = 0, 0
point_set = False
hover_x, hover_y = 0, 0

def point_callback(event, x, y, flags, param):
    """Handles mouse move (live preview) + click to confirm a point."""
    global point_x, point_y, point_set, hover_x, hover_y
    if event == cv2.EVENT_MOUSEMOVE:
        hover_x, hover_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        point_x, point_y = x, y
        point_set = True


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
        # Trim transparent padding — MUST match compositor.py's
        # trim, or preview position won't match actual render
        bbox = pil_face.getbbox()
        if bbox:
            pil_face = pil_face.crop(bbox)
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


def locate_chin_for_preview(face_img_bgra):
    """
    Same chin-detection logic as compositor.py's locate_chin(),
    adapted for the zone_finder preview image (BGRA numpy array).
    Returns (eye_cx, chin_y) in the ORIGINAL face image's pixel
    coordinates (before any preview scaling).
    Falls back to (horizontal center, bottom edge) if no face found.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    h, w = face_img_bgra.shape[:2]
    rgb_array = cv2.cvtColor(face_img_bgra, cv2.COLOR_BGRA2RGB)

    model_path = "config/blaze_face_short_range.tflite"
    if not os.path.exists(model_path):
        return w / 2, h

    base_options = BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=0.4
    )
    with mp_vision.FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        results = detector.detect(mp_image)

        if not results.detections:
            return w / 2, h

        kp = results.detections[0].keypoints
        rx, ry = kp[0].x * w, kp[0].y * h
        lx, ly = kp[1].x * w, kp[1].y * h
        eye_dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
        eye_cx, eye_cy = (lx + rx) / 2, (ly + ry) / 2

    # Must match CHIN_OFFSET_RATIO in compositor.py
    chin_y = eye_cy + (eye_dist * 2.1)
    return eye_cx, chin_y


def get_point_for_template(template_path, page_key, face_path=None):
    """
    Opens a template and lets user click ONE point —
    used for marking the neck anchor (where the chin/neck
    of the pasted face should land).

    If face_path is given, shows a live preview of the child's
    cartoon face following the cursor, bottom-aligned to the
    current point, so you can see exactly how it'll sit.
    """
    global point_x, point_y, point_set, hover_x, hover_y
    point_set = False

    image = cv2.imread(template_path)
    if image is None:
        print(f"  ✖ Could not load {template_path}")
        return None

    img_h, img_w = image.shape[:2]
    display_scale = min(800 / img_w, 800 / img_h, 1.0)
    display_w = int(img_w * display_scale)
    display_h = int(img_h * display_scale)
    hover_x, hover_y = display_w // 2, display_h // 2

    # ── Load face image for live preview ──────────────────────
    face_img_bgra = None
    if face_path and os.path.exists(face_path):
        from PIL import Image as PILImage
        import numpy as np
        pil_face = PILImage.open(face_path).convert("RGBA")
        bbox = pil_face.getbbox()
        if bbox:
            pil_face = pil_face.crop(bbox)
        face_img_bgra = cv2.cvtColor(np.array(pil_face), cv2.COLOR_RGBA2BGRA)

    chin_eye_cx = chin_chin_y = None
    if face_img_bgra is not None:
        print("  → Detecting face for accurate preview...")
        chin_eye_cx, chin_chin_y = locate_chin_for_preview(face_img_bgra)

    window_name = f"Neck Anchor — {page_key}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)
    cv2.setMouseCallback(window_name, point_callback)

    print(f"\n  → {page_key}: MOVE mouse to preview, CLICK to place neck/chin point")
    print("  → ENTER to confirm | Q to skip")

    # Preview face width — a fixed fraction of the template width
    preview_w = int(display_w * 0.22)

    while True:
        display = cv2.resize(image, (display_w, display_h))
        cv2.rectangle(display, (0, 0), (display_w, 40), (30, 30, 30), -1)
        cv2.putText(display, f"{page_key} — Click neck/chin point",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        preview_x = point_x if point_set else hover_x
        preview_y = point_y if point_set else hover_y

        # ── Live face preview, bottom-center aligned to point ──
        if face_img_bgra is not None:
            fh_orig, fw_orig = face_img_bgra.shape[:2]
            scale_face = preview_w / fw_orig
            new_fw = preview_w
            new_fh = int(fh_orig * scale_face)

            offset_x = preview_x - new_fw // 2
            if chin_chin_y is not None:
                scale_ratio = new_fh / fh_orig
                offset_x = int(preview_x - chin_eye_cx * scale_ratio)
                offset_y = int(preview_y - chin_chin_y * scale_ratio)
            else:
                offset_y = preview_y - new_fh  # fallback: bottom of face at the point

            if (offset_x >= 0 and offset_y >= 0 and
                offset_x + new_fw <= display_w and
                offset_y + new_fh <= display_h and
                new_fw > 0 and new_fh > 0):

                face_resized = cv2.resize(face_img_bgra, (new_fw, new_fh))
                face_bgr = face_resized[:, :, :3]
                face_alpha = face_resized[:, :, 3:4] / 255.0

                roi = display[offset_y:offset_y + new_fh, offset_x:offset_x + new_fw]
                blended = (face_bgr * face_alpha + roi * (1 - face_alpha)).astype('uint8')
                display[offset_y:offset_y + new_fh, offset_x:offset_x + new_fw] = blended

        # ── Mark the point itself ───────────────────────────────
        color = (0, 255, 0) if point_set else (0, 200, 255)
        cv2.circle(display, (preview_x, preview_y), 6, color, -1)
        cv2.circle(display, (preview_x, preview_y), 10, color, 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and point_set:
            actual_x = int(point_x / display_scale)
            actual_y = int(point_y / display_scale)
            print(f"  ✔ Neck anchor confirmed: x={actual_x}, y={actual_y}")
            cv2.destroyWindow(window_name)
            return {"x": actual_x, "y": actual_y}

        elif key == ord('q'):
            print(f"  → Skipped {page_key}")
            cv2.destroyWindow(window_name)
            return None


def get_zone_and_anchor_for_template(template_path, page_key, existing_zone=None, face_path=None):
    """
    Combined tool: PHASE 1 — drag to set face SIZE (face_zone).
    PHASE 2 — click to set the neck/chin POSITION (neck_anchor).
    Both are set in one pass per page and saved together.

    The preview in Phase 2 uses the SAME sizing as Phase 1's
    locked zone, and anchors the face's bottom edge to the
    clicked point — matching exactly how compositor.py places
    the face (no detection, bottom edge = neck/chin).
    """
    global start_x, start_y, end_x, end_y, rect_drawn, drawing
    global point_x, point_y, point_set, hover_x, hover_y

    image = cv2.imread(template_path)
    if image is None:
        print(f"  ✖ Could not load {template_path}")
        return None

    img_h, img_w = image.shape[:2]
    display_scale = min(800 / img_w, 800 / img_h, 1.0)
    display_w = int(img_w * display_scale)
    display_h = int(img_h * display_scale)

    face_img_bgra = None
    if face_path and os.path.exists(face_path):
        from PIL import Image as PILImage
        import numpy as np
        pil_face = PILImage.open(face_path).convert("RGBA")
        # Trim transparent padding — MUST match compositor.py's
        # trim exactly, or the point you click here won't match
        # where the face actually lands in the real output
        bbox = pil_face.getbbox()
        if bbox:
            pil_face = pil_face.crop(bbox)
        face_img_bgra = cv2.cvtColor(np.array(pil_face), cv2.COLOR_RGBA2BGRA)
    else:
        print("  ⚠ No face image found — preview will show outline only")

    window_name = f"Zone + Anchor — {page_key}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)

    rect_drawn = False
    if existing_zone and "face_zone" in existing_zone:
        fz = existing_zone["face_zone"]
        start_x = int(fz["x"] * display_scale)
        start_y = int(fz["y"] * display_scale)
        end_x   = int((fz["x"] + fz["w"]) * display_scale)
        end_y   = int((fz["y"] + fz["h"]) * display_scale)
        rect_drawn = True

    phase = 1  # 1 = draw rectangle (size), 2 = click point (position)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n  → {page_key}: PHASE 1 — DRAG to set face SIZE")
    print("  → ENTER = lock size  |  R = redraw  |  Q = skip")

    final_zone = None
    point_set = False
    hover_x, hover_y = display_w // 2, display_h // 2

    while True:
        display = cv2.resize(image, (display_w, display_h))
        cv2.rectangle(display, (0, 0), (display_w, 58), (30, 30, 30), -1)

        if phase == 1:
            cv2.putText(display, f"{page_key} — PHASE 1: Drag to set face SIZE",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, "ENTER = lock size  |  R = redraw  |  Q = skip",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            if rect_drawn or drawing:
                x1, y1 = min(start_x, end_x), min(start_y, end_y)
                x2, y2 = max(start_x, end_x), max(start_y, end_y)
                zone_w, zone_h = x2 - x1, y2 - y1

                if zone_w > 10 and zone_h > 10:
                    if face_img_bgra is not None:
                        fh_o, fw_o = face_img_bgra.shape[:2]
                        sc = min(zone_w / fw_o, zone_h / fh_o)
                        nfw, nfh = int(fw_o * sc), int(fh_o * sc)
                        ox = x1 + (zone_w - nfw) // 2
                        oy = y1 + (zone_h - nfh) // 2
                        if (ox >= 0 and oy >= 0 and ox + nfw <= display_w
                                and oy + nfh <= display_h and nfw > 0 and nfh > 0):
                            face_resized = cv2.resize(face_img_bgra, (nfw, nfh))
                            face_bgr = face_resized[:, :, :3]
                            face_alpha = face_resized[:, :, 3:4] / 255.0
                            roi = display[oy:oy + nfh, ox:ox + nfw]
                            blended = (face_bgr * face_alpha + roi * (1 - face_alpha)).astype('uint8')
                            display[oy:oy + nfh, ox:ox + nfw] = blended

                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    aw, ah = int(zone_w / display_scale), int(zone_h / display_scale)
                    cv2.putText(display, f"{aw} x {ah} px", (x1 + 4, y2 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and rect_drawn:
                x1, y1 = min(start_x, end_x), min(start_y, end_y)
                x2, y2 = max(start_x, end_x), max(start_y, end_y)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    actual_x = int(x1 / display_scale)
                    actual_y = int(y1 / display_scale)
                    actual_w = int((x2 - x1) / display_scale)
                    actual_h = int((y2 - y1) / display_scale)
                    final_zone = {"x": actual_x, "y": actual_y, "w": actual_w, "h": actual_h}
                    print(f"  ✔ Size locked: {actual_w}x{actual_h}")
                    print(f"  → {page_key}: PHASE 2 — CLICK where the neck/chin should sit")
                    print("  → ENTER = confirm  |  B = back to size  |  Q = skip")
                    phase = 2
                    point_set = False
                    cv2.setMouseCallback(window_name, point_callback)
            elif key == ord('r'):
                rect_drawn = False
                print("  → Rectangle reset. Draw again.")
            elif key == ord('q'):
                print(f"  → Skipped {page_key}")
                cv2.destroyWindow(window_name)
                return None

        else:  # phase == 2
            cv2.putText(display, f"{page_key} — PHASE 2: Click neck/chin point",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, "ENTER = confirm  |  B = back to size  |  Q = skip",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            preview_x = point_x if point_set else hover_x
            preview_y = point_y if point_set else hover_y

            zone_w = int(final_zone["w"] * display_scale)
            zone_h = int(final_zone["h"] * display_scale)

            if face_img_bgra is not None:
                fh_o, fw_o = face_img_bgra.shape[:2]
                sc = min(zone_w / fw_o, zone_h / fh_o)
                nfw, nfh = max(1, int(fw_o * sc)), max(1, int(fh_o * sc))
                ox = preview_x - nfw // 2
                oy = preview_y - nfh  # bottom edge = clicked point (matches compositor.py)

                if (ox >= 0 and oy >= 0 and ox + nfw <= display_w
                        and oy + nfh <= display_h and nfw > 0 and nfh > 0):
                    face_resized = cv2.resize(face_img_bgra, (nfw, nfh))
                    face_bgr = face_resized[:, :, :3]
                    face_alpha = face_resized[:, :, 3:4] / 255.0
                    roi = display[oy:oy + nfh, ox:ox + nfw]
                    blended = (face_bgr * face_alpha + roi * (1 - face_alpha)).astype('uint8')
                    display[oy:oy + nfh, ox:ox + nfw] = blended

            color = (0, 255, 0) if point_set else (0, 200, 255)
            cv2.circle(display, (preview_x, preview_y), 6, color, -1)
            cv2.circle(display, (preview_x, preview_y), 10, color, 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and point_set:
                actual_x = int(point_x / display_scale)
                actual_y = int(point_y / display_scale)
                print(f"  ✔ Neck anchor confirmed: x={actual_x}, y={actual_y}")
                cv2.destroyWindow(window_name)
                return {"face_zone": final_zone, "neck_anchor": {"x": actual_x, "y": actual_y}}
            elif key == ord('b'):
                phase = 1
                cv2.setMouseCallback(window_name, mouse_callback)
                print(f"  → {page_key}: back to PHASE 1 — Drag to set face SIZE")
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

def run_neck_anchor_finder():
    """
    Separate pass to mark the neck/chin anchor point per page.
    Saves as 'neck_anchor' in zones.json.
    """
    print("\n════════════════════════════════════════")
    print("  STORYPHY — Neck Anchor Finder")
    print("════════════════════════════════════════")

    zones = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            content = f.read().strip()
            if content:
                zones = json.loads(content)

    # ── Ask which face image to use for live preview ──────────
    all_faces = sorted(
        glob.glob("output/*/face_cartoon.png") +
        glob.glob("output/*/face_ready.png"),
        key=os.path.getmtime,
        reverse=True
    )

    face_path = None
    if all_faces:
        print("\n  ── Face image for preview ──────────────────")
        for i, path in enumerate(all_faces):
            folder = os.path.basename(os.path.dirname(path))
            print(f"  [{i+1}] {folder}")
        print("  → Type a NUMBER, a FILE PATH, or press ENTER for most recent")
        choice = input("  Choice: ").strip().strip('"').strip("'")

        if not choice:
            face_path = all_faces[0]
        elif choice.isdigit() and 0 <= int(choice) - 1 < len(all_faces):
            face_path = all_faces[int(choice) - 1]
        elif os.path.exists(choice):
            face_path = choice
        else:
            print("  ⚠ Not found — using most recent")
            face_path = all_faces[0]
        print(f"  → Using: {face_path}")
    else:
        print("  ⚠ No face images found — preview will show outline only")

    page_order = [f"page_{chr(i)}" for i in range(65, 91)]

    for page_key in page_order:
        template_path = os.path.join(TEMPLATES_DIR, f"{page_key}.png")
        if not os.path.exists(template_path):
            print(f"  ⚠ Template not found: {template_path} — skipping")
            continue

        if page_key in zones and "neck_anchor" in zones[page_key]:
            redo = input(f"  {page_key} already set. Redo? (y/n): ").strip().lower()
            if redo != "y":
                continue

        point = get_point_for_template(template_path, page_key, face_path)

        if point:
            if page_key not in zones:
                zones[page_key] = {}
            zones[page_key]["neck_anchor"] = point

            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(zones, f, indent=2)
            print(f"  ✔ Saved ({len(zones)} pages total)")

    print("\n  ✔ Neck anchors configured!\n")


def run_combined_finder():
    """
    Combined pass: for each page, set face SIZE (drag) then
    neck POSITION (click) in one continuous flow. Saves both
    face_zone and neck_anchor together per page.
    """
    print("\n════════════════════════════════════════")
    print("  STORYPHY — Combined Zone + Anchor Finder")
    print("════════════════════════════════════════")

    zones = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            content = f.read().strip()
            if content:
                zones = json.loads(content)

    # ── Ask which face image to use for live preview ──────────
    all_faces = sorted(
        glob.glob("output/*/face_cartoon.png") +
        glob.glob("output/*/face_ready.png"),
        key=os.path.getmtime,
        reverse=True
    )

    face_path = None
    if all_faces:
        print("\n  ── Face image for preview ──────────────────")
        for i, path in enumerate(all_faces):
            folder = os.path.basename(os.path.dirname(path))
            print(f"  [{i+1}] {folder}")
        print("  → Type a NUMBER, a FILE PATH, or press ENTER for most recent")
        choice = input("  Choice: ").strip().strip('"').strip("'")

        if not choice:
            face_path = all_faces[0]
        elif choice.isdigit() and 0 <= int(choice) - 1 < len(all_faces):
            face_path = all_faces[int(choice) - 1]
        elif os.path.exists(choice):
            face_path = choice
        else:
            print("  ⚠ Not found — using most recent")
            face_path = all_faces[0]
        print(f"  → Using: {face_path}")
    else:
        print("  ⚠ No face images found — preview will show outline only")

    page_order = (
        ["cover_front"] +
        [f"page_{chr(i)}" for i in range(65, 91)] +
        ["cover_back"]
    )

    for page_key in page_order:
        template_path = os.path.join(TEMPLATES_DIR, f"{page_key}.png")
        if not os.path.exists(template_path):
            print(f"  ⚠ Template not found: {template_path} — skipping")
            continue

        existing = zones.get(page_key)
        if existing and "face_zone" in existing and "neck_anchor" in existing:
            redo = input(f"  {page_key} already fully set. Redo? (y/n): ").strip().lower()
            if redo != "y":
                continue

        result = get_zone_and_anchor_for_template(
            template_path, page_key, existing_zone=existing, face_path=face_path
        )

        if result:
            if page_key not in zones:
                zones[page_key] = {}
            zones[page_key]["face_zone"] = result["face_zone"]
            zones[page_key]["neck_anchor"] = result["neck_anchor"]

            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(zones, f, indent=2)
            print(f"  ✔ Saved ({len(zones)} pages total)")

    print("\n  ✔ Combined zones + anchors configured!\n")


if __name__ == "__main__":

    print("\n========================================")
    print("  STORYPHY — Zone Finder")
    print("========================================")
    print("  [1] Face placement zones")
    print("  [2] Cloud/text zones")
    print("  [3] Neck anchor points")
    print("  [4] Combined: size + anchor together")
    print("========================================")

    mode = input("  Choice (1/2/3/4): ").strip()

    if mode == "1":
        run_zone_finder()
    elif mode == "2":
        run_cloud_zone_finder()
    elif mode == "3":
        run_neck_anchor_finder()
    elif mode == "4":
        run_combined_finder()
    else:
        print("  Invalid choice — enter 1, 2, 3, or 4")