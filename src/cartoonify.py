# ============================================================
# cartoonify.py
# STORYPHY — Step 2: Cartoonify Face using OpenAI gpt-image-2
# ============================================================
# Takes the child's prepped photo + a neck geometry reference and
# generates a semi-realistic illustrated version preserving the
# child's facial features, then normalizes size and position onto
# a fixed canvas for deterministic compositing.
# ============================================================

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

print("▶ Running cartoonify.py — OpenAI gpt-image-2 Cartoonification")

import os
import base64
import numpy as np
from scipy import ndimage
from openai import OpenAI
from PIL import Image
from io import BytesIO
from rembg import remove

# ── Neck-anchor normalization config ─────────────────────────
ANCHOR_CANVAS_SIZE = 800    # must match face_prep.py's STANDARD_SIZE
ANCHOR_BOTTOM_MARGIN = 0    # flush at canvas bottom, no margin
ANCHOR_ALPHA_THRESHOLD = 25 # ignore faint feathered alpha when measuring
TARGET_SUBJECT_HEIGHT_PCT = 0.68  # must match face_prep.py's TARGET_SUBJECT_HEIGHT_PCT
ANCHOR_MIN_ROW_PIXELS = 5    # legacy, unused (see clean_subject)
ANCHOR_FEATHER_PAD = 4       # dilate the kept region by this many px so soft
                             # hair feathering isn't cut into a hard edge

# ── Horizontal centering target ──────────────────────────────
# Face centre x on the reference face that composites correctly across
# all 28 templates. This is MEASURED, not assumed — it is 385, not
# canvas mid-x (400). Targeting 400 would put every face 15px off.
#
# Recalibrate if ANCHOR_CANVAS_SIZE, CARTOON_PROMPT, or the model
# changes. Reference file: config/reference_face.png
#   python -c "import sys; sys.path.insert(0,'src'); from PIL import Image; \
#     from cartoonify import find_face_center_x; \
#     print(find_face_center_x(Image.open('config/reference_face.png').convert('RGBA')))"
REFERENCE_FACE_CX = 385
FACE_CX_TOLERANCE = 25   # warn if the corrected face lands outside this


def clean_subject(pil_rgba, alpha_threshold=ANCHOR_ALPHA_THRESHOLD,
                  feather_pad=ANCHOR_FEATHER_PAD):
    """
    Isolates the child (the largest connected region of non-transparent
    pixels) and ERASES everything else by zeroing its alpha. Returns
    (cleaned_image, bbox_of_subject).

    Two things make this necessary:

    1. AI-generated / re-saved PNGs often carry artifacts — stray specks,
       or dense full-height vertical lines along an edge. Measuring with
       a plain getbbox() includes them, dragging the bbox to the canvas
       edge and wrecking both scale and horizontal centering.

    2. It isn't enough to merely measure around artifacts, because the
       artifacts stay in the saved file. Anything downstream that trims
       or centers with a plain getbbox() (compositor.py, zone_finder.py)
       would then re-inherit the same error. So we delete them here,
       once, and every later step sees a clean subject.

    The kept region is dilated slightly so the soft feathered alpha
    around hair survives instead of being clipped to a hard edge.
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


def get_robust_bbox(pil_rgba, alpha_threshold=ANCHOR_ALPHA_THRESHOLD,
                    min_pixels=ANCHOR_MIN_ROW_PIXELS):
    """Bbox of the largest connected region only (no image modification)."""
    _, bbox = clean_subject(pil_rgba, alpha_threshold)
    return bbox


def find_face_center_x(pil_rgba):
    """
    Horizontal centre of the FACE, in pixels.

    Preferred over the alpha bbox centre because hair is part of the
    bbox: a side-swept fringe extends the bbox to one side, dragging its
    centre away from the actual face and taking the face off-centre with
    it. Hair volume varies per child, so the resulting error varies too —
    which is why the drift was inconsistent rather than a fixed offset.

    Uses the eye midpoint (anatomical, unaffected by hair), falling back
    to the face-box centre, then None if no face is found at all.
    """
    import cv2
    from face_prep import _detect_face   # local import avoids circular import

    rgb = np.array(pil_rgba.convert("RGB"))
    det = _detect_face(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if det is None:
        return None

    h, w = rgb.shape[:2]
    try:
        r_eye, l_eye = det.keypoints[0], det.keypoints[1]
        return int((r_eye.x + l_eye.x) / 2 * w)
    except (AttributeError, IndexError):
        bb = det.bounding_box
        return bb.origin_x + bb.width // 2


def anchor_to_neck_bottom(cartoon_image, canvas_size=ANCHOR_CANVAS_SIZE,
                           alpha_threshold=ANCHOR_ALPHA_THRESHOLD):
    """
    Normalizes a raw model output onto a fixed canvas so compositor.py
    can place it without measuring anything.

    Guarantees the contract compositor.py relies on:
      • canvas is exactly ANCHOR_CANVAS_SIZE square
      • subject height is exactly TARGET_SUBJECT_HEIGHT_PCT of canvas
      • neck bottom is flush at the canvas bottom edge
      • face centre sits at REFERENCE_FACE_CX

    Vertical uses the alpha bbox bottom (correct — the neck genuinely is
    the lowest point). Horizontal uses face detection (the bbox centre is
    unreliable, see find_face_center_x).

    Takes a PIL RGBA image, returns a new PIL RGBA image.
    """
    # Strip artifacts (stray specks, edge lines) from the image itself,
    # not just from our measurement — otherwise they remain in the saved
    # PNG and every downstream step that trims with a plain getbbox()
    # re-inherits the same off-centre error.
    cartoon_image, bbox = clean_subject(cartoon_image, alpha_threshold)

    # Safety check: if the bbox covers (almost) the ENTIRE canvas, the
    # image has no real transparency. Two cases hit this path:
    #   • a file downloaded from ChatGPT's web UI (flattened onto white)
    #   • gpt-image-2 output — the model has no transparent background
    #     option, so we request a flat white background in the prompt
    #     and remove it here.
    # Either way, run rembg first so measurement is meaningful instead
    # of scaling based on the whole canvas.
    if bbox and (bbox[2] - bbox[0]) >= cartoon_image.width * 0.95 and \
       (bbox[3] - bbox[1]) >= cartoon_image.height * 0.95:
        print("  → No transparency in source — running background removal")
        cartoon_image = remove(cartoon_image)
        cartoon_image, bbox = clean_subject(cartoon_image, alpha_threshold)

    if not bbox:
        print("  ⚠ Could not find subject bounds — skipping neck anchor")
        return cartoon_image

    x1, y1, x2, y2 = bbox
    subject_h = y2 - y1

    target_x = REFERENCE_FACE_CX
    target_y = canvas_size - int(canvas_size * ANCHOR_BOTTOM_MARGIN)
    target_h = int(canvas_size * TARGET_SUBJECT_HEIGHT_PCT)

    # Scale the whole image so subject height hits the fixed target %
    # — same standard face_prep.py already applies to face_ready.png,
    # so option 1 (fresh API) and option 2 (reuse) both end up
    # identically scaled, not just identically positioned.
    #
    # Detection must run BEFORE the resize, since `scale` is applied to
    # its result below. Detecting after would double-apply the scale.
    pre_resize_image = cartoon_image

    scale = target_h / subject_h
    new_w = int(cartoon_image.width * scale)
    new_h = int(cartoon_image.height * scale)
    print(f"  → Scaling subject height {subject_h}px → {target_h}px ({scale:.2f}x)")
    cartoon_image = cartoon_image.resize((new_w, new_h), Image.LANCZOS)

    # HORIZONTAL — centre on the detected face, not the bbox.
    face_cx = find_face_center_x(pre_resize_image)
    if face_cx is not None:
        neck_center_x = int(face_cx * scale)
        print(f"  → Centering on detected face (x={face_cx} pre-scale)")
    else:
        neck_center_x = int(((x1 + x2) // 2) * scale)
        print("  ⚠ No face detected — using bbox centre (may drift)")

    # VERTICAL — scale the ORIGINAL bbox coordinate rather than
    # re-measuring on the resized image. Re-measuring is unreliable for
    # wispy/curly hair: thin, partially-transparent strands can drop
    # below alpha_threshold after LANCZOS resampling, making the subject
    # look artificially shorter and throwing off the anchor position.
    # Geometry (scaling a known coordinate) doesn't have this problem.
    neck_bottom_y = int(y2 * scale)

    offset_x = target_x - neck_center_x
    offset_y = target_y - neck_bottom_y

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    canvas.paste(cartoon_image, (offset_x, offset_y), cartoon_image)

    print(f"  ✔ Neck anchored: bottom y={neck_bottom_y} → {target_y}, "
          f"face x={neck_center_x} → {target_x}")

    # ── Verify ───────────────────────────────────────────────
    # If this fires AFTER correction, detection was unreliable on this
    # image — an odd head angle, or a style the detector struggles with.
    # Worth catching here rather than after 28 pages are composed.
    check_cx = find_face_center_x(canvas)
    if check_cx is None:
        print("  ⚠ Cannot verify centering — no face detected in output")
    elif abs(check_cx - REFERENCE_FACE_CX) > FACE_CX_TOLERANCE:
        print(f"  ⚠ CENTERING OFF: face at x={check_cx}, expected "
              f"{REFERENCE_FACE_CX} ±{FACE_CX_TOLERANCE} — check before printing")
    else:
        print(f"  ✔ Centering verified: face at x={check_cx}")

    return canvas


# ── Reference images ─────────────────────────────────────────
# A fixed neck-geometry reference passed alongside the child's photo.
# Giving the model a geometry to match is far more reliable than
# describing the neck in words — the prompt alone produced random
# neck widths, which was the root of the placement inconsistency.
# Tested without it: likeness was unchanged but the neck faded out
# with no defined bottom edge, breaking the anchor measurement.
# The prompt MUST label the images by number (see CARTOON_PROMPT).
NECK_REFERENCE_PATH = "config/neck_reference.png"

# ── Daily spend tracker ──────────────────────────────────────
SPEND_LOG_FILE = "config/spend_log.json"
DAILY_LIMIT    = 2.00   # max $ per day (safety limit)

# TODO: verify against OpenAI's current gpt-image-2 rates. This figure
# is carried over from gpt-image-1 and is almost certainly wrong, which
# means DAILY_LIMIT isn't protecting what you think it is.
COST_PER_IMAGE = 0.19


def check_daily_spend():
    """
    Tracks spending per day. Blocks API calls if daily 
    limit is hit. Helps prevent accidentally using all 
    credits in one bad run.
    """
    import json
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")

    # Load existing spend log
    spend_log = {}
    if os.path.exists(SPEND_LOG_FILE):
        try:
            with open(SPEND_LOG_FILE, "r") as f:
                spend_log = json.load(f)
        except:
            spend_log = {}

    today_spend = spend_log.get(today, 0.0)

    print(f"\n  💰 Today's spend so far: ${today_spend:.2f}")
    print(f"  💰 Daily limit         : ${DAILY_LIMIT:.2f}")

    if today_spend >= DAILY_LIMIT:
        print(f"  ⛔ Daily limit reached! No more API calls today.")
        print(f"     Edit DAILY_LIMIT in cartoonify.py to change.")
        return False

    return True


def log_spend(amount):
    """Records spend after each API call."""
    import json
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")

    spend_log = {}
    if os.path.exists(SPEND_LOG_FILE):
        try:
            with open(SPEND_LOG_FILE, "r") as f:
                spend_log = json.load(f)
        except:
            spend_log = {}

    spend_log[today] = spend_log.get(today, 0.0) + amount

    os.makedirs(os.path.dirname(SPEND_LOG_FILE), exist_ok=True)
    with open(SPEND_LOG_FILE, "w") as f:
        json.dump(spend_log, f, indent=2)

    print(f"  💰 Spend logged: ${amount:.2f} | Today total: ${spend_log[today]:.2f}")


# ── Configuration ────────────────────────────────────────────

CARTOON_PROMPT = """
Two input images are provided.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS TASK IS

This is a light retouch of a photograph. It is NOT a character design,
NOT a re-illustration, and NOT a new character.

Keep the photograph's exact facial geometry. Change only the surface:
smooth the skin, simplify fine detail, soften shading into clean even
tones. Approximately 80% photographic, 20% illustrated.

The output must be immediately recognisable as this specific child
by their own parent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE 1 — THE CHILD

Keep exactly as photographed:
- Face width, jaw shape, chin shape and taper
- Cheek fullness
- Eye size, eye shape, iris size, spacing between the eyes
- Nose width and length
- Lip thickness
- Ear size, shape, and how far they sit out from the head
- Hairline, hair direction, hair volume
- Skin tone and its exact depth
- Head angle and tilt

PROPORTIONS — measure these against the photograph, do not invent them:
- Face height is roughly 1.35x face width. Do not shorten or widen.
- Eye width is about one fifth of face width. Do not enlarge.
- Iris diameter matches the photograph exactly.
- Forehead occupies the top third of the face, no more.
- Jaw width matches the photograph. Do not broaden or round.

DO NOT:
- Beautify, idealise, or make the child more conventionally cute
- Widen, round, or shorten the face
- Enlarge the eyes or the irises
- Darken, lighten, or saturate the skin tone
- Correct natural asymmetry
- Change the head angle or tilt
- Alter the expression or the shape of the smile

If a feature looks unusual or imperfect, keep it. Those features are
what make this child recognisable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE 2 — NECK GEOMETRY REFERENCE ONLY

Match only: neck width, neck length, neck silhouette, and the flat
horizontal ending at the bottom edge.

This image carries NO identity information. Ignore its face, its skin
tone, and every other aspect of it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STYLE

Digitally painted portrait. Soft airbrushed shading, matte finish,
even front lighting. Simplified skin detail with clean edges.
Illustrated in surface, photographic in anatomy.

NOT a 3D animated film character. Not Pixar, Disney, or DreamWorks
styling. No enlarged eyes, no rounded caricature proportions, no
chibi or stylised child proportions. No glossy plastic rendering.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT

Render only: hair, head, face, ears, and the standardised neck.
Do not render: shoulders, collarbones, chest, clothing, trapezius.

Plain solid white background, evenly lit, no shadows. Square 1:1.
"""


# ── Main Function ─────────────────────────────────────────────
def cartoonify_face(input_path, output_path):
    """
    Sends the child's photo + neck reference to OpenAI gpt-image-2,
    then normalizes the result onto the fixed placement canvas.

    Args:
        input_path  : path to the child's face photo (PNG/JPG)
        output_path : where to save the cartoonified PNG
    """

    print(f"  → Input face : {input_path}")
    
    # ── Step 0: Check daily spend limit ──────────────────────
    if not check_daily_spend():
        return None
    
    # ── Step 1: Verify API key is set ────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print("  ✖ ERROR: OPENAI_API_KEY not set in .env file!")
        print("    Add this to your .env file:")
        print("    OPENAI_API_KEY=sk-your-key-here")
        return None

    # ── Step 2: Verify input files exist ─────────────────────
    if not os.path.exists(input_path):
        print(f"  ✖ ERROR: Input image not found: {input_path}")
        return None

    # ── Step 3: Initialize OpenAI client ─────────────────────
    client = OpenAI()

    # ── Step 4: Confirmation before API call ─────────────────
    print("\n  ──────────────────────────────────────────")
    print("  💳 ABOUT TO MAKE API CALL")
    print("  ──────────────────────────────────────────")
    print(f"  → Model       : gpt-image-2")
    print(f"  → Quality     : high (~${COST_PER_IMAGE:.2f} per image)")
    print(f"  → Input image : {input_path}")
    print("  ──────────────────────────────────────────")

    confirm = input("  Proceed with API call? (y/n): ").strip().lower()

    if confirm != "y":
        print("  → Cancelled by user. No credits used.")
        return None

    # ── Step 5: Send to gpt-image-2 ──────────────────────────
    print("\n  → Sending to OpenAI gpt-image-2...")
    print("  → This takes 15-30 seconds. Please wait...")

    try:
        # The edit endpoint accepts a LIST of images. Order matters and
        # must match how CARTOON_PROMPT refers to them:
        #   image 1 = the child's photo (identity source)
        #   image 2 = the neck geometry reference (shape source only)
        open_files = []
        try:
            child_file = open(input_path, "rb")
            open_files.append(child_file)
            images_payload = [child_file]

            if os.path.exists(NECK_REFERENCE_PATH):
                neck_file = open(NECK_REFERENCE_PATH, "rb")
                open_files.append(neck_file)
                images_payload.append(neck_file)
                print(f"  → Neck reference: {NECK_REFERENCE_PATH}")
            else:
                print(f"  ⚠ Neck reference not found at {NECK_REFERENCE_PATH}")
                print("    Sending child photo only — neck geometry will be "
                      "random, as before.")

            # NOTE: gpt-image-2 supports neither `input_fidelity` (it
            # handles fidelity internally) nor `background="transparent"`.
            # Both were gpt-image-1 parameters and now raise a 400.
            # Transparency is handled downstream by rembg — the prompt
            # asks for flat white, which rembg separates reliably.
            response = client.images.edit(
                model="gpt-image-2",
                image=images_payload,
                prompt=CARTOON_PROMPT,
                size="1024x1024",
                quality="high",
                n=1
            )
        finally:
            for f in open_files:
                f.close()

        print("  ✔ AI processing complete!")

    except Exception as e:
        print(f"  ✖ ERROR: OpenAI API call failed!")
        print(f"    Details: {e}")
        return None

    # ── Step 6: Decode the result ─────────────────────────────
    print("  → Decoding result...")

    try:
        image_b64   = response.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)

        cartoon_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        print(f"  ✔ Cartoon image decoded: {cartoon_image.size}")

    except Exception as e:
        print(f"  ✖ ERROR: Failed to decode result: {e}")
        return None

    # ── Step 7: Normalize onto the placement canvas ──────────
    # The model doesn't guarantee identical framing across kids, so we
    # measure where THIS image's neck ends and where its face sits, then
    # reposition onto fixed targets. Deterministic per-image, with no
    # reliance on prompt obedience.
    print("  → Anchoring neck position...")
    cartoon_image = anchor_to_neck_bottom(cartoon_image)

    # ── Step 8: Save final result ─────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cartoon_image.save(output_path, format="PNG")

    print(f"  ✔ Cartoon face saved: {output_path}")

    log_spend(COST_PER_IMAGE)

    print("  ✔ cartoonify.py complete!\n")

    return output_path


# ── Run directly for testing ──────────────────────────────────
if __name__ == "__main__":

    import glob

    # Find most recent face_crop_hires.png for testing — the tight head
    # crop is the identity input, not face_ready.png (which wastes most
    # of its canvas on empty space, leaving fewer real pixels on the face).
    candidates = sorted(
        glob.glob("output/*/face_crop_hires.png"),
        key=os.path.getmtime,
        reverse=True
    )

    if not candidates:
        print("❌ No face_crop_hires.png found. Run face_prep.py first.")
        exit(1)

    test_input  = candidates[0]
    test_output = os.path.join(
        os.path.dirname(test_input), 
        "face_cartoon.png"
    )

    print(f"  → Test input  : {test_input}")
    print(f"  → Test output : {test_output}")

    result = cartoonify_face(test_input, test_output)

    if result:
        print(f"\n✅ Success! Open '{test_output}' to see the result!")
    else:
        print("\n❌ Something went wrong. Check errors above.")