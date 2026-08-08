# ============================================================
# cartoonify.py
# STORYPHY — Step 2: Cartoonify Face using OpenAI gpt-image-1
# ============================================================
# Takes the child's real photo + a reference style image and
# generates a Pixar-style cartoon version preserving the
# child's facial features.
# ============================================================

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

print("▶ Running cartoonify.py — OpenAI gpt-image-1 Cartoonification")

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


def anchor_to_neck_bottom(cartoon_image, canvas_size=ANCHOR_CANVAS_SIZE,
                           alpha_threshold=ANCHOR_ALPHA_THRESHOLD):
    """
    Finds the subject's bounding box (ignoring faint feathered edges),
    scales the WHOLE image so the subject height always equals a fixed
    target % of the canvas (same standard as face_prep.py), then
    bottom-anchors + horizontally centers it. This guarantees every
    face_cartoon.png — whether fresh from the API or reused from an
    old run — ends up the same size AND position, regardless of how
    gpt-image-1 happened to frame this particular child.

    Takes a PIL RGBA image, returns a new PIL RGBA image on a fixed
    transparent canvas.
    """
    # Strip artifacts (stray specks, edge lines) from the image itself,
    # not just from our measurement — otherwise they remain in the saved
    # PNG and every downstream step that trims with a plain getbbox()
    # re-inherits the same off-centre error.
    cartoon_image, bbox = clean_subject(cartoon_image, alpha_threshold)

    # Safety check: if the bbox covers (almost) the ENTIRE canvas, the
    # image likely has no real transparency (e.g. a file downloaded
    # from ChatGPT's web UI, flattened onto solid white with alpha
    # fully opaque everywhere) rather than a true transparent PNG from
    # our own API call. Run rembg on it first so measurement is
    # actually meaningful, instead of scaling based on the whole canvas.
    if bbox and (bbox[2] - bbox[0]) >= cartoon_image.width * 0.95 and \
       (bbox[3] - bbox[1]) >= cartoon_image.height * 0.95:
        print("  ⚠ No real transparency detected — running background "
              "removal first (this file likely wasn't from our own API call)")
        cartoon_image = remove(cartoon_image)
        cartoon_image, bbox = clean_subject(cartoon_image, alpha_threshold)

    if not bbox:
        print("  ⚠ Could not find subject bounds — skipping neck anchor")
        return cartoon_image

    x1, y1, x2, y2 = bbox
    subject_h = y2 - y1

    target_x = canvas_size // 2
    target_y = canvas_size - int(canvas_size * ANCHOR_BOTTOM_MARGIN)
    target_h = int(canvas_size * TARGET_SUBJECT_HEIGHT_PCT)

    # Scale the whole image so subject height hits the fixed target %
    # — same standard face_prep.py already applies to face_ready.png,
    # so option 1 (fresh API) and option 2 (reuse) both end up
    # identically scaled, not just identically positioned.
    scale = target_h / subject_h
    new_w = int(cartoon_image.width * scale)
    new_h = int(cartoon_image.height * scale)
    print(f"  → Scaling subject height {subject_h}px → {target_h}px ({scale:.2f}x)")
    cartoon_image = cartoon_image.resize((new_w, new_h), Image.LANCZOS)

    # Scale the ORIGINAL bbox coordinates by the same factor instead of
    # re-measuring on the resized image. Re-measuring is unreliable for
    # wispy/curly hair — thin, partially-transparent strands can drop
    # below alpha_threshold after LANCZOS resampling, making the subject
    # look artificially shorter and throwing off the anchor position.
    # Geometry (scaling known coordinates) doesn't have this problem.
    neck_center_x = int(((x1 + x2) // 2) * scale)
    neck_bottom_y = int(y2 * scale)

    offset_x = target_x - neck_center_x
    offset_y = target_y - neck_bottom_y

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    canvas.paste(cartoon_image, (offset_x, offset_y), cartoon_image)

    print(f"  ✔ Neck anchored: bottom moved from y={neck_bottom_y} "
          f"→ y={target_y}, center from x={neck_center_x} → x={target_x}")

    return canvas

# ── Reference images ─────────────────────────────────────────
# A fixed neck-geometry reference passed alongside the child's photo.
# Giving the model a geometry to match is far more reliable than
# describing the neck in words — the prompt alone produced random
# neck widths, which was the root of the placement inconsistency.
# The prompt MUST label the images by number (see CARTOON_PROMPT).
NECK_REFERENCE_PATH = "config/neck_reference.png"

# ── Daily spend tracker ──────────────────────────────────────
SPEND_LOG_FILE = "config/spend_log.json"
DAILY_LIMIT    = 2.00   # max $ per day (safety limit)


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

# Your standard Pixar conversion prompt
CARTOON_PROMPT = """Two input images are provided.
IMAGE 1 — CHILD IDENTITY
This image is the ONLY identity reference.
Preserve exactly:
• Face shape
• Facial proportions
• Hair
• Hairline
• Hair texture
• Eyes
• Eyebrows
• Nose
• Lips
• Smile
• Ears
• Skin tone
• Age
• Gender
Do NOT alter the child's identity.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE 2 — STORYPHY NECK GEOMETRY
This image is NOT an identity reference.
Use it ONLY to reproduce the neck geometry.
Match ONLY:
• Neck width
• Neck length
• Neck silhouette
• Neck ending
Ignore every other aspect of Image 2.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
Create a premium semi-realistic 3D Storyphy character.
Generate ONLY:
• Hair
• Head
• Face
• Ears
• Standardized neck
Do NOT generate:
• Shoulders
• Collarbones
• Chest
• Clothing
• Trapezius muscles
The face must come entirely from Image 1.
The neck geometry must come entirely from Image 2.
Background must be completely transparent.
Generate a perfectly centered square image (1:1).
The head should occupy approximately 70–75% of the canvas while leaving comfortable transparent padding around the hair.
The child should face directly forward with a centered head and a warm, natural smile.
The output should be a production-ready transparent PNG suitable for direct placement into Storyphy templates."""

# ── Main Function ─────────────────────────────────────────────
def cartoonify_face(input_path, output_path):
    """
    Sends child's photo + reference style image to OpenAI 
    gpt-image-1 and saves the Pixar cartoon result.

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
    # Each call to gpt-image-1 costs approximately $0.04 - $0.19
    # Quality settings: low ~$0.04, medium ~$0.08, high ~$0.19
    print("\n  ──────────────────────────────────────────")
    print("  💳 ABOUT TO MAKE API CALL")
    print("  ──────────────────────────────────────────")
    print(f"  → Model       : gpt-image-1")
    print(f"  → Quality     : high (~$0.19 per image)")
    print(f"  → Input image : {input_path}")
    print("  ──────────────────────────────────────────")

    confirm = input("  Proceed with API call? (y/n): ").strip().lower()

    if confirm != "y":
        print("  → Cancelled by user. No credits used.")
        return None

    # ── Step 5: Send to gpt-image-1 ──────────────────────────
    print("\n  → Sending to OpenAI gpt-image-1...")
    print("  → This takes 15-30 seconds. Please wait...")

    try:
        # gpt-image-1's edit endpoint accepts a LIST of images. Order
        # matters and must match how CARTOON_PROMPT refers to them:
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

            response = client.images.edit(
                model="gpt-image-1",
                image=images_payload,
                prompt=CARTOON_PROMPT,
                size="1024x1024",
                quality="high",
                input_fidelity="high",
                background="transparent",
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

    # ── Step 5: Decode the result ─────────────────────────────
    print("  → Decoding result...")

    try:
        # gpt-image-1 returns base64 encoded image
        image_b64   = response.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)

        cartoon_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        print(f"  ✔ Cartoon image decoded: {cartoon_image.size}")

    except Exception as e:
        print(f"  ✖ ERROR: Failed to decode result: {e}")
        return None

    # ── Step 5b: Anchor neck-bottom to a fixed canvas position ──
    # gpt-image-1 doesn't guarantee identical framing across kids,
    # so we measure where THIS image's neck actually ends and
    # reposition it onto a fixed target — deterministic per-image,
    # no reliance on prompt obedience.
    print("  → Anchoring neck position...")
    cartoon_image = anchor_to_neck_bottom(cartoon_image)

    # ── Step 6: Save final result ─────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cartoon_image.save(output_path, format="PNG")

    print(f"  ✔ Cartoon face saved: {output_path}")
    print("  ✔ cartoonify.py complete!\n")
        
    # Log the spend (high quality = $0.19 per image)
    log_spend(0.19)
    
    print("  ✔ cartoonify.py complete!\n")

    return output_path


# ── Run directly for testing ──────────────────────────────────
if __name__ == "__main__":

    import glob

    # Find most recent face_ready.png for testing
    candidates = sorted(
        glob.glob("output/*/face_ready.png"),
        key=os.path.getmtime,
        reverse=True
    )

    if not candidates:
        print("❌ No face_ready.png found. Run main.py first.")
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