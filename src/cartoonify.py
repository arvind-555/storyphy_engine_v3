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
from openai import OpenAI
from PIL import Image
from io import BytesIO

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
REFERENCE_IMAGE = "input/reference_style.png"

# Your standard Pixar conversion prompt
CARTOON_PROMPT = """I am uploading a real photo of a child.
Convert this EXACT child's face into Pixar 3D animation style.

STRICT RULES — do not break any of these:

Identity — copy everything from the uploaded photo:
- Skin tone must match the photo exactly — sample the real
  skin color and preserve it, do not make it darker or more orange
- Face shape must match the photo — if the child has a round
  face keep it round, if slim keep it slim
- Eye shape and size must match the photo exactly — do NOT
  enlarge or widen the eyes
- Nose shape and size must match the photo
- Mouth width and smile must match the photo
- Hair color, hair texture, and hairline must match the photo
  exactly — straight, wavy, or curly as shown
- Preserve every distinctive feature visible in the photo —
  dimples, birthmarks, bindi, glasses — only if they actually
  appear in the photo. Do not add features that are not there.

Art style:
- Pixar 3D animation rendering — smooth skin, soft warm
  lighting from above, subtle shading
- Bright natural eyes with a small white catchlight
- Clean crisp edges, vibrant but natural colors

Output:
- Transparent background
- Face and short neck only — no body, no clothing
- Circular crop around the face
- Clean flat edge at the bottom of the neck

The cartoon must be INSTANTLY recognizable as the SAME child
from the uploaded photo. Do not invent or change any feature.
If the output looks like a different child — that is a FAILURE."""

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
    print(f"  → Reference   : {REFERENCE_IMAGE}")
    print("  ──────────────────────────────────────────")

    confirm = input("  Proceed with API call? (y/n): ").strip().lower()

    if confirm != "y":
        print("  → Cancelled by user. No credits used.")
        return None

    # ── Step 5: Send to gpt-image-1 ──────────────────────────
    print("\n  → Sending to OpenAI gpt-image-1...")
    print("  → This takes 15-30 seconds. Please wait...")

    try:
        # Open both images as file objects
        with open(input_path, "rb") as child_file:

            response = client.images.edit(
                model="gpt-image-1",
                image=child_file,
                prompt=CARTOON_PROMPT,
                size="1024x1024",
                quality="high",
                background="transparent",
                n=1
            )

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