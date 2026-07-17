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

# Your standard Pixar conversion prompt
CARTOON_PROMPT = """Transform the uploaded child photo into a premium semi-realistic 3D Storyphy character.

This is an image transformation task, NOT a re-imagination task.

The uploaded photo is the ONLY identity reference.

Preserve the child's exact identity, including:
• Face shape
• Eyes
• Eyebrows
• Nose
• Lips
• Smile
• Hair
• Ears
• Skin tone
• Age

Do NOT beautify, stylize, or alter the child's facial proportions.

Create a premium semi-realistic 3D storybook illustration with soft cinematic lighting, Pixar-quality rendering, smooth natural skin, and luxury children's book quality.

The child must face directly forward with a centered, upright head and a natural happy smile.

BACKGROUND
• Transparent background only.
• No shadows.
• No outline.
• No glow.

NECK (CRITICAL)

Generate only the head and neck.

The neck should be straight, centered, and maintain nearly the same width from beneath the jaw to the bottom.

Terminate the neck in a small, smooth oval.

Do NOT generate:
• Shoulders
• Shoulder curves
• Trapezius muscles
• Collarbones
• Upper chest
• Side extensions
• Neck widening toward the bottom

The output must resemble a clean production-ready head asset, not a portrait or bust.

Do NOT generate:
• Shoulders
• Collarbones
• Upper chest
• Trapezius muscles
• Neck-to-shoulder transitions
• Side extensions beside the neck

The widest part below the jaw should be the neck itself.

The output must end before the shoulders begin.

The neck should fit naturally into a Storyphy character template without requiring additional editing.

OUTPUT

• Square image
• Transparent PNG
• Complete hair
• Complete ears
• Complete neck
• Head centered
• Comfortable padding around the head
• Premium production-ready quality

The style should change.
The child's identity should NOT."""

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