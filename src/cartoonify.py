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
CARTOON_PROMPT = """Edit the uploaded image.

This is an image editing task, NOT an image generation task.

The uploaded child is the final character.

Do NOT redesign the child.

Do NOT create a different child.

Your only job is to convert the rendering style while preserving the child's identity exactly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HIGHEST PRIORITY

Preserve the child's identity with the highest possible accuracy.

The final image must immediately look like the same child.

Preserve exactly:

• Face shape
• Facial proportions
• Eye size
• Eye spacing
• Eyebrows
• Nose
• Lips
• Smile shape
• Jawline
• Chin
• Forehead
• Hairline
• Hairstyle
• Hair direction
• Hair texture
• Ear shape
• Skin tone

Do NOT enlarge the eyes.

Do NOT reduce the nose.

Do NOT sharpen the jawline.

Do NOT smooth away unique facial features.

Do NOT beautify the child.

Do NOT make the child more "cute."

Do NOT westernize facial features.

Do NOT change ethnicity.

Do NOT change the child's apparent age.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STYLE

Only convert the rendering into a premium semi-realistic 3D children's storybook illustration.

The face geometry must remain unchanged.

Only modify:

• rendering
• lighting
• textures
• materials
• shading

Do NOT modify facial structure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPRESSION

Generate a gentle natural smile.

Keep the smile subtle.

Do not exaggerate it.

Keep the child's natural personality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAMERA

Keep exactly the same camera angle as the uploaded photograph.

Keep the face looking directly toward the camera.

Keep the same perspective.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BACKGROUND

Remove the background completely.

Generate a transparent PNG.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT

Generate a perfectly centered square portrait.

Output only:

• Head
• Hair
• Ears
• Neck

Do not generate:

• Shoulders
• Torso
• Clothing below the neck

End naturally at the base of the neck.

Leave comfortable transparent padding around the hair.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUALITY

Premium semi-realistic Storyphy style.

Soft cinematic lighting.

Natural skin texture.

High-resolution.

Transparent PNG.

Production-ready for compositing into Storyphy templates.

The result should look like the uploaded child illustrated in Storyphy's art style, not a newly generated child."""

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