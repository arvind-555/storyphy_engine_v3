# ============================================================
# normalize.py
# STORYPHY — Photo Normalization
# ============================================================
# This script normalizes the input photo BEFORE cartoonifying
# so that different lighting, filters, and phone cameras all
# produce consistent cartoon results.
#
# Steps:
#   1. White balance correction (removes color casts)
#   2. Exposure normalization (consistent brightness)
#   3. Skin tone enhancement (natural skin colors)
#   4. Noise reduction (cleaner base image)
# ============================================================

print("▶ Running normalize.py — Photo Normalization")

import cv2
import numpy as np
from PIL import Image
import os


# ── Configuration ────────────────────────────────────────────
# These values are tuned for portrait/face photos
CLAHE_CLIP_LIMIT    = 2.0   # contrast enhancement strength
CLAHE_GRID_SIZE     = 8     # local contrast grid size
DENOISE_STRENGTH    = 6     # noise removal strength (3-10)
SHARPEN_AMOUNT      = 0.4   # sharpening after denoise (0-1)


# ── White Balance Correction ──────────────────────────────────
def correct_white_balance(img_bgr):
    """
    Removes color casts (yellow, blue, green tints) from photos.
    Uses the Gray World algorithm — assumes average color
    of a natural scene should be neutral gray.
    """

    # Convert to float for calculation
    img_float = img_bgr.astype(np.float32)

    # Calculate mean of each channel
    mean_b = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_r = np.mean(img_float[:, :, 2])

    # Overall mean brightness
    mean_gray = (mean_b + mean_g + mean_r) / 3

    # Scale each channel to neutral gray
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (mean_gray / mean_b), 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (mean_gray / mean_g), 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (mean_gray / mean_r), 0, 255)

    return img_float.astype(np.uint8)


# ── Exposure Normalization ────────────────────────────────────
def normalize_exposure(img_bgr):
    """
    Makes dark and bright photos consistent in exposure.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
    which normalizes brightness locally — much better than
    global histogram equalization which can look unnatural.
    """

    # Work in LAB color space — L channel is brightness only
    # This lets us normalize brightness without affecting colors
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to just the L (lightness) channel
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE)
    )
    l_normalized = clahe.apply(l)

    # Merge back and convert to BGR
    lab_normalized = cv2.merge([l_normalized, a, b])
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)


# ── Skin Tone Enhancement ─────────────────────────────────────
def enhance_skin_tones(img_bgr):
    """
    Gently enhances skin tones to look more natural and vibrant.
    Works across all skin tones — light, medium, dark, deep.
    Only affects skin-colored pixels, leaves background alone.
    """

    # Convert to HSV to work with color ranges
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Define skin tone range in HSV
    # These ranges cover all human skin tones
    skin_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 60],   dtype=np.uint8),  # lower bound
        np.array([25, 255, 255], dtype=np.uint8)   # upper bound
    )

    # Slightly boost saturation of skin pixels only
    s_float = s.astype(np.float32)
    s_float[skin_mask > 0] = np.clip(
        s_float[skin_mask > 0] * 1.15,  # gentle 15% boost
        0, 255
    )

    # Slightly brighten skin pixels
    v_float = v.astype(np.float32)
    v_float[skin_mask > 0] = np.clip(
        v_float[skin_mask > 0] * 1.05,  # gentle 5% boost
        0, 255
    )

    # Merge back
    hsv_enhanced = cv2.merge([
        h,
        s_float.astype(np.uint8),
        v_float.astype(np.uint8)
    ])

    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


# ── Noise Reduction ───────────────────────────────────────────
def reduce_noise(img_bgr):
    """
    Removes grain and noise from low quality phone photos.
    Uses Non-local Means Denoising — best quality but
    slightly slower than other methods.
    """

    denoised = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=DENOISE_STRENGTH,           # luminance noise filter strength
        hColor=DENOISE_STRENGTH,      # color noise filter strength
        templateWindowSize=7,         # size of patch used for denoising
        searchWindowSize=21           # size of window to search for patches
    )

    # Add slight sharpening after denoising
    # (denoising can make image slightly soft)
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    # Blend original sharpening with denoised
    sharpened = cv2.filter2D(denoised, -1, kernel)
    result = cv2.addWeighted(
        denoised,  1 - SHARPEN_AMOUNT,
        sharpened, SHARPEN_AMOUNT,
        0
    )

    return result


# ── Main Function ─────────────────────────────────────────────
def normalize_photo(input_path, output_path):
    """
    Runs all normalization steps on the input photo.
    Saves normalized version ready for cartoonification.

    Args:
        input_path  : path to original face PNG (from face_prep)
        output_path : where to save normalized face PNG
    """

    print(f"  → Loading image: {input_path}")

    # Load with PIL to handle transparency
    pil_img  = Image.open(input_path).convert("RGBA")
    r, g, b, alpha = pil_img.split()

    # Convert RGB to BGR for OpenCV
    rgb_array = np.array(Image.merge("RGB", (r, g, b)))
    img_bgr   = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    print(f"  → Image size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # ── Step 1: White balance ─────────────────────────────────
    print("  → Step 1: Correcting white balance...")
    img_bgr = correct_white_balance(img_bgr)
    print("  ✔ White balance corrected")

    # ── Step 2: Exposure normalization ───────────────────────
    print("  → Step 2: Normalizing exposure...")
    img_bgr = normalize_exposure(img_bgr)
    print("  ✔ Exposure normalized")

    # ── Step 3: Skin tone enhancement ────────────────────────
    print("  → Step 3: Enhancing skin tones...")
    img_bgr = enhance_skin_tones(img_bgr)
    print("  ✔ Skin tones enhanced")

    # ── Step 4: Noise reduction ───────────────────────────────
    print("  → Step 4: Reducing noise...")
    img_bgr = reduce_noise(img_bgr)
    print("  ✔ Noise reduced")

    # ── Restore transparency and save ────────────────────────
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = Image.fromarray(img_rgb).convert("RGBA")
    result.putalpha(alpha)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path, format="PNG")

    print(f"  ✔ Normalized image saved: {output_path}")
    print("  ✔ normalize.py complete!\n")

    return output_path


# ── Run directly for testing ──────────────────────────────────
if __name__ == "__main__":

    test_input  = "output/face_ready.png"
    test_output = "output/face_normalized.png"

    result = normalize_photo(test_input, test_output)

    if result:
        print(f"✅ Success! Check '{test_output}' to see normalized face.")
    else:
        print("❌ Something went wrong. Check errors above.")