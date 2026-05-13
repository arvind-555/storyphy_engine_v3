# ============================================================
# main.py
# STORYPHY — Master Pipeline Controller
# ============================================================
# This is the ONLY file you need to run to create a book.
#
# Usage:
#   python main.py
#
# What it does:
#   1. Asks for child's name and photo path
#   2. Runs face_prep.py  — detects, crops, removes background
#   3. Runs cartoonify.py — applies Pixar-style cartoon effect
#   4. Runs compositor.py — pastes face + text onto all 28 pages
#   5. Runs pdf_builder.py — assembles final print-ready PDF
#
# Each order gets its own output folder:
#   output/{child_name}_{timestamp}/
#       ├── face_ready.png
#       ├── face_cartoon.png
#       ├── pages/
#       └── {child_name}_alphabet_book.pdf
# ============================================================

import os
import sys
import time
import shutil
from datetime import datetime

# ── Add src/ to path so we can import our modules ────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ── Load environment variables from .env ─────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Import all pipeline modules ───────────────────────────────
from face_prep   import prepare_face
from cartoonify  import cartoonify_face
from compositor  import compose_all_pages
from pdf_builder import build_pdf


# ── Helper: Print a nice section banner ──────────────────────
def print_banner(step_num, total_steps, title):
    print("\n" + "═" * 55)
    print(f"  STEP {step_num}/{total_steps} — {title}")
    print("═" * 55)


# ── Helper: Print final summary ───────────────────────────────
def print_summary(child_name, output_dir, pdf_path, duration_seconds):
    print("\n" + "★" * 55)
    print("  ✅ BOOK CREATED SUCCESSFULLY!")
    print("★" * 55)
    print(f"  Child name : {child_name}")
    print(f"  Output dir : {output_dir}")
    print(f"  PDF file   : {pdf_path}")
    print(f"  Time taken : {duration_seconds:.1f} seconds")
    print("★" * 55 + "\n")


# ── Main Pipeline ─────────────────────────────────────────────
def run_pipeline(child_name, photo_path):
    """
    Runs the full Storyphy book creation pipeline.

    Args:
        child_name  : child's name as a string
        photo_path  : path to the child's photo
    """

    start_time = time.time()

    print("\n" + "═" * 55)
    print("  🎨 STORYPHY — Personalized Book Creator")
    print("  Alphabet Book Pipeline")
    print("═" * 55)
    print(f"  Child name : {child_name}")
    print(f"  Photo      : {photo_path}")
    print("═" * 55)

    # ── Create unique output folder for this order ────────────
    # Format: output/Emma_20240315_143022/
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    order_name = f"{child_name.strip().capitalize()}_{timestamp}"
    output_dir = os.path.join("output", order_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  → Order folder created: {output_dir}")

    # ── Define all file paths for this order ──────────────────
    face_ready_path   = os.path.join(output_dir, "face_ready.png")
    face_cartoon_path = os.path.join(output_dir, "face_cartoon.png")
    pages_dir         = os.path.join(output_dir, "pages")
    pdf_path          = os.path.join(output_dir, f"{child_name}_alphabet_book.pdf")

    # ══════════════════════════════════════════════════════════
    # STEP 1 — Face Preparation
    # ══════════════════════════════════════════════════════════
    print_banner(1, 4, "Face Detection & Background Removal")

    face_result = prepare_face(photo_path, face_ready_path)

    if not face_result:
        print("\n❌ PIPELINE FAILED at Step 1 — Face Preparation")
        print("   Please check the photo and try again.")
        return None

    # ══════════════════════════════════════════════════════════
    # STEP 2a — Normalize photo
    # ══════════════════════════════════════════════════════════
    print_banner(2, 5, "Normalizing Photo")

    from normalize import normalize_photo

    face_normalized_path = os.path.join(output_dir, "face_normalized.png")
    normalize_result     = normalize_photo(face_ready_path, face_normalized_path)

    if not normalize_result:
        print("  ⚠ Normalization failed — using original face")
        face_normalized_path = face_ready_path

    # ══════════════════════════════════════════════════════════
    # STEP 2b — Cartoonify normalized photo
    # ══════════════════════════════════════════════════════════
    print_banner(3, 5, "Cartoonify Face (Pixar Style)")

    # Use normalized face as input instead of raw face
    cartoon_result = cartoonify_face(face_normalized_path, face_cartoon_path)

    # ══════════════════════════════════════════════════════════
    # STEP 3 — Compose All Pages
    # ══════════════════════════════════════════════════════════
    print_banner(4, 5, "Composing All 28 Pages")

    pages_result = compose_all_pages(
        cartoon_face_path = face_cartoon_path,
        child_name        = child_name,
        output_dir        = pages_dir
    )

    if not pages_result:
        print("\n❌ PIPELINE FAILED at Step 3 — Page Composition")
        print("   Check that zones.json exists and templates are in place.")
        return None
    

    # ══════════════════════════════════════════════════════════
    # STEP 4 — Add text to pages
    # ══════════════════════════════════════════════════════════
    print_banner(4, 5, "Adding Rhyme Text to Pages")

    import glob as glob_module
    import importlib.util
    spec   = importlib.util.spec_from_file_location(
                "add_text", 
                os.path.join("tools", "add_text.py")
             )
    add_text_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(add_text_mod)
    add_text_to_page = add_text_mod.add_text_to_page

    # Load zones
    import json
    with open("config/zones.json", "r") as f:
        zones = json.load(f)

    # Process all composed pages
    page_files = sorted(glob_module.glob(os.path.join(pages_dir, "*.png")))
    print(f"  → Adding text to {len(page_files)} pages...")

    for page_path in page_files:
        filename = os.path.basename(page_path)
        parts    = filename.replace(".png", "").split("_", 1)
        page_key = parts[1] if len(parts) > 1 else parts[0]
        add_text_to_page(page_path, page_key, child_name, zones)

    print("  ✔ Text added to all pages!\n")


    # ══════════════════════════════════════════════════════════
    # STEP 5 — Build PDF
    # ══════════════════════════════════════════════════════════
    print_banner(5, 5, "Assembling Final PDF")

    pdf_result = build_pdf(
        pages_dir       = pages_dir,
        output_pdf_path = pdf_path,
        child_name      = child_name
    )

    if not pdf_result:
        print("\n❌ PIPELINE FAILED at Step 4 — PDF Assembly")
        return None

    # ══════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════
    duration = time.time() - start_time
    print_summary(child_name, output_dir, pdf_path, duration)

    return pdf_path


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "═" * 55)
    print("  🎨 STORYPHY — Personalized Book Creator")
    print("═" * 55)

    # ── Get child's name ──────────────────────────────────────
    child_name = input("\n  Enter child's name: ").strip()

    if not child_name:
        print("  ✖ ERROR: Name cannot be empty.")
        sys.exit(1)

    # ── Get photo path ────────────────────────────────────────
    print("\n  Enter path to child's photo.")
    print("  (or press ENTER to use input/test_child.jpg)")
    photo_input = input("  Photo path: ").strip()

    # Use default test photo if nothing entered
    if not photo_input:
        photo_path = "input/test_child.jpg"
    else:
        # Remove quotes if user dragged and dropped file
        photo_path = photo_input.strip('"').strip("'")

    # Check photo exists
    if not os.path.exists(photo_path):
        print(f"\n  ✖ ERROR: Photo not found at: {photo_path}")
        print("    Make sure the file exists and try again.")
        sys.exit(1)

    print(f"\n  ✔ Name  : {child_name}")
    print(f"  ✔ Photo : {photo_path}")
 
    # ── Confirm before running ────────────────────────────────
    confirm = input("\n  Start creating the book? (y/n): ").strip().lower()

    if confirm != "y":
        print("  → Cancelled.")
        sys.exit(0)

    # ── Run the pipeline ──────────────────────────────────────
    result = run_pipeline(child_name, photo_path)

    if not result:
        print("\n❌ Book creation failed. Check errors above.")
        sys.exit(1)