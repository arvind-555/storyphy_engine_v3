# ============================================================
# main.py
# STORYPHY — Master Pipeline Controller
# ============================================================
# Pipeline:
#   1. Face prep   — crop face + remove background (local)
#   2. Cartoonify  — OpenAI gpt-image-1 Pixar style
#   3. Compose     — paste cartoon face on all 28 pages
#   4. Add text    — overlay rhyme text on each page
#   5. PDF         — assemble final book
#
# NOTE — Step 2 now sends the STANDARDIZED face_ready.png
# (fixed 800x800, face always in the same relative position)
# instead of the raw photo. This gives consistent framing in
# the AI's output, which compositor.py relies on for
# deterministic, detection-free face placement.
# ============================================================

import os
import sys
import time
import glob
import json
import shutil
import importlib.util
from datetime import datetime

# ── Add src/ to path ─────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ── Load environment variables ───────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Import pipeline modules ──────────────────────────────────
from face_prep   import prepare_face
from cartoonify  import cartoonify_face, anchor_to_neck_bottom
from compositor  import compose_all_pages
from pdf_builder import build_pdf


# ── Helper: Print section banner ─────────────────────────────
def print_banner(step_num, total_steps, title):
    print("\n" + "=" * 55)
    print(f"  STEP {step_num}/{total_steps} — {title}")
    print("=" * 55)


# ── Helper: Print summary ─────────────────────────────────────
def print_summary(child_name, output_dir, pdf_path, duration_seconds):
    print("\n" + "*" * 55)
    print("  BOOK CREATED SUCCESSFULLY!")
    print("*" * 55)
    print(f"  Child name : {child_name}")
    print(f"  Output dir : {output_dir}")
    print(f"  PDF file   : {pdf_path}")
    print(f"  Time taken : {duration_seconds:.1f} seconds")
    print("*" * 55 + "\n")


# ── Step 2 helper: run the chosen cartoonify mode ────────────
def run_cartoonify(face_ready_path, face_cartoon_path):
    """
    Asks the user which cartoonify mode to use, then runs it.

    Option 1 — call the OpenAI API with the STANDARDIZED
               face_ready.png (uses credits)
    Option 2 — reuse an existing face_cartoon.png (free)

    NOTE: option 1 sends face_ready.png (NOT the raw photo) —
    this gives consistent framing across every child's photo,
    which the compositor relies on for reliable face placement
    without needing per-image detection.

    Returns the path to the cartoon face, or None on failure.
    """

    print("\n  Cartoonify options:")
    print("  [1] Call OpenAI API      — fresh Pixar face (uses credits)")
    print("  [2] Reuse existing face  — copy an old result (free)")

    choice = input("  Choice (1 or 2): ").strip()

    # ── Option 1 — call the API with the standardized face ───
    if choice == "1":
        print("  -> Mode: API (OpenAI gpt-image-1)")
        print(f"  -> Sending standardized face to API: {face_ready_path}")
        return cartoonify_face(face_ready_path, face_cartoon_path)

    # ── Option 2 — reuse an existing cartoon face ────────────
    elif choice == "2":
        print("  -> Mode: REUSE (no API, no credits)")

        existing = sorted(
            glob.glob("output/*/face_cartoon.png"),
            key=os.path.getmtime,
            reverse=True
        )

        if not existing:
            print("  ERROR: No existing face_cartoon.png found to reuse.")
            print("  Use option 1 at least once first.")
            return None

        # Show all available cartoon faces to choose from
        print("\n  Available cartoon faces:")
        for i, path in enumerate(existing):
            folder = os.path.basename(os.path.dirname(path))
            print(f"  [{i+1}] {folder}")

        print("\n  Enter number to select")
        print("  (or press ENTER for most recent)")
        print("  (or paste a custom path to any face_cartoon.png)")
        face_choice = input("  Choice: ").strip().strip('"').strip("'")

        if not face_choice:
            selected = existing[0]
        elif face_choice.isdigit():
            idx = int(face_choice) - 1
            selected = existing[idx] if 0 <= idx < len(existing) else existing[0]
        else:
            selected = face_choice if os.path.exists(face_choice) else existing[0]

        # Run the same neck-anchor step used in option 1, so reused
        # faces end up just as consistently framed as fresh API calls —
        # a plain copy would skip normalization entirely.
        print(f"  -> Reusing: {selected}")
        print("  -> Anchoring neck position (same as fresh API calls)...")
        from PIL import Image
        reused_image  = Image.open(selected).convert("RGBA")
        anchored_image = anchor_to_neck_bottom(reused_image)
        os.makedirs(os.path.dirname(face_cartoon_path), exist_ok=True)
        anchored_image.save(face_cartoon_path, format="PNG")
        return face_cartoon_path

    # ── Invalid choice ───────────────────────────────────────
    else:
        print("  ERROR: Invalid choice — please enter 1 or 2.")
        return None


# ── Main Pipeline ─────────────────────────────────────────────
def run_pipeline(child_name, photo_path):
    """
    Runs the full Storyphy book creation pipeline.
    """

    start_time = time.time()

    print("\n" + "=" * 55)
    print("  STORYPHY — Personalized Book Creator")
    print("=" * 55)
    print(f"  Child name : {child_name}")
    print(f"  Photo      : {photo_path}")
    print("=" * 55)

    # Create unique output folder
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    order_name = f"{child_name.strip().capitalize()}_{timestamp}"
    output_dir = os.path.join("output", order_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  -> Order folder: {output_dir}")

    # Define paths
    face_ready_path   = os.path.join(output_dir, "face_ready.png")
    face_cartoon_path = os.path.join(output_dir, "face_cartoon.png")
    pages_dir         = os.path.join(output_dir, "pages")
    pdf_path          = os.path.join(output_dir, f"{child_name}_alphabet_book.pdf")

    # ══════════════════════════════════════════════════════════
    # STEP 1 — Face Preparation (crop + remove background)
    # ══════════════════════════════════════════════════════════
    # face_ready.png is now the STANDARDIZED input for Step 2 —
    # fixed 800x800, face always in the same relative position.
    print_banner(1, 5, "Face Detection & Background Removal")

    face_result = prepare_face(photo_path, face_ready_path)

    if not face_result:
        print("\nPIPELINE FAILED at Step 1 — Face Preparation")
        return None

    # ══════════════════════════════════════════════════════════
    # STEP 2 — Cartoonify (asks: API or reuse)
    # ══════════════════════════════════════════════════════════
    print_banner(2, 5, "Cartoonify Face (OpenAI Pixar Style)")

    # Pass the STANDARDIZED face_ready_path, NOT the raw photo —
    # gives consistent framing in the AI's output, which the
    # compositor relies on for detection-free face placement
    cartoon_result = run_cartoonify(face_ready_path, face_cartoon_path)

    if not cartoon_result:
        print("\nPIPELINE FAILED at Step 2 — Cartoonify")
        return None

    # ══════════════════════════════════════════════════════════
    # STEP 3 — Compose pages
    # ══════════════════════════════════════════════════════════
    print_banner(3, 5, "Composing All 28 Pages")

    pages_result = compose_all_pages(
        cartoon_face_path = face_cartoon_path,
        child_name        = child_name,
        output_dir        = pages_dir
    )

    if not pages_result:
        print("\nPIPELINE FAILED at Step 3 — Page Composition")
        return None

    # ══════════════════════════════════════════════════════════
    # STEP 4 — Add text
    # ══════════════════════════════════════════════════════════
    print_banner(4, 5, "Adding Rhyme Text to Pages")

    # Dynamically load add_text.py from the tools/ folder
    spec = importlib.util.spec_from_file_location(
        "add_text",
        os.path.join("tools", "add_text.py")
    )
    add_text_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(add_text_mod)
    add_text_to_page = add_text_mod.add_text_to_page

    # Load zones
    with open("config/zones.json", "r") as f:
        zones = json.load(f)

    # Process all composed pages
    page_files = sorted(glob.glob(os.path.join(pages_dir, "*.png")))
    print(f"  -> Adding text to {len(page_files)} pages...")

    for page_path in page_files:
        filename = os.path.basename(page_path)
        parts    = filename.replace(".png", "").split("_", 1)
        page_key = parts[1] if len(parts) > 1 else parts[0]
        add_text_to_page(page_path, page_key, child_name, zones)

    print("  -> Text added to all pages!\n")

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
        print("\nPIPELINE FAILED at Step 5 — PDF Assembly")
        return None

    duration = time.time() - start_time
    print_summary(child_name, output_dir, pdf_path, duration)

    return pdf_path


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  STORYPHY — Personalized Book Creator")
    print("=" * 55)

    # Get child's name
    child_name = input("\n  Enter child's name: ").strip()
    if not child_name:
        print("  ERROR: Name cannot be empty.")
        sys.exit(1)

    # Get photo path
    print("\n  Enter path to child's photo.")
    print("  (or press ENTER to use input/test_child.jpg)")
    photo_input = input("  Photo path: ").strip()

    if not photo_input:
        photo_path = "input/test_child.jpg"
    else:
        photo_path = photo_input.strip('"').strip("'")

    if not os.path.exists(photo_path):
        print(f"\n  ERROR: Photo not found at: {photo_path}")
        sys.exit(1)

    print(f"\n  Name  : {child_name}")
    print(f"  Photo : {photo_path}")

    # Confirm before running
    confirm = input("\n  Start creating the book? (y/n): ").strip().lower()
    if confirm != "y":
        print("  -> Cancelled.")
        sys.exit(0)

    # Run pipeline
    result = run_pipeline(child_name, photo_path)
    if not result:
        print("\nBook creation failed.")
        sys.exit(1)