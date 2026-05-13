# ============================================================
# pdf_builder.py
# STORYPHY — Step 4: Assemble Final PDF
# ============================================================
# This script takes all 28 composed page images and:
#   1. Sorts them in correct page order
#   2. Assembles them into a single PDF file
#   3. Saves the final print-ready PDF to output/
# ============================================================

print("▶ Running pdf_builder.py — Assembling Final PDF")

import os
import glob
from PIL import Image


# ── Main Function ─────────────────────────────────────────────
def build_pdf(pages_dir, output_pdf_path, child_name):
    """
    Takes all composed page images and assembles them into
    a single print-ready PDF file.

    Args:
        pages_dir       : folder containing all composed page PNGs
        output_pdf_path : where to save the final PDF
        child_name      : used for print messages only
    """

    print(f"  → Looking for pages in: {pages_dir}")
    print(f"  → Output PDF path     : {output_pdf_path}")

    # ── Step 1: Find all page images ─────────────────────────
    # Grab all PNG files in the pages directory
    page_files = glob.glob(os.path.join(pages_dir, "*.png"))

    if not page_files:
        print(f"  ✖ ERROR: No PNG pages found in {pages_dir}")
        print("    Run compositor.py first to generate the pages.")
        return None

    # Sort pages by filename so they are in correct order
    # Files are named 01_cover_front.png, 02_page_A.png etc.
    page_files.sort()

    print(f"  ✔ Found {len(page_files)} pages to assemble")

    # ── Step 2: Load all pages as PIL Images ─────────────────
    print("\n  → Loading all pages...")

    pil_pages = []

    for i, page_path in enumerate(page_files, start=1):
        page_name = os.path.basename(page_path)

        # Open image and convert to RGB
        # PDF doesn't support transparency so RGB is correct here
        img = Image.open(page_path).convert("RGB")
        pil_pages.append(img)

        print(f"  [{i:02d}/{len(page_files)}] Loaded: {page_name} — {img.size}")

    if not pil_pages:
        print("  ✖ ERROR: Could not load any page images.")
        return None

    # ── Step 3: Assemble into PDF ─────────────────────────────
    print(f"\n  → Assembling {len(pil_pages)} pages into PDF...")

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    # The first page is saved separately, rest are appended
    first_page  = pil_pages[0]
    other_pages = pil_pages[1:]

    # Save as PDF — PIL handles multi-page PDF assembly
    first_page.save(
        output_pdf_path,
        format="PDF",
        save_all=True,          # include all pages
        append_images=other_pages,
        resolution=300.0,       # 300 DPI — print quality
        optimize=False          # keep full quality
    )

    # ── Step 4: Verify output ─────────────────────────────────
    if os.path.exists(output_pdf_path):
        # Get file size in MB
        file_size_mb = os.path.getsize(output_pdf_path) / (1024 * 1024)
        print(f"\n  ✔ PDF assembled successfully!")
        print(f"  ✔ Pages    : {len(pil_pages)}")
        print(f"  ✔ File size: {file_size_mb:.1f} MB")
        print(f"  ✔ Saved to : {output_pdf_path}")
        print("  ✔ pdf_builder.py complete!\n")
        return output_pdf_path
    else:
        print("  ✖ ERROR: PDF file was not created.")
        return None


# ── Run directly for testing ──────────────────────────────────
if __name__ == "__main__":

    test_pages_dir  = "output/pages"
    test_child_name = "Emma"
    test_output_pdf = f"output/{test_child_name}_alphabet_book.pdf"

    result = build_pdf(test_pages_dir, test_output_pdf, test_child_name)

    if result:
        print(f"✅ Success! Open '{result}' to preview the full book.")
    else:
        print("❌ Something went wrong. Check the error messages above.")