# ============================================================
# preview.py
# STORYPHY — Helper Tool: Preview Composed Pages
# ============================================================
# This tool lets you preview how each page looks after
# composition — face placement, rhyme text, everything.
#
# How to use:
#   python tools/preview.py
#
# Controls:
#   → RIGHT ARROW or D : next page
#   → LEFT ARROW or A  : previous page
#   → R                : rerun compositor for this page only
#   → Q                : quit
# ============================================================

print("▶ Running preview.py — Page Preview Tool")

import cv2
import os
import sys
import json
import glob

# ── Add src/ to path so we can import compositor ─────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Configuration ────────────────────────────────────────────
PAGES_DIR     = "output/pages"
ZONES_FILE    = "config/zones.json"
WINDOW_NAME   = "Storyphy Preview"
DISPLAY_SIZE  = 700   # max width/height for display window


def load_pages():
    """Load all composed page images from output/pages/"""
    pages = sorted(glob.glob(os.path.join(PAGES_DIR, "*.png")))
    return pages


def display_page(image_path, current_index, total_pages):
    """
    Loads and displays a single page with navigation info.
    Returns the loaded image.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ✖ Could not load: {image_path}")
        return None

    img_h, img_w = img.shape[:2]

    # Scale down for display
    scale     = min(DISPLAY_SIZE / img_w, DISPLAY_SIZE / img_h, 1.0)
    display_w = int(img_w * scale)
    display_h = int(img_h * scale)
    display   = cv2.resize(img, (display_w, display_h))

    # Get page name from filename
    page_name = os.path.basename(image_path).replace(".png", "")

    # ── Overlay navigation info ───────────────────────────────
    # Dark bar at bottom for info
    bar_h = 60
    cv2.rectangle(
        display,
        (0, display_h - bar_h),
        (display_w, display_h),
        (30, 30, 30),
        -1
    )

    # Page name
    cv2.putText(
        display,
        f"{page_name}",
        (12, display_h - bar_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1
    )

    # Page counter
    counter_text = f"Page {current_index + 1} of {total_pages}"
    cv2.putText(
        display,
        counter_text,
        (12, display_h - bar_h + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (180, 180, 180), 1
    )

    # Navigation hint
    hint = "← A/LEFT: prev  |  D/RIGHT: next  |  Q: quit"
    hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(
        display,
        hint,
        (display_w - hint_size[0] - 10, display_h - bar_h + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (150, 150, 150), 1
    )

    return display


def run_preview():
    """Main preview loop."""

    print("\n════════════════════════════════════════")
    print("  STORYPHY — Page Preview Tool")
    print("════════════════════════════════════════")

    # ── Check if pages exist ──────────────────────────────────
    # First check if there are order folders
    output_dir = "output"
    order_folders = [
        f for f in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, f))
        and f != "pages"  # skip old direct pages folder
    ]

    if order_folders:
        order_folders.sort(reverse=True)  # newest first
        most_recent = order_folders[0]

        print("\n  Enter the order folder path to preview.")
        print("  (or drag-and-drop the folder into the terminal)")
        print(f"  (or press ENTER for most recent: {most_recent})")
        choice = input("  Path: ").strip().strip('"').strip("'")

        if not choice:
            # Press ENTER — use most recent order
            pages_dir = os.path.join(output_dir, most_recent, "pages")
            print(f"\n  → Previewing most recent: {most_recent}")
        else:
            # User gave a path — figure out the pages folder
            # Accept either the order folder OR its pages subfolder
            if os.path.basename(choice.rstrip("/\\")) == "pages":
                pages_dir = choice
            else:
                pages_dir = os.path.join(choice, "pages")

            if not os.path.isdir(pages_dir):
                print(f"  ✖ No 'pages' folder found at: {pages_dir}")
                return

            print(f"\n  → Previewing: {pages_dir}")

    else:
        # Fall back to default pages dir
        pages_dir = PAGES_DIR

    # ── Load pages ────────────────────────────────────────────
    pages = sorted(glob.glob(os.path.join(pages_dir, "*.png")))

    if not pages:
        print(f"\n  ✖ No pages found in {pages_dir}")
        print("    Run main.py first to generate pages.")
        return

    print(f"  ✔ Found {len(pages)} pages")
    print("\n  Controls:")
    print("  → RIGHT ARROW or D : next page")
    print("  → LEFT ARROW  or A : previous page")
    print("  → Q               : quit\n")

    # ── Setup window ──────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    current_index = 0
    total_pages   = len(pages)

    while True:
        # Display current page
        display = display_page(
            pages[current_index],
            current_index,
            total_pages
        )

        if display is not None:
            # Resize window to fit content
            h, w = display.shape[:2]
            cv2.resizeWindow(WINDOW_NAME, w, h)
            cv2.imshow(WINDOW_NAME, display)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:
            # Q or ESC — quit
            print("  → Closing preview.")
            break

        elif key == ord('d') or key == 83:
            # D or RIGHT ARROW — next page
            current_index = (current_index + 1) % total_pages
            print(f"  → Page {current_index + 1}/{total_pages}: {os.path.basename(pages[current_index])}")

        elif key == ord('a') or key == 81:
            # A or LEFT ARROW — previous page
            current_index = (current_index - 1) % total_pages
            print(f"  → Page {current_index + 1}/{total_pages}: {os.path.basename(pages[current_index])}")

    cv2.destroyAllWindows()
    print("  ✔ Preview closed.\n")


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    run_preview()



