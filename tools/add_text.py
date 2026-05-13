# ============================================================
# add_text.py
# STORYPHY — Overlay rhyme text inside cloud zone on each page
# ============================================================
# This script:
#   1. Reads cloud/text zone coordinates from zones.json
#   2. Places rhyme text centered inside the cloud
#   3. Saves updated pages ready for PDF
# ============================================================

print("▶ Running add_text.py — Overlaying rhyme text on pages")

from PIL import Image, ImageDraw, ImageFont
import json
import os
import glob

# ── Configuration ────────────────────────────────────────────
ZONES_FILE    = "config/zones.json"
FONT_PATH     = "fonts/Nunito-Bold.ttf"
FALLBACK_FONT = "fonts/Nunito-Bold.ttf"

# Text styling
FONT_SIZE    = 70            # increased size
TEXT_COLOR   = (40,  40,  80)   # dark navy — readable on white cloud
STROKE_COLOR = (255, 255, 255)  # white outline
STROKE_WIDTH = 2
LINE_SPACING = 85               # increased spacing for bigger font


# ── Helper: Load font ────────────────────────────────────────
def load_font(size):
    for path in [FONT_PATH, FALLBACK_FONT]:
        if os.path.exists(path):
            print(f"  ✔ Font loaded: {path}")
            return ImageFont.truetype(path, size)
    print("  ⚠ No custom font — using default")
    return ImageFont.load_default()


# ── Helper: Draw outlined text ────────────────────────────────
def draw_outlined_text(draw, text, x, y, font,
                        text_color, stroke_color, stroke_width):
    # Draw stroke
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text,
                      font=font, fill=stroke_color)
    # Draw main text
    draw.text((x, y), text, font=font, fill=text_color)


# ── Main Function ─────────────────────────────────────────────
def add_text_to_page(page_path, page_key, child_name, zones):
    """
    Places rhyme text centered inside the cloud zone
    defined in zones.json for this page.
    """

    # Get zone data
    zone       = zones.get(page_key, {})
    rhyme_text = zone.get("rhyme", "")
    text_zone  = zone.get("text_zone")

    # Skip if no rhyme or no text zone defined
    if not rhyme_text:
        print(f"  → {page_key}: no rhyme — skipping")
        return

    if not text_zone:
        print(f"  ⚠ {page_key}: no text_zone in zones.json — skipping")
        print(f"    Run zone_finder.py option 2 to set cloud zones")
        return
    
    print(f"  → rhyme    : {rhyme_text[:40]}...")
    print(f"  → text_zone: {text_zone}")
    print(f"  → page_path: {page_path}")

    # Load page
    page         = Image.open(page_path).convert("RGBA")
    img_w, img_h = page.size

    # ── Get cloud zone coordinates ────────────────────────────
    tz_x = text_zone["x"]
    tz_y = text_zone["y"]
    tz_w = text_zone["w"]
    tz_h = text_zone["h"]

    print(f"  → Cloud zone: x={tz_x}, y={tz_y}, w={tz_w}, h={tz_h}")

    # ── Prepare text ──────────────────────────────────────────
    formatted_name = child_name.strip().capitalize()
    rhyme_filled   = rhyme_text.replace("{name}", formatted_name)
    lines          = rhyme_filled.split("\n")

    # ── Auto fit font size to cloud zone ─────────────────────
    # Reduce font size until text fits inside cloud width
    font_size = FONT_SIZE
    font      = load_font(font_size)
    draw      = ImageDraw.Draw(page)

    while font_size > 40:
        font      = load_font(font_size)
        max_width = max(
            draw.textbbox((0, 0), line, font=font)[2]
            for line in lines
        )
        if max_width <= tz_w - 40:  # 20px padding each side
            break
        font_size -= 2
        print(f"  → Font too wide — reducing to {font_size}px")

    print(f"  ✔ Final font size: {font_size}px")

    # ── Calculate text position ───────────────────────────────
    total_text_h = len(lines) * LINE_SPACING

    # Center text block vertically inside cloud zone
    text_start_y = tz_y + (tz_h - total_text_h) // 2

    # ── Draw each line ────────────────────────────────────────
    for i, line in enumerate(lines):
        bbox       = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center horizontally inside cloud zone
        x = tz_x + (tz_w - text_width) // 2
        y = text_start_y + (i * LINE_SPACING)

        draw_outlined_text(
            draw, line, x, y,
            font,
            TEXT_COLOR,
            STROKE_COLOR,
            STROKE_WIDTH
        )

        print(f"  ✔ Line {i+1}: '{line[:50]}' at ({x}, {y})")

    # ── Save page ─────────────────────────────────────────────
    page.convert("RGB").save(page_path, format="PNG", dpi=(300, 300))
    print(f"  ✔ Saved: {page_path}\n")


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n════════════════════════════════════════")
    print("  STORYPHY — Add Text to Pages")
    print("════════════════════════════════════════")

    # Load zones
    with open(ZONES_FILE, "r") as f:
        zones = json.load(f)

    # Find all order folders
    order_folders = sorted(
        glob.glob("output/*/pages/"),
        key=os.path.getmtime,
        reverse=True
    )

    if not order_folders:
        print("❌ No order folders found. Run main.py first.")
        exit(1)

    # Show available orders
    print("\n  Available orders:")
    for i, folder in enumerate(order_folders):
        order_name = folder.split(os.sep)[-3]
        page_count = len(glob.glob(os.path.join(folder, "*.png")))
        print(f"  [{i+1}] {order_name} ({page_count} pages)")

    print("\n  Select order (or ENTER for most recent):")
    choice = input("  Choice: ").strip()

    if not choice:
        pages_dir = order_folders[0]
    else:
        try:
            pages_dir = order_folders[int(choice) - 1]
        except (ValueError, IndexError):
            pages_dir = order_folders[0]

    # Get child name from folder
    order_name = pages_dir.split(os.sep)[-3]
    child_name = order_name.split("_")[0]

    print(f"\n  → Order     : {order_name}")
    print(f"  → Child name: {child_name}")
    print(f"  → Pages dir : {pages_dir}\n")

    # Process all pages
    page_files = sorted(glob.glob(os.path.join(pages_dir, "*.png")))
    print(f"  → Processing {len(page_files)} pages...\n")

    for page_path in page_files:
        filename = os.path.basename(page_path)
        parts    = filename.replace(".png", "").split("_", 1)
        page_key = parts[1] if len(parts) > 1 else parts[0]

        print(f"  [{filename}]")
        add_text_to_page(page_path, page_key, child_name, zones)

    # Rebuild PDF
    print("════════════════════════════════════════")
    print("  → Rebuilding PDF...")

    import sys
    sys.path.append("src")
    from pdf_builder import build_pdf

    pdf_path = os.path.join(
        "output", order_name,
        f"{child_name}_alphabet_book.pdf"
    )

    build_pdf(pages_dir, pdf_path, child_name)

    print(f"\n✅ Done! PDF saved to: {pdf_path}")