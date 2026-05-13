# ============================================================
# fix_rhyme_position.py
# STORYPHY — Fix rhyme text position for 1254x1254 templates
# ============================================================

print("▶ Fixing rhyme positions in zones.json...")

import json
import os

ZONES_FILE = "config/zones.json"

# ── Updated style for 1254x1254 templates ───────────────────
RHYME_STYLE = {
    "font_size":    42,             # bigger font for larger template
    "font_color":   [30, 30, 30],   # dark gray
    "y_position":   1120,           # adjusted for 1254px tall template
    "align":        "center",
    "line_spacing": 55,             # more spacing for larger font
    "stroke_color": [255, 255, 255],
    "stroke_width": 3,
}

# ── Load zones.json ──────────────────────────────────────────
with open(ZONES_FILE, "r") as f:
    zones = json.load(f)

print(f"  → Loaded {len(zones)} pages")

# ── Update rhyme style for every page ────────────────────────
updated = 0
for page_key, page_data in zones.items():
    if "rhyme" in page_data:
        zones[page_key]["rhyme_style"] = RHYME_STYLE.copy()
        updated += 1
        print(f"  ✔ Updated: {page_key}")

# ── Save ─────────────────────────────────────────────────────
with open(ZONES_FILE, "w") as f:
    json.dump(zones, f, indent=2)

print(f"\n  ✔ Updated {updated} pages")
print("  ✔ Saved to zones.json")
print("\n  Now run: python main.py to regenerate the book!")