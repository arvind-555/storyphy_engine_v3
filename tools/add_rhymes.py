# ============================================================
# add_rhymes.py
# STORYPHY — Helper Tool: Add Rhymes to zones.json
# ============================================================
# Run this ONCE to inject all rhymes into your existing
# zones.json file. Safe to run again — won't overwrite
# face zone coordinates.
# ============================================================

print("▶ Running add_rhymes.py — Adding rhymes to zones.json")

import json
import os

ZONES_FILE = "config/zones.json"

# ── All 26 rhymes with {name} placeholder ───────────────────
RHYMES = {
    "page_A": "A is for Apple, so shiny and red,\n{name} holds it up and nods their head!",
"page_B": "B is for Bear, so fluffy and brown,\n{name} hugs it tight in the whole little town!",
"page_C": "C is for Cat, so tiny and sweet,\n{name} pets it gently — what a treat!",
"page_D": "D is for Duck, that goes waddle waddle,\n{name} laughs along at their funny model!",
"page_E": "E is for Elephant, big and gray,\n{name} stands beside it on a sunny day!",
"page_F": "F is for Flower, so pretty and bright,\n{name} holds a bouquet — what a sight!",
"page_G": "G is for Giraffe, so tall and grand,\n{name} reaches up to hold its hand!",
"page_H": "H is for House, so cozy and bright,\n{name} spreads arms wide with pure delight!",
"page_I": "I is for Ice Cream, cold and sweet,\n{name} holds a cone — what a yummy treat!",
"page_J": "J is for Jar, full of candy so bright,\n{name} peeks inside with pure delight!",
"page_K": "K is for Kite, soaring up high,\n{name} holds the string watching it fly!",
"page_L": "L is for Lion, fluffy and small,\n{name} sits beside the friendliest of all!",
"page_M": "M is for Moon, glowing at night,\n{name} looks up at its soft silver light!",
"page_N": "N is for Nature, so colorful and free,\n{name} walks the path as happy as can be!",
"page_O": "O is for Ocean, sparkling and blue,\n{name} sits beside and enjoys the view!",
"page_P": "P is for Penguin, in black and white,\n{name} stands beside it — what a sight!",
"page_Q": "Q is for Quiz, who knows the answer today,\n{name} raises a hand and shouts hooray!",
"page_R": "R is for Rainbow, after the rain,\n{name} spreads arms wide and smiles again!",
"page_S": "S is for Sun, rising so bright,\n{name} opens arms wide to greet the light!",
"page_T": "T is for Train, going choo choo choo,\n{name} waves excitedly — woo hoo hoo!",
"page_U": "U is for Umbrella, in the rain,\n{name} splashes in puddles again and again!",
"page_V": "V is for Vegetables, fresh from the ground,\n{name} fills up a basket — the best to be found!",
"page_W": "W is for Watermelon, juicy and sweet,\n{name} takes a big slice — what a summer treat!",
"page_X": "X is for Xylophone, ding ding ding,\n{name} taps every bar and loves to sing!",
"page_Y": "Y is for Yacht, sailing the sea,\n{name} stands at the bow as happy as can be!",
"page_Z": "Z is for Zebra, striped black and white,\n{name} sits beside it — what a beautiful sight!",
"cover_front": "",   # no rhyme on covers
"cover_back": "",    # no rhyme on covers
}

# ── Text styling config for the rhyme overlay ────────────────
# You can tweak these values after seeing the first output
RHYME_STYLE = {
    "font_size":    36,           # text size in pixels
    "font_color":   [50, 50, 50], # dark gray [R, G, B]
    "y_position":   880,          # vertical position from top (1024px tall page)
    "align":        "center",     # center aligned on page
    "line_spacing": 48,           # pixels between lines
    "stroke_color": [255, 255, 255],  # white outline around text (improves readability)
    "stroke_width": 2,                # thickness of the white outline
}


def add_rhymes():

    # Load existing zones.json
    if not os.path.exists(ZONES_FILE):
        print(f"  ✖ ERROR: {ZONES_FILE} not found!")
        print("    Run zone_finder.py first.")
        return

    with open(ZONES_FILE, "r") as f:
        zones = json.load(f)

    print(f"  → Loaded zones.json with {len(zones)} pages")

    # Add rhyme + style to each page
    updated = 0
    for page_key, rhyme_text in RHYMES.items():
        if page_key not in zones:
            print(f"  ⚠ {page_key} not found in zones.json — skipping")
            continue

        # Add rhyme text
        zones[page_key]["rhyme"] = rhyme_text

        # Add text styling config
        zones[page_key]["rhyme_style"] = RHYME_STYLE.copy()

        updated += 1

    # Save updated zones.json
    with open(ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=2)

    print(f"  ✔ Added rhymes to {updated} pages")
    print(f"  ✔ Saved to: {ZONES_FILE}")
    print("  ✔ add_rhymes.py complete!\n")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    add_rhymes()