# ============================================================
# app.py (Flask version)
# STORYPHY — Internal Team Tool
# ============================================================
# Two views:
#   1. Create New Book — upload photo + name, run the pipeline
#   2. Order History    — browse/download past books
#
# Run with:  python app.py
# Then open: http://127.0.0.1:5000
# (from inside D:\storyphy_engine_v3)
# ============================================================

import os
import sys
import glob
import json
import shutil
from datetime import datetime
import importlib.util

from flask import (
    Flask, render_template, request, redirect,
    url_for, send_file, flash
)
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from face_prep import prepare_face
from cartoonify import cartoonify_face
from compositor import compose_all_pages
from pdf_builder import build_pdf

ZONES_FILE = "config/zones.json"
UPLOAD_DIR = "uploads_tmp"

app = Flask(__name__, template_folder="web_templates")
app.secret_key = "storyphy-internal-tool"  # only used for flash messages

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Helper: load add_text_to_page dynamically (same as main.py) ─────
def get_add_text_fn():
    spec = importlib.util.spec_from_file_location(
        "add_text", os.path.join("tools", "add_text.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.add_text_to_page


# ── Helper: list all past orders ─────────────────────────────
def list_orders():
    order_dirs = sorted(
        glob.glob("output/*"),
        key=os.path.getmtime,
        reverse=True,
    )
    orders = []
    for d in order_dirs:
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        pdf_matches = glob.glob(os.path.join(d, "*_alphabet_book.pdf"))
        cover = os.path.join(d, "pages", "01_cover_front.png")
        page_files = sorted(glob.glob(os.path.join(d, "pages", "*.png")))
        orders.append({
            "folder": d,
            "name": name,
            "child": name.split("_")[0],
            "pdf": pdf_matches[0] if pdf_matches else None,
            "cover": cover if os.path.exists(cover) else None,
            "pages": page_files,
            "modified": datetime.fromtimestamp(os.path.getmtime(d)),
        })
    return orders


# ── Helper: list existing cartoon faces for "reuse" option ──
def list_existing_faces():
    existing = sorted(
        glob.glob("output/*/face_cartoon.png"),
        key=os.path.getmtime,
        reverse=True,
    )
    return [
        {"path": p, "label": os.path.basename(os.path.dirname(p))}
        for p in existing
    ]


# ============================================================
# ROUTE: Home — redirect to create page
# ============================================================
@app.route("/")
def home():
    return redirect(url_for("create_book"))


# ============================================================
# ROUTE: Create New Book (GET = show form, POST = run pipeline)
# ============================================================
@app.route("/create", methods=["GET", "POST"])
def create_book():
    if request.method == "GET":
        return render_template(
            "create.html", existing_faces=list_existing_faces()
        )

    # ── POST — form submitted ─────────────────────────────────
    child_name = request.form.get("child_name", "").strip()
    mode = request.form.get("mode")  # "generate" or "reuse"
    reuse_select = request.form.get("reuse_select", "")
    reuse_path_input = request.form.get("reuse_path_input", "").strip().strip('"').strip("'")
    reuse_path = reuse_path_input if reuse_path_input else reuse_select
    photo_file = request.files.get("photo")
    photo_path_input = request.form.get("photo_path", "").strip().strip('"').strip("'")

    if not child_name:
        flash("Please enter the child's name.")
        return redirect(url_for("create_book"))

    has_upload = photo_file and photo_file.filename != ""
    has_pasted_path = bool(photo_path_input)

    if mode == "generate" and not has_upload and not has_pasted_path:
        flash("Please upload a photo or paste a file path.")
        return redirect(url_for("create_book"))

    if mode == "generate" and has_pasted_path and not has_upload:
        if not os.path.exists(photo_path_input):
            flash(f"File not found at: {photo_path_input}")
            return redirect(url_for("create_book"))

    if mode == "reuse" and not reuse_path:
        flash("Please select or paste a cartoon face image.")
        return redirect(url_for("create_book"))

    if mode == "reuse" and not os.path.exists(reuse_path):
        flash(f"File not found at: {reuse_path}")
        return redirect(url_for("create_book"))

    # ── Set up order folder ──────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    order_name = f"{child_name.capitalize()}_{timestamp}"
    output_dir = os.path.join("output", order_name)
    os.makedirs(output_dir, exist_ok=True)

    face_ready_path = os.path.join(output_dir, "face_ready.png")
    face_cartoon_path = os.path.join(output_dir, "face_cartoon.png")
    pages_dir = os.path.join(output_dir, "pages")
    pdf_path = os.path.join(output_dir, f"{child_name}_alphabet_book.pdf")

    log = []

    try:
        if mode == "reuse":
            shutil.copy(reuse_path, face_cartoon_path)
            log.append(f"Reused existing cartoon face: {reuse_path}")
        else:
            photo_path = os.path.join(output_dir, "input_photo.png")
            if has_upload:
                photo_file.save(photo_path)
            else:
                shutil.copy(photo_path_input, photo_path)

            log.append("Step 1/5 — Face detection & background removal...")
            if not prepare_face(photo_path, face_ready_path):
                flash("Failed at Step 1 — Face Prep. Check terminal for details.")
                return redirect(url_for("create_book"))

            log.append("Step 2/5 — Cartoonifying face (OpenAI API)...")
            result = cartoonify_face(photo_path, face_cartoon_path, auto_confirm=True)
            if not result:
                flash("Failed at Step 2 — Cartoonify. Check API key / daily spend limit in terminal.")
                return redirect(url_for("create_book"))

        log.append("Step 3/5 — Composing 28 pages...")
        pages_result = compose_all_pages(
            cartoon_face_path=face_cartoon_path,
            child_name=child_name,
            output_dir=pages_dir,
        )
        if not pages_result:
            flash("Failed at Step 3 — Page Composition. Check terminal for details.")
            return redirect(url_for("create_book"))

        log.append("Step 4/5 — Adding rhyme text...")
        add_text_to_page = get_add_text_fn()
        with open(ZONES_FILE, "r") as f:
            zones = json.load(f)

        for page_path in sorted(glob.glob(os.path.join(pages_dir, "*.png"))):
            filename = os.path.basename(page_path)
            parts = filename.replace(".png", "").split("_", 1)
            page_key = parts[1] if len(parts) > 1 else parts[0]
            add_text_to_page(page_path, page_key, child_name, zones)

        log.append("Step 5/5 — Assembling PDF...")
        pdf_result = build_pdf(
            pages_dir=pages_dir, output_pdf_path=pdf_path, child_name=child_name
        )
        if not pdf_result:
            flash("Failed at Step 5 — PDF Assembly. Check terminal for details.")
            return redirect(url_for("create_book"))

    except Exception as e:
        flash(f"Pipeline error: {e}")
        return redirect(url_for("create_book"))

    return redirect(url_for("order_detail", order_name=order_name))


# ============================================================
# ROUTE: Order History — list all past books
# ============================================================
@app.route("/history")
def order_history():
    search = request.args.get("search", "").strip().lower()
    orders = list_orders()
    if search:
        orders = [o for o in orders if search in o["child"].lower()]
    return render_template("history.html", orders=orders, search=search)


# ============================================================
# ROUTE: Single order detail page
# ============================================================
@app.route("/order/<order_name>")
def order_detail(order_name):
    output_dir = os.path.join("output", order_name)
    pages_dir = os.path.join(output_dir, "pages")
    pdf_matches = glob.glob(os.path.join(output_dir, "*_alphabet_book.pdf"))
    page_files = sorted(glob.glob(os.path.join(pages_dir, "*.png")))

    order = {
        "name": order_name,
        "child": order_name.split("_")[0],
        "pdf": pdf_matches[0] if pdf_matches else None,
        "pdf_abspath": os.path.abspath(pdf_matches[0]) if pdf_matches else None,
        "pages": page_files,
        "page_abspaths": [os.path.abspath(p) for p in page_files],
    }
    return render_template("order_detail.html", order=order)


# ============================================================
# ROUTE: Serve files (images / PDFs) from output/ folder
# ============================================================
@app.route("/file/<path:filepath>")
def serve_file(filepath):
    full_path = os.path.join("output", filepath)
    if not os.path.exists(full_path):
        return "File not found", 404
    return send_file(full_path)


@app.route("/download/<path:filepath>")
def download_file(filepath):
    full_path = os.path.join("output", filepath)
    if not os.path.exists(full_path):
        return "File not found", 404
    return send_file(full_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
