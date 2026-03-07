"""
Generate CLI help screenshots for OllamaAid.
Produces .txt (always) and .png (if Pillow available) under images/.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

PYTHON = sys.executable
TOOL = "ollama-aid"
MODULE = "ollama_aid"

# Subcommands to capture
SUBCOMMANDS = [
    None,       # main --help
    "list",
    "export",
    "import",
    "delete",
    "update",
    "info",
    "trends",
    "test",
    "run",
    "resolve",
]

BG_COLOR = (30, 30, 30)
FG_COLOR = (204, 204, 204)
TITLE_BG = (50, 50, 50)
FONT_SIZE = 14
PADDING = 16
TITLE_HEIGHT = 32


def capture_help(subcmd=None):
    args = [PYTHON, "-m", MODULE]
    if subcmd:
        args += [subcmd]
    args += ["--help"]
    env = os.environ.copy()
    env["COLUMNS"] = "100"
    result = subprocess.run(args, capture_output=True, text=True, timeout=15, env=env)
    return result.stdout.strip()


def save_txt(name, text):
    path = IMAGES / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"  -> {path}")


def save_png(name, text, title_cmd):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  [skip png] Pillow not installed")
        return

    # Try to load a monospace font
    font = None
    font_candidates = [
        "DejaVuSansMono.ttf",
        "Consolas",
        "Courier New",
        "Liberation Mono",
        "monospace",
    ]
    for fname in font_candidates:
        try:
            font = ImageFont.truetype(fname, FONT_SIZE)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    lines = text.split("\n")
    # Measure
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    char_bbox = draw.textbbox((0, 0), "M", font=font)
    char_w = char_bbox[2] - char_bbox[0]
    line_h = char_bbox[3] - char_bbox[1] + 4

    max_cols = max((len(line) for line in lines), default=80)
    img_w = max(max_cols * char_w + PADDING * 2, 600)
    img_h = TITLE_HEIGHT + len(lines) * line_h + PADDING * 2

    img = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([(0, 0), (img_w, TITLE_HEIGHT)], fill=TITLE_BG)
    draw.text((PADDING, (TITLE_HEIGHT - line_h) // 2), f"$ {title_cmd}", fill=FG_COLOR, font=font)

    # Body
    y = TITLE_HEIGHT + PADDING // 2
    for line in lines:
        draw.text((PADDING, y), line, fill=FG_COLOR, font=font)
        y += line_h

    path = IMAGES / f"{name}.png"
    img.save(str(path))
    print(f"  -> {path}")


def main():
    print("Generating CLI help screenshots for OllamaAid...")
    for subcmd in SUBCOMMANDS:
        if subcmd:
            name = f"{TOOL}_{subcmd}_help"
            title_cmd = f"{TOOL} {subcmd} --help"
        else:
            name = f"{TOOL}_help"
            title_cmd = f"{TOOL} --help"

        print(f"\n[{name}]")
        try:
            text = capture_help(subcmd)
        except Exception as e:
            print(f"  [error] {e}")
            continue

        save_txt(name, text)
        save_png(name, text, title_cmd)

    print("\nDone!")


if __name__ == "__main__":
    main()
