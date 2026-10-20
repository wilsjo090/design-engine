"""Put a "Design Engine" shortcut (with its own icon) on the desktop.

    python tools/create_shortcut.py
"""
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "tools" / "design_engine.ico"


def make_icon(path):
    """Cyan contour lines on a dark rounded square, like the editor's field view."""
    n = 256
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([8, 8, n - 8, n - 8], radius=48, fill=(31, 33, 37, 255), outline=(70, 74, 82, 255), width=4)
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([20, 20, n - 20, n - 20], radius=38, fill=255)
    lines = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    for k in range(-3, 9):  # wavy ridges whose spacing changes across the icon
        pts = []
        for i in range(0, n + 1, 4):
            x = i
            y = 30 + k * 26 + 14 * math.sin(i / 34 + k * 0.6) + 0.0009 * (i - 128) ** 2
            pts.append((x, y))
        ld.line(pts, fill=(0, 229, 229, 255), width=9, joint="curve")
    img.paste(lines, (0, 0), Image.composite(lines, Image.new("RGBA", (n, n)), mask))
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def create_shortcut():
    ps = f"""
$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'Design Engine.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
$s.TargetPath = '{ROOT / "start.bat"}'
$s.WorkingDirectory = '{ROOT}'
$s.IconLocation = '{ICON},0'
$s.Description = 'Start the Design Engine node editor'
$s.WindowStyle = 1
$s.Save()
Write-Output $link
"""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, check=True)
    return out.stdout.strip()


if __name__ == "__main__":
    make_icon(ICON)
    print(f"icon: {ICON}")
    print(f"shortcut: {create_shortcut()}")
