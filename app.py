"""Start the node editor:  python app.py   (opens http://127.0.0.1:8765 in your browser)

- If the editor is already running, this just opens it in the browser.
- If the editor UI (frontend/src) changed since it was last built, it is rebuilt first (needs Node.js).
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST_INDEX = FRONTEND / "dist" / "index.html"
UI_SOURCES = ["src", "index.html", "package.json", "vite.config.ts", "tsconfig.json"]


def engine_running(url):
    """True if a Design Engine server answers at `url`."""
    try:
        with urllib.request.urlopen(f"{url}/api/catalog", timeout=2) as res:
            return "nodes" in json.loads(res.read())
    except Exception:
        return False


def port_taken(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ui_is_stale():
    """True if the built UI is missing or older than any of its source files."""
    if not DIST_INDEX.exists():
        return True
    built = DIST_INDEX.stat().st_mtime
    for name in UI_SOURCES:
        path = FRONTEND / name
        files = path.rglob("*") if path.is_dir() else [path]
        if any(f.is_file() and f.stat().st_mtime > built for f in files):
            return True
    return False


def build_ui():
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm") or shutil.which("npm")
    if npm is None:
        print("! Node.js (npm) not found, so the editor UI can't be rebuilt.")
        return DIST_INDEX.exists()  # an older build still works
    if not (FRONTEND / "node_modules").exists():
        print("Installing editor UI packages (first time only)...")
        if subprocess.run([npm, "install"], cwd=FRONTEND).returncode != 0:
            return False
    print("Building the editor UI...")
    return subprocess.run([npm, "run", "build"], cwd=FRONTEND).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    ap.add_argument("--no-build", action="store_true", help="never rebuild the editor UI")
    ap.add_argument("--clear-cache", action="store_true", help="empty the SSD cache of computed fields and exit")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    if args.clear_cache:
        from engine.cache import DiskCache
        disk = DiskCache()
        size = disk.size()
        disk.clear()
        print(f"Cleared {size / 2**30:.1f} GB from {disk.folder}")
        return 0

    if engine_running(url):
        print(f"Design Engine is already running: opening {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    if port_taken(args.port):
        print(f"! Port {args.port} is used by another program. Start with a different one, e.g.:  start.bat --port 8766")
        return 1

    if not args.no_build and ui_is_stale() and not build_ui():
        print("! The editor UI could not be built (see above).")
        if not DIST_INDEX.exists():
            return 1
        print("  Starting with the previous build.")

    import uvicorn
    from engine.server import create_app

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Design Engine running at {url}")
    print("Close this window or press Ctrl+C to stop it.")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # required on Windows: worker processes re-import this file
    sys.exit(main())
