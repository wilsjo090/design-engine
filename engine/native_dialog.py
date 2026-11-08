"""Windows' own Open / Save dialog for the editor (the Explorer dialog: Quick access, recent files, search).

The engine runs on the PC the editor is used on, so `/api/file_dialog` shows the dialog there and answers
the real path. It runs in a process of its own (this file as a script, tkinter's file dialogs, which on
Windows are the system's IFileDialog): shell extensions load into whatever process shows the dialog, and
one that crashes must not take the engine and its caches with it. The dialog's hidden owner window is
topmost, so the dialog opens over the browser, which the engine's process cannot bring to the front.

Run as a script it reads {mode, title, exts, dir, name} as JSON on stdin and prints {"path": ...} (None when
cancelled). Keep imports here light: the server imports this module too.
"""
import json
import os
import subprocess
import sys
import threading

_open = threading.Lock()


class Unavailable(RuntimeError):
    """No system dialog here (not Windows, no tkinter, the helper failed): the editor uses its own."""


class Busy(RuntimeError):
    """A dialog is already open."""


def ask(mode, title="", exts=(), folder="", name=""):
    """Show the dialog and wait for it: the chosen path, or None when cancelled."""
    if sys.platform != "win32":
        raise Unavailable("the system file dialog is only used on Windows")
    if not _open.acquire(blocking=False):
        raise Busy("a file dialog is already open")
    try:
        request = {"mode": "save" if mode == "save" else "open", "title": str(title or ""),
                   "exts": [str(e) for e in exts or ()], "dir": str(folder or ""), "name": str(name or "")}
        try:
            done = subprocess.run([sys.executable, "-I", os.path.abspath(__file__)],
                                  input=json.dumps(request), capture_output=True, text=True,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            raise Unavailable(f"could not start the file dialog: {e}") from e
        try:
            answer = json.loads(done.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            raise Unavailable(f"the file dialog failed: {(done.stderr or done.stdout).strip()[-300:]}") from None
        if answer.get("error"):
            raise Unavailable(answer["error"])
        return answer.get("path") or None
    finally:
        _open.release()


def _label(exts):
    names = []
    for e in exts:
        n = e.lstrip(".").upper()
        if n not in names:
            names.append(n)
    return " / ".join(names) + " files"


def _show(req):
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # sharp, not a scaled bitmap, on high-DPI screens
    except (AttributeError, OSError):
        pass
    import tkinter as tk
    from tkinter import filedialog

    exts = [e if e.startswith(".") else "." + e for e in req.get("exts") or []]
    kinds = ([(_label(exts), tuple("*" + e for e in exts))] if exts else []) + [("All files", "*.*")]
    folder = req.get("dir") or ""
    while folder and not os.path.isdir(folder) and os.path.dirname(folder) != folder:
        folder = os.path.dirname(folder)  # an export's folder that does not exist yet: the nearest one that does
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    opts = {"parent": root, "title": req.get("title") or None, "filetypes": kinds,
            "initialdir": folder if folder and os.path.isdir(folder) else None,
            "initialfile": req.get("name") or None}
    if req.get("mode") == "save":
        path = filedialog.asksaveasfilename(defaultextension=exts[0] if exts else "", confirmoverwrite=True, **opts)
    else:
        path = filedialog.askopenfilename(**opts)
    root.destroy()
    return os.path.normpath(path) if isinstance(path, str) and path else None


def main():
    try:
        out = {"path": _show(json.loads(sys.stdin.read() or "{}"))}
    except Exception as e:  # noqa: BLE001 - every failure means "use the editor's own dialog"
        out = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out))  # ASCII escapes: the path survives any console code page


if __name__ == "__main__":
    main()
