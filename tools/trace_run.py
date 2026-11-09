"""Run a recipe headless with the timeline trace on (engine/trace.py), as the editor runs it, and write the report.

    python tools/trace_run.py recipes/wave_offset.json --quality draft medium --workers 4
    python tools/trace_run.py recipes/lamp_noise.json --quality final --set amplitude.value=0.5
    python tools/trace_run.py recipes/hex_studs.json --mode export
    python tools/trace_run.py recipes/topo_wall_bracket_bolted.json --optimize topology --quality medium

The run goes through the editor's own server code (`server.Session`: `build_run_graph` with the preview
quality, the hidden view meshers of the recipe's visible nodes, the views packed and coloured), with the
same number of worker processes as the editor unless --workers says otherwise. Each quality given is one
run, in that order, in one process: later ones find the earlier ones' results and caches warm, as
refining in the editor does. Every run's folder gets a report.html; its summary is printed.

Paths in the recipe are read from the project folder, as the editor does. Traces go to
%LOCALAPPDATA%\\DesignEngine\\trace (DESIGN_ENGINE_TRACE_DIR or --trace-dir to change).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("--quality", nargs="+", default=["draft"], choices=["draft", "medium", "final"],
                    help="preview quality of each run, in order (default: draft)")
    ap.add_argument("--mode", default="preview", choices=["preview", "export"],
                    help="preview (the editor's views of the visible nodes) or export (writes the Export nodes' files)")
    ap.add_argument("--optimize", metavar="NODE", help="run an Optimize of this node (Topology or Frame Optimization)")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: as the editor, "
                                                               "max(2, min(8, cores / 2)))")
    ap.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=VALUE", help="override an input")
    ap.add_argument("--show", metavar="NODE,NODE", help="the nodes shown (default: those visible in the recipe)")
    ap.add_argument("--trace-dir", help="where run folders go (default: DESIGN_ENGINE_TRACE_DIR, else "
                                        "%%LOCALAPPDATA%%\\DesignEngine\\trace)")
    args = ap.parse_args()

    recipe_path = Path(args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    for item in args.set:
        target, _, value = item.partition("=")
        node_id, _, port = target.partition(".")
        if not port or not value or node_id not in recipe.get("nodes", {}):
            ap.error(f"--set needs NODE.INPUT=VALUE with a node of the recipe, got '{item}'")
        recipe["nodes"][node_id].setdefault("inputs", {})[port] = parse_value(value)
    visible = (args.show.split(",") if args.show else
               [n for n, spec in recipe.get("nodes", {}).items() if spec.get("visible")])

    # before the engine is imported (it reads these once); the worker processes inherit them
    os.environ["DESIGN_ENGINE_TRACE"] = "1"
    if args.trace_dir:
        os.environ["DESIGN_ENGINE_TRACE_DIR"] = str(Path(args.trace_dir).resolve())
    os.chdir(ROOT)  # recipes name their files relative to the project folder, as the editor reads them
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    from engine import trace
    from engine.server import Hub, Session
    from trace_report import summary_text, write_report

    class Recorder(Hub):
        """Keeps the events the editor would get."""

        def __init__(self):
            super().__init__()
            self.events = []

        def publish(self, event):
            self.events.append(event)

    hub = Recorder()
    session = Session(hub)
    if args.workers:
        session.evaluator.workers = args.workers
    mode = "optimize" if args.optimize else args.mode
    print(f"{recipe_path.name}: {mode} with {session.evaluator.workers} workers, shown: {', '.join(visible) or 'nothing'}",
          flush=True)
    failed = False
    try:
        for quality in args.quality:
            seen, t0 = len(hub.events), time.perf_counter()
            trace.last_folder = None
            session.start(recipe, mode, quality, visible, optimize=args.optimize,
                          label=f"optimize {args.optimize}" if args.optimize else None)
            if session.thread is not None:
                session.thread.join()
            events = hub.events[seen:]
            done = next((e for e in events if e.get("type") == "run_done"), {})
            for e in events:
                if e.get("type") == "node_error" or (e.get("type") == "message" and e.get("level") in ("warning", "error")):
                    print(f"  {e.get('node', '')}: {e.get('message') or e.get('text')}")
            state = "ok" if done.get("ok") else f"FAILED: {done.get('error') or 'cancelled'}"
            failed |= not done.get("ok")
            print(f"\n=== {quality}: {state} in {time.perf_counter() - t0:.2f} s", flush=True)
            folder = trace.last_folder
            if folder is None:
                print("no trace was written (see the messages above)")
                continue
            path, summary = write_report(folder)
            print(summary_text(summary))
            print(f"\nreport: {path}", flush=True)
    finally:
        session.evaluator.close()
    return 1 if failed else 0


if __name__ == "__main__":  # required on Windows: worker processes re-import this file
    sys.exit(main())
