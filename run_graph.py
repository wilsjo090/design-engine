"""Run a graph recipe (JSON).

    python run_graph.py recipes/wave_offset.json
    python run_graph.py recipes/lamp_petals.json --set import.path="C:/parts/lamp.STEP" --set amplitude.value=0.15
    python run_graph.py --nodes            # list every node type and its inputs

--set NODE.INPUT=VALUE overrides an input for this run (numbers are parsed automatically).
--save FILE writes the graph, including overrides, to a new recipe.
"""
import argparse
import json
import sys
import time

from engine.graph import ARRAY_CACHE, Evaluator, Graph, GraphError, node_catalog


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def print_catalog():
    category = None
    for t in node_catalog():
        if t["category"] != category:
            category = t["category"]
            print(f"\n{category}")
        ins = ", ".join(p["name"] + (f"={p['default']}" if "default" in p else "") for p in t["inputs"])
        outs = ", ".join(p["name"] for p in t["outputs"])
        print(f"  {t['type']:<26} {t['title']:<30} ({ins}) -> {outs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", nargs="?")
    ap.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=VALUE")
    ap.add_argument("--save", metavar="FILE")
    ap.add_argument("--nodes", action="store_true", help="list available node types")
    ap.add_argument("--workers", type=int, default=4, help="threads used when processing many bodies")
    args = ap.parse_args()

    if args.nodes:
        print_catalog()
        return
    if not args.recipe:
        ap.error("a recipe is required")

    graph = Graph.load(args.recipe)
    for item in args.set:
        target, _, value = item.partition("=")
        node_id, _, port = target.partition(".")
        if not port or not value:
            ap.error(f"--set needs NODE.INPUT=VALUE, got '{item}'")
        graph.set(node_id, port, parse_value(value))
    if args.save:
        graph.save(args.save)
        print(f"saved {args.save}")

    t0 = time.perf_counter()
    evaluator = Evaluator(graph, workers=args.workers)
    try:
        results = evaluator.run()
    except GraphError as err:
        sys.exit(f"error: {err}")
    finally:
        evaluator.close()
    for node_id in graph.sinks():
        for port, value in results[node_id].items():
            print(f"{node_id}.{port} = {value}")
    print(f"done in {time.perf_counter() - t0:.1f}s (array cache: {ARRAY_CACHE.hits} hits, {ARRAY_CACHE.misses} misses)")


if __name__ == "__main__":
    main()
