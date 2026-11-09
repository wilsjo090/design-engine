import json
import os
import sys
import time
from pathlib import Path

import cloudpickle
import pytest

from engine import trace
from engine.graph.core import REGISTRY, Evaluator, Graph, Many, Port, node
from engine.graph.workers import StickyPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import trace_report  # noqa: E402

# Worker processes cannot import this test module, so ship its functions by value.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _items(count):
    return Many(range(int(count)))


def _square(x):
    with trace.span("work", device="cpu", value=x):
        return x * x


@pytest.fixture
def nodes():
    node("test.trace_items", "Items", "Test", [Port("count", "int", 4)], [Port("out", "int")])(_items)
    node("test.trace_square", "Square", "Test", [Port("x", "int")], [Port("out", "int")], parallel=True)(_square)
    yield
    REGISTRY.pop("test.trace_items", None)
    REGISTRY.pop("test.trace_square", None)


def _graph():
    g = Graph()
    g.add("items", "test.trace_items", count=4)
    g.add("square", "test.trace_square", x={"link": "items.out"})
    return g


def _run(graph, workers=2):
    evaluator = Evaluator(graph, workers=workers, log=None)
    try:
        return evaluator.run()
    finally:
        evaluator.close()


def test_spans_of_the_main_process_and_its_workers_land_in_the_run_folder(tmp_path, monkeypatch, nodes):
    monkeypatch.setenv("DESIGN_ENGINE_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "ENABLED", True)
    before = time.time()
    results = _run(_graph())
    assert list(results["square"]["out"]) == [0, 1, 4, 9]   # the trace only watches
    assert not trace.ACTIVE
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert info["pid"] == os.getpid() and info["end"] > info["start"]
    assert abs(info["start"] - before) < 5   # near the wall clock

    run = trace_report.load_run(folder)
    main = [s for s in run["spans"] if s["pid"] == os.getpid()]
    work = [s for s in run["spans"] if s["pid"] != os.getpid()]
    assert {s["node"] for s in main if s["stage"] == "node"} == {"items", "square"}
    sends = {s["item"]: s for s in main if s["stage"] == "send"}
    recvs = {s["item"]: s for s in main if s["stage"] == "recv"}
    assert sorted(sends) == sorted(recvs) == [0, 1, 2, 3]
    (plan,) = [s for s in main if s["stage"] == "map plan"]
    assert plan["node"] == "square" and plan["items"] == 4 and plan["active"] == 2 and plan["gpu"] is False

    assert {s["role"] for s in work} == {"worker 0", "worker 1"}
    tasks = {s["item"]: s for s in work if s["stage"] == "task"}
    assert sorted(tasks) == [0, 1, 2, 3] and all(s["node"] == "square" for s in tasks.values())
    assert len([s for s in work if s["stage"] == "load args"]) == 4
    inner = [s for s in work if s["stage"] == "work"]
    assert sorted(s["value"] for s in inner) == [0, 1, 2, 3]
    assert all(s["depth"] == 1 and s["device"] == "cpu" and s["node"] == "square" for s in inner)
    for i, task in tasks.items():   # one clock in every process
        assert sends[i]["t0"] <= task["t0"] <= task["t1"] <= recvs[i]["t0"]
        assert task["role"] == f"worker {i % 2}"

    samples = run["samples"]   # (from a process of its own: a C extension holding the GIL cannot starve it)
    assert len(samples) >= 2 and any(str(os.getpid()) in s.get("procs", {}) for s in samples)
    sampler = run["head"]["sampler"]
    assert sampler != os.getpid() and not any(str(sampler) in s.get("procs", {}) for s in samples)
    assert any({str(s["pid"]) for s in work} <= set(smp.get("procs", {})) for smp in samples)   # the workers too

    path, summary = trace_report.write_report(folder)
    page = path.read_text(encoding="utf-8")
    assert path == folder / "report.html" and "<canvas" in page and "worker 1" in page
    assert {p["role"] for p in summary["procs"]} >= {"main", "worker 0", "worker 1"}
    assert any(r["stage"] == "work" and r["where"] == "workers" and r["count"] == 4 for r in summary["stages"])
    assert summary["transfer"]["count"] == 4 and summary["transfer"]["mean"] >= 0
    assert "work" in trace_report.summary_text(summary)


def test_nothing_is_written_when_tracing_is_off(tmp_path, monkeypatch, nodes):
    monkeypatch.setenv("DESIGN_ENGINE_TRACE_DIR", str(tmp_path / "trace"))
    monkeypatch.setattr(trace, "ENABLED", False)
    assert list(_run(_graph())["square"]["out"]) == [0, 1, 4, 9]
    assert not (tmp_path / "trace").exists()
    assert not trace.ACTIVE and trace.span("x") is trace._NOTHING and trace.task_meta() is None


def test_a_run_inside_a_traced_run_is_part_of_it(tmp_path, monkeypatch):
    monkeypatch.setenv("DESIGN_ENGINE_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "ENABLED", True)
    with trace.run("job") as outer:
        assert outer is not None and trace.begin_run("run") is None   # (Evaluator.run inside a server job)
        with trace.span("pack", node="mesh") as span:
            span.set(triangles=12)
            with trace.span("inner"):
                trace.mark("moment")
    assert not trace.ACTIVE
    (folder,) = list(tmp_path.iterdir())
    spans = {s["stage"]: s for s in trace_report.load_run(folder)["spans"]}
    assert spans["pack"]["triangles"] == 12 and spans["pack"]["depth"] == 0
    assert spans["inner"]["node"] == "mesh" and spans["inner"]["depth"] == 1   # inherits the node around it
    assert spans["moment"]["t0"] == spans["moment"]["t1"] and spans["moment"]["depth"] == 2


def test_workers_stop_tracing_when_the_run_ends(tmp_path, monkeypatch):
    monkeypatch.setenv("DESIGN_ENGINE_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "ENABLED", True)
    pool = StickyPool(1)
    try:
        with trace.run("first"):
            assert pool.map(_square, [{"x": 2}]) == [4]
        assert pool.map(_square, [{"x": 3}]) == [9]   # not traced: the task carries no run
    finally:
        pool.close()
    (folder,) = list(tmp_path.iterdir())
    work = [s for s in trace_report.load_run(folder)["spans"] if s["stage"] == "work"]
    assert [s["value"] for s in work] == [2]
