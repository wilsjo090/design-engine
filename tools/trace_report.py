"""Draw a traced run (engine/trace.py) as one self-contained HTML page, and sum it up.

    python tools/trace_report.py <run folder> [-o report.html]
    python tools/trace_report.py                   # the newest run in the trace folder
    python tools/trace_report.py --match final     # the newest run with "final" in its name
    python tools/trace_report.py --list

The page: a row per process (and per other thread of the main process), its spans coloured by stage,
nested spans below the ones around them, the process's CPU (cores) shaded behind them; under the rows
the GPU's use and memory, the engine's CPU (main process, workers, the whole PC) and its memory, all on
one time axis. Wheel zooms, drag pans, double-click shows everything; hover shows a span's stage, node,
item, device and duration. Below: totals per stage, each process's busy share, the GPU's, and the time
the GPU sat idle while the workers worked (and the other way round). The same summary is printed.
Written as report.html in the run folder unless -o says otherwise.
"""
import argparse
import html
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

GPU_IDLE = 10.0   # % of GPU use under which the GPU counts as idle


# --- reading -------------------------------------------------------------------------------------
def _lines(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # (a line cut short by a process that was stopped)
    except OSError:
        pass
    return out


def load_run(folder):
    """{"folder", "info" (run.json), "spans" (every process's, marks included), "samples", "head" (the
    sampler's first line: cores, GPU name and memory)} of a run folder."""
    folder = Path(folder)
    try:
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    spans = []
    for path in sorted(folder.glob("*.jsonl")):
        if path.name != "samples.jsonl":
            spans += _lines(path)
    samples = _lines(folder / "samples.jsonl")
    head = next((s for s in samples if s.get("info")), {})
    samples = [s for s in samples if "t" in s]
    return {"folder": folder, "info": info, "spans": spans, "samples": samples, "head": head}


# --- intervals -----------------------------------------------------------------------------------
def _union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _subtract(A, B):
    """Merged intervals A less merged intervals B."""
    out, j = [], 0
    for a, b in A:
        while j < len(B) and B[j][1] <= a:
            j += 1
        k, start = j, a
        while k < len(B) and B[k][0] < b:
            if B[k][0] > start:
                out.append([start, B[k][0]])
            start = max(start, B[k][1])
            k += 1
        if start < b:
            out.append([start, b])
    return out


def _intersect(A, B):
    return _subtract(A, _subtract(A, B))


def _length(merged):
    return sum(b - a for a, b in merged)


# --- the summary ---------------------------------------------------------------------------------
def _is_cpu_worker(role):
    """A CPU-only worker (two-stage maps: it runs the stages the GPU workers hand over)."""
    return str(role).startswith("cpu worker")


def _is_worker(role):
    return str(role).startswith("worker") or _is_cpu_worker(role)


def _where(role):
    return "cpu wk" if _is_cpu_worker(role) else "workers" if _is_worker(role) else "main"


def _role_key(role):
    if role == "main":
        return (0, 0)
    if _is_worker(role):
        try:
            return (2 if _is_cpu_worker(role) else 1, int(str(role).split()[-1]))
        except (IndexError, ValueError):
            return (2 if _is_cpu_worker(role) else 1, 999)
    return (3, 0)


def summarize(run):
    """The numbers of the summary tables (seconds, shares 0..1)."""
    info, spans, samples = run["info"], run["spans"], run["samples"]
    start = info.get("start") or min([s["t0"] for s in spans] + [s["t"] for s in samples] or [0.0])
    end = info.get("end") or max([s["t1"] for s in spans] + [s["t"] for s in samples] or [start])
    wall = max(end - start, 1e-9)
    work = [s for s in spans if not s.get("mark")]
    marks = [s for s in spans if s.get("mark")]

    stages = defaultdict(lambda: [0, 0.0, 0.0])
    for s in work:
        key = (s["stage"], _where(s.get("role")), s.get("device", ""))
        d = s["t1"] - s["t0"]
        e = stages[key]
        e[0], e[1], e[2] = e[0] + 1, e[1] + d, max(e[2], d)
    stage_rows = sorted(({"stage": k[0], "where": k[1], "device": k[2], "count": v[0], "total": v[1],
                          "mean": v[1] / v[0], "max": v[2]} for k, v in stages.items()), key=lambda r: -r["total"])

    roles = {}
    for s in spans:
        roles.setdefault(s["pid"], s.get("role", "?"))
    # busy: what a process's outermost spans cover, less the time the main process waits for its workers
    # (inside "map") except while it sends them their items
    busy = {}
    for pid in roles:
        own = [s for s in work if s["pid"] == pid]
        top = _union([(s["t0"], s["t1"]) for s in own if s.get("depth", 0) == 0])
        waits = _union([(s["t0"], s["t1"]) for s in own if s["stage"] == "map"])
        sends = _union([(s["t0"], s["t1"]) for s in own if s["stage"] == "send"])
        busy[pid] = _union(_subtract(top, waits) + sends)

    cpu, rss = defaultdict(list), defaultdict(int)
    for smp in samples:
        for pid, (cores, mem) in (smp.get("procs") or {}).items():
            cpu[int(pid)].append(cores)
            rss[int(pid)] = max(rss[int(pid)], mem)
    pids = sorted(set(roles) | set(cpu), key=lambda p: (_role_key(roles.get(p, "")), p))
    procs = [{"role": roles.get(pid, "(no spans)"), "pid": pid,
              "busy": _length(busy.get(pid, [])) / wall if pid in roles else None,
              "cpu_mean": sum(cpu[pid]) / len(samples) if samples and cpu.get(pid) else None,
              "cpu_max": max(cpu[pid]) if cpu.get(pid) else None, "rss_peak": rss.get(pid) or None,
              "spans": sum(1 for s in work if s["pid"] == pid)} for pid in pids]

    gpu_samples = [(s["t"], s["gpu"]) for s in samples if "gpu" in s]
    gpu = None
    if gpu_samples:
        mem = [s["gpu_mem"] for s in samples if "gpu_mem" in s]
        gpu = {"mean": sum(g for _, g in gpu_samples) / len(gpu_samples) / 100,
               "busy_share": sum(1 for _, g in gpu_samples if g >= GPU_IDLE) / len(gpu_samples),
               "mem_peak": max(mem) if mem else None, "total": run["head"].get("gpu_total"),
               "name": run["head"].get("gpu_name")}

    # who works when: workers busy (in a task) and in what kind of stage, the main process, the GPU (each sample's
    # use held until the next one), as exact interval sums
    workers = [pid for pid in roles if _is_worker(roles[pid])]
    cpu_workers = [pid for pid in workers if _is_cpu_worker(roles[pid])]
    whole = [[start, end]]
    any_worker = _union([iv for pid in workers for iv in busy[pid]])
    main_alone = _subtract(busy.get(info.get("pid"), []), any_worker)
    maps = _union([(s["t0"], s["t1"]) for s in work if s["stage"] == "map"])
    o = defaultdict(float)
    for pid in workers:
        on = {dev: _union([(s["t0"], s["t1"]) for s in work if s["pid"] == pid and s.get("device") == dev])
              for dev in ("gpu", "cpu")}
        o["worker_gpu"] += _length(_intersect(busy[pid], on["gpu"]))
        o["worker_cpu"] += _length(_intersect(busy[pid], _subtract(on["cpu"], on["gpu"])))
        o["worker_other"] += _length(_subtract(_subtract(busy[pid], on["gpu"]), on["cpu"]))
        o["cpu_workers_busy_while_mapping" if pid in cpu_workers else "busy_while_mapping"] += \
            _length(_intersect(busy[pid], maps))
    o["mapping"] = _length(maps)
    o["main_alone"] = _length(main_alone)
    if gpu_samples:
        steps = [(t, gpu_samples[k + 1][0] if k + 1 < len(gpu_samples) else end, g)
                 for k, (t, g) in enumerate(gpu_samples)]
        idle = _intersect(_union([(a, b) for a, b, g in steps if g < GPU_IDLE and b > a]), whole)
        used = _intersect(_union([(a, b) for a, b, g in steps if g >= GPU_IDLE and b > a]), whole)
        o["gpu_idle_workers_busy"] = _length(_intersect(idle, any_worker))
        o["gpu_busy_workers_idle"] = _length(_subtract(used, any_worker))
        o["gpu_idle_main_alone"] = _length(_intersect(idle, main_alone))
    overlap = dict(o)
    overlap["workers"] = len(workers) - len(cpu_workers)
    overlap["cpu_workers"] = len(cpu_workers)
    overlap["busy_mean_while_mapping"] = o["busy_while_mapping"] / o["mapping"] if o["mapping"] else None
    overlap["cpu_busy_mean_while_mapping"] = (o["cpu_workers_busy_while_mapping"] / o["mapping"]
                                              if o["mapping"] and cpu_workers else None)

    # an item's result: from its worker's task end to the main process receiving it (paired in order)
    tasks, recvs = defaultdict(list), defaultdict(list)
    for s in work:
        if s["stage"] == "task":
            tasks[(s.get("node"), s.get("item"))].append(s["t1"])
    for m in marks:
        if m["stage"] == "recv":
            recvs[(m.get("node"), m.get("item"))].append(m["t0"])
    latency = []
    for key, got in recvs.items():
        ends = sorted(tasks.get(key, []))
        for r in sorted(got):
            before = [e for e in ends if e <= r + 1e-4]
            if before:
                latency.append(max(r - before[-1], 0.0))
                ends.remove(before[-1])
    transfer = {"count": len(latency), "mean": sum(latency) / len(latency) if latency else None,
                "max": max(latency) if latency else None,
                "send": sum(s["t1"] - s["t0"] for s in work if s["stage"] == "send"),
                "load": sum(s["t1"] - s["t0"] for s in work if s["stage"] == "load args"
                            and not _is_cpu_worker(s.get("role")))}

    # two-stage maps: GPU workers hand CPU stages to CPU workers (pickled and queued there, read and unpickled
    # by a CPU worker; the wait from the hand-over to the CPU worker's task is the queue plus the pipe)
    hand = [s for s in work if s["stage"] == "hand over"]
    handover = None
    if hand:
        starts = defaultdict(list)
        for s in work:
            if s["stage"] == "task" and _is_cpu_worker(s.get("role")):
                starts[(s.get("node"), s.get("item"))].append(s["t0"])
        waits = []
        for s in sorted(hand, key=lambda s: s["t1"]):
            later = [t for t in starts.get((s.get("node"), s.get("item")), []) if t >= s["t1"] - 1e-4]
            if later:
                waits.append(max(min(later) - s["t1"], 0.0))
                starts[(s.get("node"), s.get("item"))].remove(min(later))
        loads = [s for s in work if s["stage"] == "load args" and _is_cpu_worker(s.get("role"))]
        handover = {"count": len(hand), "time": sum(s["t1"] - s["t0"] for s in hand),
                    "bytes": sum(s.get("bytes", 0) for s in hand),
                    "load": sum(s["t1"] - s["t0"] for s in loads),
                    "wait_mean": sum(waits) / len(waits) if waits else None,
                    "wait_median": sorted(waits)[len(waits) // 2] if waits else None,
                    "wait_max": max(waits) if waits else None}
    transfer["handover"] = handover

    plans = defaultdict(list)
    for m in marks:
        if m["stage"] == "map plan":
            plans[m.get("node")].append(m)
    nodes = []
    for s in sorted((s for s in work if s["stage"] == "node"), key=lambda s: s["t0"]):
        plan = "; ".join(f"{p.get('step', '')}: {p.get('items')} items, {p.get('active')} of {p.get('workers')} "
                         f"workers{' (memory allows ' + str(p.get('memory')) + ')' if p.get('memory') not in (None, p.get('active')) else ''}"
                         f"{', GPU' if p.get('gpu') else ''}"
                         f"{', CPU stages to up to ' + str(p.get('cpu')) + ' CPU workers' if p.get('cpu') else ''}"
                         for p in plans.get(s.get("node"), []) if s["t0"] <= p["t0"] <= s["t1"])
        nodes.append({"node": s.get("node"), "type": s.get("type"), "start": s["t0"] - start,
                      "duration": s["t1"] - s["t0"], "plan": plan, "error": s.get("error")})
    cached = sum(1 for m in marks if m["stage"] == "cached")
    return {"run": info.get("run", run["folder"].name), "label": info.get("label"), "start": start, "end": end,
            "wall": wall, "cpus": info.get("cpus") or run["head"].get("cpus"), "stages": stage_rows, "procs": procs,
            "gpu": gpu, "overlap": overlap, "transfer": transfer, "nodes": nodes, "cached": cached,
            "samples": len(samples)}


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def _gb(x):
    return "-" if not x else f"{x / 2**30:.2f}"


def _ms(x):
    return "-" if x is None else f"{1000 * x:.1f}"


def summary_text(s):
    """The summary as plain text (printed by trace_run.py and this tool)."""
    out = [f"Run {s['run']} ({s['label']}): {s['wall']:.2f} s, {s['cpus']} logical cores, {s['samples']} samples"]
    out.append("")
    out.append(f"{'stage':<18}{'where':<9}{'device':<7}{'count':>7}{'total s':>10}{'mean ms':>10}{'max ms':>10}")
    for r in s["stages"]:
        out.append(f"{r['stage']:<18}{r['where']:<9}{r['device'] or '':<7}{r['count']:>7}{r['total']:>10.3f}"
                   f"{1000 * r['mean']:>10.1f}{1000 * r['max']:>10.1f}")
    out.append("")
    out.append(f"{'process':<12}{'pid':>8}{'busy':>7}{'CPU mean':>10}{'CPU max':>9}{'RSS peak GB':>13}{'spans':>7}")
    for p in s["procs"]:
        mean = "-" if p["cpu_mean"] is None else f"{p['cpu_mean']:.2f}"
        top = "-" if p["cpu_max"] is None else f"{p['cpu_max']:.2f}"
        out.append(f"{p['role']:<12}{p['pid']:>8}{_pct(p['busy']):>7}{mean:>10}{top:>9}{_gb(p['rss_peak']):>13}"
                   f"{p['spans']:>7}")
    out.append("(busy: share of the run inside the process's own work, the main process's waits for its workers "
               "left out; CPU in cores)")
    out.append("")
    g, o = s["gpu"], s["overlap"]
    if g:
        out.append(f"GPU {g['name'] or ''}: mean use {_pct(g['mean'])}, busy (>= {GPU_IDLE:g}%) {_pct(g['busy_share'])} of "
                   f"the time, memory peak {_gb(g['mem_peak'])} of {_gb(g['total'])} GB")
        out.append(f"GPU idle while a worker works: {o.get('gpu_idle_workers_busy', 0):.2f} s "
                   f"({_pct(o.get('gpu_idle_workers_busy', 0) / s['wall'])} of the run); GPU busy while no worker "
                   f"works: {o.get('gpu_busy_workers_idle', 0):.2f} s; GPU idle while only the main process works: "
                   f"{o.get('gpu_idle_main_alone', 0):.2f} s")
    else:
        out.append("GPU: no samples (no NVIDIA driver, or NVML missing)")
    busy_mean = o.get("busy_mean_while_mapping")
    cpu_mean = o.get("cpu_busy_mean_while_mapping")
    out.append(f"Worker time by stage device: GPU {o.get('worker_gpu', 0):.2f} s, CPU {o.get('worker_cpu', 0):.2f} s, "
               f"other {o.get('worker_other', 0):.2f} s; while mapping ({o.get('mapping', 0):.2f} s) "
               f"{'-' if busy_mean is None else f'{busy_mean:.1f}'} of {o['workers']} workers busy on average"
               + (f", {'-' if cpu_mean is None else f'{cpu_mean:.1f}'} of {o['cpu_workers']} CPU workers"
                  if o.get("cpu_workers") else ""))
    out.append(f"Main process working while no worker does: {o.get('main_alone', 0):.2f} s")
    tr = s["transfer"]
    if tr["count"]:
        out.append(f"Items: {tr['count']} results, worker task end -> main receives: mean {1000 * tr['mean']:.1f} ms, "
                   f"max {1000 * tr['max']:.1f} ms; sending (pickling) {tr['send']:.3f} s, loading args in "
                   f"workers {tr['load']:.3f} s")
    h = tr.get("handover")
    if h:
        out.append(f"Hand-over: {h['count']} CPU stages, {h['bytes'] / h['count'] / 2**20:.1f} MB each; pickling and "
                   f"queueing in GPU workers {h['time']:.3f} s ({1000 * h['time'] / h['count']:.1f} ms each), "
                   f"unpickling in CPU workers {h['load']:.3f} s; hand-over -> CPU task start: mean "
                   f"{_ms(h['wait_mean'])}, median {_ms(h['wait_median'])}, max {_ms(h['wait_max'])} ms")
    out.append("")
    out.append(f"{'node':<28}{'type':<26}{'start s':>9}{'took s':>9}  workers")
    for n in s["nodes"]:
        out.append(f"{str(n['node'])[:27]:<28}{str(n['type'])[:25]:<26}{n['start']:>9.2f}{n['duration']:>9.3f}  "
                   f"{n['plan']}{'  (' + n['error'] + ')' if n['error'] else ''}")
    if s["cached"]:
        out.append(f"({s['cached']} nodes came from the cache)")
    return "\n".join(out)


# --- the page ------------------------------------------------------------------------------------
STAGE_COLORS = {
    "node": "#c9ced6", "map": "#e3e6ea", "item": "#d8dde6", "task": "#b8c4d6", "block": "#d6dbe4",
    "iteration": "#d6dbe4", "send": "#c58fbf", "load args": "#9467bd", "shared read": "#7b4ea3",
    "field": "#4e79a7", "probe": "#86bcb6", "marching cubes": "#59a14f", "decimate": "#e15759",
    "reduce": "#ff9da7", "dual contour": "#edc948", "join blocks": "#b07aa1", "stitch": "#f28e2b",
    "fuse": "#9c755f", "pack": "#76b7b2", "paint": "#bab0ac", "slice": "#a0cbe8", "solve": "#17becf",
    "oc update": "#bcbd22", "live view": "#8c564b", "checkpoint": "#7f7f7f", "fea setup": "#aec7e8",
    "fea solve": "#1f77b4", "gc": "#a5a5a5", "recv": "#333333", "map plan": "#d62728", "cached": "#999999",
    "hand over": "#d37295", "handed": "#8c8c8c",
}
SPARE = ["#8cd17d", "#b6992d", "#499894", "#d37295", "#fabfd2", "#79706e", "#d4a6c8", "#9d7660"]


def _page_data(run, s):
    start, info = s["start"], run["info"]
    spans = run["spans"]
    roles = {}
    for sp in spans:
        roles.setdefault(sp["pid"], sp.get("role", "?"))
    main_pid = info.get("pid")
    keys = sorted({(sp["pid"], sp.get("thread", "")) for sp in spans},
                  key=lambda k: (_role_key(roles.get(k[0], "")), k[0], k[1] != "", k[1]))
    rows = {k: i for i, k in enumerate(keys)}
    stage_names, node_names = [], []
    stage_ix, node_ix = {}, {}

    def ix(table, names, value):
        if value not in table:
            table[value] = len(names)
            names.append(value)
        return table[value]

    depth = defaultdict(int)
    out_spans, out_marks = [], []
    skip = {"pid", "role", "stage", "node", "item", "device", "t0", "t1", "depth", "thread", "mark"}
    for sp in spans:
        r = rows[(sp["pid"], sp.get("thread", ""))]
        extra = {k: v for k, v in sp.items() if k not in skip}
        item = sp.get("item")
        item = "" if item is None else json.dumps(item) if not isinstance(item, str) else item
        st, nd = ix(stage_ix, stage_names, sp["stage"]), ix(node_ix, node_names, sp.get("node") or "")
        if sp.get("mark"):
            out_marks.append([r, round(sp["t0"] - start, 6), st, nd, item, json.dumps(extra) if extra else ""])
            continue
        d = int(sp.get("depth", 0))
        depth[r] = max(depth[r], d + 1)
        out_spans.append([r, round(sp["t0"] - start, 6), round(sp["t1"] - start, 6), d, st, nd, item,
                          sp.get("device", ""), json.dumps(extra) if extra else ""])
    colors, spare = [], 0
    for name in stage_names:
        if name in STAGE_COLORS:
            colors.append(STAGE_COLORS[name])
        else:
            colors.append(SPARE[spare % len(SPARE)])
            spare += 1
    row_list = []
    for (pid, thread), r in sorted(rows.items(), key=lambda kv: kv[1]):
        label = roles.get(pid, "?") + (f" · {thread}" if thread else "")
        row_list.append({"label": label, "pid": pid, "lanes": max(depth[r], 1), "cpu": not thread})
    samples = run["samples"]
    procs = {}
    for i, smp in enumerate(samples):
        for pid, (cores, mem) in (smp.get("procs") or {}).items():
            procs.setdefault(pid, [None] * len(samples))[i] = cores
    workers = [str(p) for p, r in roles.items() if _is_worker(r) and not _is_cpu_worker(r)]
    cpu_workers = [str(p) for p, r in roles.items() if _is_cpu_worker(r)]
    sm = {
        "t": [round(x["t"] - start, 4) for x in samples],
        "gpu": [x.get("gpu") for x in samples] if any("gpu" in x for x in samples) else None,
        "gpu_mem": [x.get("gpu_mem") for x in samples] if any("gpu_mem" in x for x in samples) else None,
        "gpu_total": run["head"].get("gpu_total"),
        "sys_cpu": [x.get("sys_cpu") for x in samples],
        "engine_cpu": [round(sum(c for c, _ in (x.get("procs") or {}).values()), 3) for x in samples],
        "main_cpu": [((x.get("procs") or {}).get(str(main_pid)) or [None])[0] for x in samples],
        "workers_cpu": [round(sum(c for p, (c, _) in (x.get("procs") or {}).items() if p in workers), 3)
                        for x in samples],
        "cpu_workers_cpu": [round(sum(c for p, (c, _) in (x.get("procs") or {}).items() if p in cpu_workers), 3)
                            for x in samples] if cpu_workers else None,
        "rss": [sum(m for _, m in (x.get("procs") or {}).values()) for x in samples],
        "commit": [x.get("commit_room") for x in samples],
    }
    return {"end": round(s["end"] - start, 6), "rows": row_list, "spans": out_spans, "marks": out_marks,
            "stages": stage_names, "colors": colors, "nodes": node_names, "samples": sm, "procs": procs,
            "cpus": s["cpus"] or 1, "gpuIdle": GPU_IDLE}


def _table(head, rows):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _summary_html(s):
    parts = []
    g, o, tr = s["gpu"], s["overlap"], s["transfer"]
    facts = [("Wall time", f"{s['wall']:.2f} s"), ("Logical cores", s["cpus"])]
    if g:
        facts += [("GPU", g["name"] or "?"), ("GPU mean use", _pct(g["mean"])),
                  (f"GPU busy (>= {GPU_IDLE:g}%)", _pct(g["busy_share"]) + " of the time"),
                  ("GPU memory peak", f"{_gb(g['mem_peak'])} of {_gb(g['total'])} GB"),
                  ("GPU idle while a worker works", f"{o.get('gpu_idle_workers_busy', 0):.2f} s "
                                                    f"({_pct(o.get('gpu_idle_workers_busy', 0) / s['wall'])})"),
                  ("GPU busy while no worker works", f"{o.get('gpu_busy_workers_idle', 0):.2f} s"),
                  ("GPU idle while only the main process works", f"{o.get('gpu_idle_main_alone', 0):.2f} s")]
    else:
        facts.append(("GPU", "no samples"))
    busy_mean = o.get("busy_mean_while_mapping")
    cpu_mean = o.get("cpu_busy_mean_while_mapping")
    facts += [("Worker time in GPU stages", f"{o.get('worker_gpu', 0):.2f} s"),
              ("Worker time in CPU stages", f"{o.get('worker_cpu', 0):.2f} s"),
              ("Worker time in other work", f"{o.get('worker_other', 0):.2f} s"),
              ("Workers busy while mapping", "-" if busy_mean is None else
               f"{busy_mean:.1f} of {o['workers']} on average over {o.get('mapping', 0):.2f} s")]
    if o.get("cpu_workers"):
        facts.append(("CPU workers busy while mapping", "-" if cpu_mean is None else
                      f"{cpu_mean:.1f} of {o['cpu_workers']} on average"))
    facts.append(("Main process working while no worker does", f"{o.get('main_alone', 0):.2f} s"))
    if tr["count"]:
        facts += [("Result: task end to received", f"mean {1000 * tr['mean']:.1f} ms, max {1000 * tr['max']:.1f} ms "
                                                   f"({tr['count']} items)"),
                  ("Sending items (pickling, main)", f"{tr['send']:.3f} s"),
                  ("Loading items (unpickling, workers)", f"{tr['load']:.3f} s")]
    h = tr.get("handover")
    if h:
        facts += [("CPU stages handed over", f"{h['count']}, {h['bytes'] / h['count'] / 2**20:.1f} MB each"),
                  ("Hand-over in GPU workers (pickling, queueing)",
                   f"{h['time']:.3f} s, {1000 * h['time'] / h['count']:.1f} ms each"),
                  ("Unpickling in CPU workers", f"{h['load']:.3f} s"),
                  ("Hand-over to CPU task start", f"mean {_ms(h['wait_mean'])} ms, median {_ms(h['wait_median'])} ms, "
                                                  f"max {_ms(h['wait_max'])} ms")]
    parts.append("<h2>Summary</h2>" + _table(["", ""], facts))
    parts.append("<h2>Processes</h2>" + _table(
        ["process", "pid", "busy", "CPU mean (cores)", "CPU max", "RSS peak GB", "spans"],
        [[p["role"], p["pid"], _pct(p["busy"]), "-" if p["cpu_mean"] is None else f"{p['cpu_mean']:.2f}",
          "-" if p["cpu_max"] is None else f"{p['cpu_max']:.2f}", _gb(p["rss_peak"]), p["spans"]] for p in s["procs"]])
        + "<p class=note>busy: share of the run inside the process's own work; the main process's waits for its "
          "workers (inside map, except sending) are left out.</p>")
    parts.append("<h2>Stages</h2>" + _table(
        ["stage", "where", "device", "count", "total s", "mean ms", "max ms"],
        [[r["stage"], r["where"], r["device"], r["count"], f"{r['total']:.3f}", f"{1000 * r['mean']:.1f}",
          f"{1000 * r['max']:.1f}"] for r in s["stages"]]))
    parts.append("<h2>Nodes</h2>" + _table(
        ["node", "type", "start s", "took s", "workers"],
        [[n["node"], n["type"], f"{n['start']:.2f}", f"{n['duration']:.3f}",
          n["plan"] + (f" ({n['error']})" if n["error"] else "")] for n in s["nodes"]])
        + (f"<p class=note>{s['cached']} nodes came from the cache.</p>" if s["cached"] else ""))
    return "\n".join(parts)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace __TITLE__</title>
<style>
:root { --bg:#fbfbfc; --fg:#1d232b; --muted:#6b7480; --line:#e2e5ea; --row:#f3f5f8; --cpu:#dfe6f0; }
@media (prefers-color-scheme: dark) { :root { --bg:#16191e; --fg:#e4e7eb; --muted:#9aa3ad; --line:#2c323a;
  --row:#1d2127; --cpu:#26303c; } }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui, sans-serif; }
h1 { font-size:18px; margin:0 0 4px; } h2 { font-size:15px; margin:24px 0 8px; }
.sub { color:var(--muted); margin-bottom:10px; }
#wrap { position:relative; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
canvas { display:block; width:100%; cursor:grab; }
#tip { position:absolute; pointer-events:none; background:rgba(20,24,30,.92); color:#fff; padding:6px 8px;
  border-radius:4px; font-size:12px; white-space:pre; display:none; z-index:2; max-width:520px; }
#legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin:10px 0; }
#legend span { display:inline-flex; align-items:center; gap:5px; }
#legend i { width:12px; height:12px; border-radius:2px; display:inline-block; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--line); padding:3px 10px; text-align:left; }
th { color:var(--muted); font-weight:600; }
.note { color:var(--muted); font-size:12px; }
button { font:inherit; padding:2px 10px; }
</style></head><body>
<h1>__TITLE__</h1>
<div class="sub">__SUB__ &middot; wheel: zoom &middot; drag: pan &middot; double-click: all
 &middot; <button id="reset">Show all</button></div>
<div id="wrap"><canvas id="c"></canvas><div id="tip"></div></div>
<div id="legend"></div>
__SUMMARY__
<script>
const D = __DATA__;
const cv = document.getElementById('c'), tip = document.getElementById('tip'), wrap = document.getElementById('wrap');
const ctx = cv.getContext('2d');
const LABEL = 150, LANE = 13, PAD = 5, AXIS = 22, CH = 64, GAP = 14, RIGHT = 12;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let view = [0, Math.max(D.end, 1e-3)], W = 800, mouse = null, drag = null;
// rows and their spans by lane, sorted by start (hover search)
const rowY = [], lanes = D.rows.map(r => Array.from({length: r.lanes}, () => []));
let y = AXIS;
D.rows.forEach((r, i) => { rowY.push(y); y += r.lanes * LANE + 2 * PAD; });
const rowsEnd = y;
for (const s of D.spans) lanes[s[0]][s[3]].push(s);
lanes.forEach(l => l.forEach(a => a.sort((p, q) => p[1] - q[1])));
const S = D.samples;
const charts = [];
if (S.gpu) charts.push({name: 'GPU use %', max: 100, series: [{v: S.gpu, color: '#2ca02c', fill: true}], idle: D.gpuIdle});
if (S.gpu_mem && S.gpu_total) charts.push({name: 'GPU memory GB', max: S.gpu_total / 2**30,
  series: [{v: S.gpu_mem.map(x => x == null ? null : x / 2**30), color: '#2ca02c'}]});
charts.push({name: 'CPU cores', max: D.cpus, series: [
  {v: S.sys_cpu, color: '#9aa3ad', label: 'whole PC'},
  {v: S.engine_cpu, color: '#4e79a7', fill: true, label: 'engine'},
  {v: S.workers_cpu, color: '#9467bd', label: 'workers'},
  ...(S.cpu_workers_cpu ? [{v: S.cpu_workers_cpu, color: '#e15759', label: 'CPU workers'}] : []),
  {v: S.main_cpu, color: '#f28e2b', label: 'main'}]});
charts.push({name: 'Engine RAM GB', max: Math.max(0.5, ...S.rss.map(x => x / 2**30)) * 1.1,
  series: [{v: S.rss.map(x => x / 2**30), color: '#e15759'}]});
const chartY = []; y = rowsEnd + GAP;
charts.forEach(c => { chartY.push(y); y += CH + GAP; });
const H = y;
const X = t => LABEL + (t - view[0]) / (view[1] - view[0]) * (W - LABEL - RIGHT);
const T = x => view[0] + (x - LABEL) / (W - LABEL - RIGHT) * (view[1] - view[0]);
function niceStep(range, px) {
  const raw = range / Math.max(px / 90, 1), p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}
function fmt(t) { const r = view[1] - view[0]; return r < 0.05 ? (t * 1000).toFixed(2) + ' ms' : r < 2 ? (t * 1000).toFixed(0) + ' ms' : t.toFixed(r < 20 ? 2 : 1) + ' s'; }
function draw() {
  const dpr = window.devicePixelRatio || 1;
  W = wrap.clientWidth;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = css('--bg'); ctx.fillRect(0, 0, W, H);
  ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'middle';
  // rows
  D.rows.forEach((r, i) => {
    const y0 = rowY[i], h = r.lanes * LANE + 2 * PAD;
    if (i % 2 === 0) { ctx.fillStyle = css('--row'); ctx.fillRect(0, y0, W, h); }
    const cpu = r.cpu ? D.procs[String(r.pid)] : null;
    if (cpu) {  // the process's CPU (cores) shaded behind its spans, scaled to its own peak (at least 1)
      const top = Math.max(1, ...cpu.filter(v => v != null));
      ctx.fillStyle = css('--cpu'); ctx.beginPath(); ctx.moveTo(X(S.t[0] || 0), y0 + h);
      S.t.forEach((t, k) => { const v = cpu[k] || 0; ctx.lineTo(X(t), y0 + h - v / top * (h - 2)); });
      ctx.lineTo(X(S.t[S.t.length - 1] || 0), y0 + h); ctx.closePath(); ctx.fill();
    }
  });
  ctx.save(); ctx.beginPath(); ctx.rect(LABEL, 0, W - LABEL - RIGHT, H); ctx.clip();
  const pxPerS = (W - LABEL - RIGHT) / (view[1] - view[0]);
  for (const s of D.spans) {
    if (s[2] < view[0] || s[1] > view[1]) continue;
    const x0 = X(s[1]), w = Math.max((s[2] - s[1]) * pxPerS, 0.7), yy = rowY[s[0]] + PAD + s[3] * LANE;
    ctx.fillStyle = D.colors[s[4]]; ctx.fillRect(x0, yy + 1, w, LANE - 2);
    if (s[7] === 'gpu') { ctx.fillStyle = '#00a651'; ctx.fillRect(x0, yy + 1, w, 3); }
    if (w > 44) {
      ctx.fillStyle = '#10141a';
      const label = D.stages[s[4]] === 'node' ? D.nodes[s[5]] : D.stages[s[4]] + (s[6] !== '' ? ' ' + s[6] : '');
      ctx.fillText(label.slice(0, Math.floor(w / 6)), Math.max(x0, LABEL) + 3, yy + LANE / 2);
    }
  }
  for (const m of D.marks) {
    if (m[1] < view[0] || m[1] > view[1]) continue;
    const r = D.rows[m[0]], yy = rowY[m[0]];
    ctx.fillStyle = D.colors[m[2]];
    ctx.fillRect(X(m[1]) - 0.5, yy + PAD + (r.lanes - 1) * LANE + LANE - 5, 1.5, 5);
  }
  // charts
  charts.forEach((c, i) => {
    const y0 = chartY[i], yv = v => y0 + CH - Math.min(v, c.max) / c.max * CH;
    ctx.strokeStyle = css('--line'); ctx.strokeRect(LABEL, y0, W - LABEL - RIGHT, CH);
    if (c.idle != null) { ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(LABEL, yv(c.idle)); ctx.lineTo(W, yv(c.idle)); ctx.stroke(); ctx.setLineDash([]); }
    for (const sr of c.series) {
      ctx.beginPath(); let on = false;
      S.t.forEach((t, k) => { const v = sr.v[k]; if (v == null) { on = false; return; }
        if (!on) { ctx.moveTo(X(t), yv(v)); on = true; } else ctx.lineTo(X(t), yv(v)); });
      ctx.strokeStyle = sr.color; ctx.lineWidth = 1.3; ctx.stroke();
      if (sr.fill && S.t.length) {
        ctx.lineTo(X(S.t[S.t.length - 1]), y0 + CH); ctx.lineTo(X(S.t[0]), y0 + CH); ctx.closePath();
        ctx.globalAlpha = 0.18; ctx.fillStyle = sr.color; ctx.fill(); ctx.globalAlpha = 1;
      }
    }
  });
  ctx.restore();
  // labels
  ctx.fillStyle = css('--fg');
  D.rows.forEach((r, i) => ctx.fillText(r.label.slice(0, 24), 6, rowY[i] + PAD + LANE / 2));
  charts.forEach((c, i) => {
    ctx.fillStyle = css('--fg'); ctx.fillText(c.name, 6, chartY[i] + 8);
    ctx.fillStyle = css('--muted'); ctx.fillText('max ' + (c.max >= 10 ? c.max.toFixed(0) : c.max.toFixed(1)), 6, chartY[i] + 22);
    let lx = LABEL + 6;  // the series' names along the plot's top
    for (const sr of c.series) if (sr.label) { ctx.fillStyle = sr.color; ctx.fillText(sr.label, lx, chartY[i] + 8);
      lx += ctx.measureText(sr.label).width + 12; }
  });
  // time axis
  const step = niceStep(view[1] - view[0], W - LABEL);
  ctx.strokeStyle = css('--line'); ctx.fillStyle = css('--muted');
  for (let t = Math.ceil(view[0] / step) * step; t <= view[1]; t += step) {
    const x = X(t); ctx.beginPath(); ctx.moveTo(x, AXIS - 4); ctx.lineTo(x, H); ctx.globalAlpha = 0.5; ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillText(fmt(t), x + 3, AXIS / 2);
  }
  if (mouse && mouse.x > LABEL) { ctx.strokeStyle = css('--fg'); ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(mouse.x, AXIS); ctx.lineTo(mouse.x, H); ctx.stroke(); ctx.globalAlpha = 1; }
}
function at(x, yy) {
  const t = T(x), slack = 2 / ((W - LABEL - RIGHT) / (view[1] - view[0]));
  for (let i = 0; i < D.rows.length; i++) {
    const y0 = rowY[i] + PAD, lane = Math.floor((yy - y0) / LANE);
    if (lane < 0 || lane >= D.rows[i].lanes) continue;
    let best = null;
    for (const s of lanes[i][lane]) { if (s[1] - slack > t) break; if (s[2] + slack >= t) best = s; }
    return best ? {span: best} : null;
  }
  for (let i = 0; i < charts.length; i++) if (yy >= chartY[i] && yy <= chartY[i] + CH) return {chart: charts[i], t};
  return null;
}
function sampleAt(t) { let k = 0; while (k + 1 < S.t.length && S.t[k + 1] <= t) k++; return k; }
function show(e) {
  const r = cv.getBoundingClientRect(); mouse = {x: e.clientX - r.left, y: e.clientY - r.top};
  const hit = at(mouse.x, mouse.y);
  let text = '';
  if (hit && hit.span) {
    const s = hit.span; text = D.stages[s[4]] + (s[7] ? '  [' + s[7] + ']' : '') + '\n';
    if (D.nodes[s[5]]) text += 'node  ' + D.nodes[s[5]] + '\n';
    if (s[6] !== '') text += 'item  ' + s[6] + '\n';
    text += 'took  ' + ((s[2] - s[1]) * 1000).toFixed(2) + ' ms   (at ' + s[1].toFixed(3) + ' s)';
    if (s[8]) text += '\n' + s[8];
  } else if (hit && hit.chart && S.t.length) {
    const k = sampleAt(hit.t); text = hit.chart.name + ' at ' + S.t[k].toFixed(2) + ' s';
    for (const sr of hit.chart.series) { const v = sr.v[k]; text += '\n' + (sr.label || 'value') + ': ' + (v == null ? '-' : (+v).toFixed(2)); }
  } else if (mouse.x > LABEL) text = fmt(T(mouse.x));
  tip.style.display = text ? 'block' : 'none'; tip.textContent = text;
  tip.style.left = Math.min(mouse.x + 14, W - 300) + 'px'; tip.style.top = (mouse.y + 14) + 'px';
}
cv.addEventListener('wheel', e => {
  e.preventDefault(); const r = cv.getBoundingClientRect(), t = T(e.clientX - r.left), k = Math.pow(1.0015, e.deltaY);
  view = [t - (t - view[0]) * k, t + (view[1] - t) * k]; draw();
}, {passive: false});
cv.addEventListener('mousedown', e => { drag = {x: e.clientX, view: view.slice()}; cv.style.cursor = 'grabbing'; });
window.addEventListener('mouseup', () => { drag = null; cv.style.cursor = 'grab'; });
window.addEventListener('mousemove', e => {
  if (drag) { const dt = (e.clientX - drag.x) / (W - LABEL - RIGHT) * (drag.view[1] - drag.view[0]);
    view = [drag.view[0] - dt, drag.view[1] - dt]; }
  if (e.target === cv || drag) { show(e); draw(); }
});
cv.addEventListener('mouseleave', () => { mouse = null; tip.style.display = 'none'; draw(); });
const reset = () => { view = [0, Math.max(D.end, 1e-3)]; draw(); };
cv.addEventListener('dblclick', reset); document.getElementById('reset').addEventListener('click', reset);
window.addEventListener('resize', draw);
const lg = document.getElementById('legend');
D.stages.forEach((n, i) => { const s = document.createElement('span'); s.innerHTML = '<i></i>'; s.firstChild.style.background = D.colors[i];
  s.appendChild(document.createTextNode(n)); lg.appendChild(s); });
const g = document.createElement('span'); g.innerHTML = '<i style="background:#00a651"></i>top stripe: on the GPU'; lg.appendChild(g);
draw();
</script></body></html>
"""


def write_report(folder, out=None):
    """Write the page for run folder `folder` (to `out`, default <folder>/report.html). Returns (path, summary)."""
    run = load_run(folder)
    s = summarize(run)
    data = json.dumps(_page_data(run, s), separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(f"{s['label'] or 'run'}: {s['wall']:.2f} s")
    sub = html.escape(f"{s['run']} · {s['cpus']} logical cores"
                      + (f" · {s['gpu']['name']}" if s["gpu"] and s["gpu"]["name"] else ""))
    page = (PAGE.replace("__TITLE__", title).replace("__SUB__", sub).replace("__SUMMARY__", _summary_html(s))
            .replace("__DATA__", data))
    path = Path(out) if out else Path(folder) / "report.html"
    path.write_text(page, encoding="utf-8")
    return path, s


def trace_dir():
    """As engine.trace.trace_dir (not imported: the engine package loads OpenCascade)."""
    base = os.environ.get("DESIGN_ENGINE_TRACE_DIR")
    if base:
        return Path(base)
    root = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".cache")
    return Path(root) / "DesignEngine" / "trace"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="a run folder (default: the newest in the trace folder, "
                                              "%%LOCALAPPDATA%%\\DesignEngine\\trace or DESIGN_ENGINE_TRACE_DIR)")
    ap.add_argument("--match", help="without a folder: the newest run whose name has this in it (e.g. final, "
                                    "export, optimize)")
    ap.add_argument("--list", action="store_true", help="list the runs in the trace folder and stop")
    ap.add_argument("-o", "--out", help="where to write the page (default: report.html in the run folder)")
    args = ap.parse_args()
    if args.list or not args.folder:
        runs = sorted(p for p in trace_dir().glob("*") if (p / "run.json").exists() and
                      (not args.match or args.match in p.name))
        if args.list:
            print("\n".join(str(p) for p in runs) or f"no runs in {trace_dir()}")
            return
        if not runs:
            sys.exit(f"no traced runs{' matching ' + args.match if args.match else ''} in {trace_dir()}")
        args.folder = runs[-1]
    if not (Path(args.folder) / "run.json").exists():
        sys.exit(f"{args.folder} is not a trace run folder (no run.json)")
    path, s = write_report(args.folder, args.out)
    print(summary_text(s))
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
