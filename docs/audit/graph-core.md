# graph-core audit

Area: engine/graph/core.py, workers.py, engine/server.py, frontend store.ts / graphConvert.ts / api.ts.
Scripts and outputs: `SCRATCH/audit/graph-core/` (t1..t10 Python, f1_store.ts for the frontend store, run in plain Node
via esbuild; no server was started, nothing in the repo was changed).
All Python runs went through guard.py. The 2-worker pool test (t10) was killed by the guard at 3.0-3.6 GB
(main + 2 spawned workers), so worker-pool behaviour is SUSPECTED from code only.

Findings are ranked: bugs first, then UX / improvements. New findings are appended below as they are confirmed.

---

### [BUG] Editor store: after an engine restart the UI silently ignores every event and never shows new views
Severity: high · Status: VERIFIED (real store code run in Node: `f1_store.ts`, section F1)
Evidence: `frontend/src/store.ts:392` drops events with `e.run < s.run.id`; `frontend/src/store.ts:475` (`setView`) drops views with
`version <= existing version`; `engine/server.py:243-246` start `run_id` and `view_version` at 0 on every start; `frontend/src/api.ts:204-206`
reconnects the WebSocket by itself. Repro output:
```
before restart: run = {"id":57,...,"message":"Done in 1.2s"}  views.m.version = 120
after restart, first run events (run=1): run = {"id":57,...}  status = {}  log last = Update finished in 1.2s
new view (version 3, faces 20) accepted? views.m.faces = 10   (10 = old view kept)
```
So after `python app.py` is restarted (CLAUDE.md: "Python changes need a server restart") with the page left open, the new server's
`run_start`, `node_*`, `problems`, `run_done` (including errors) events are all treated as "stale", node cards keep old status, error
messages never reach the log, and every `view_ready` fetch is thrown away because its version (1, 2, 3...) is lower than the old
one (120). The 3D view keeps showing the pre-restart shape until the page is reloaded. If the restart happened mid-run, `run.running` stays
true forever (spinner + Cancel button).
Why it matters to the user: this is exactly "I changed something and nothing happens / it shows the old result", right after the
workflow the project documents (edit Python, restart server, keep the browser pane open).
Suggestion: give the server a boot id (or start `run_id` / `view_version` from `int(time.time())`) and send it in every event or
at WebSocket connect; in `ws.onopen` / on a changed boot id reset `run`, `status`, `views` versions and `slices`/`paints`. Cheapest:
`self.run_id = self.view_version = int(time.time())` in `Session.__init__`.

---

### [BUG] Memory watchdog halves the RAM array cache every 0.5 s while memory is "tight" and nothing ever restores it
Severity: medium · Status: VERIFIED (`t5_watchdog.py`, free RAM faked at 2.5 GB)
Evidence: `engine/graph/core.py:570-576` (`ARRAY_CACHE.set_limit(ARRAY_CACHE.max_bytes // 2)` inside the 0.5 s loop, no floor, never
raised again); the worker side has a 16 MB floor (`workers.py:91`) but the main process has none. Output:
```
ARRAY_CACHE.max_bytes at start: 1591 MB
after 1 s: 795 MB   2 s: 199 MB   3 s: 50 MB   ...   8 s: 0.05 MB
memory free again, watchdog stopped: 0.05 MB (was 1591 MB);  after a new Evaluator.run(): 0.05 MB
```
It is only restored when a StickyPool is created or terminated (`workers.py:68`, `:169`), i.e. only on runs that use worker processes.
Why it matters: on this PC (other programs 15-21 GB, FREE_TIGHT = 3 GB, or used > 85% of a limit that is 75% of what was free at
start) one tight moment during a heavy run switches the RAM cache off for the rest of the session. Everything then goes through the
SSD cache: correct but visibly slower ("the editor got slow after the big export").
Suggestion: halve at most once per tight episode (the `shrunk` flag already exists but is not used to guard the halving), keep a
floor (e.g. 64 MB), and restore `ram_cache_budget()[0]` when the run ends (in `Evaluator.run`'s `finally`) or when free RAM is back above FREE_TIGHT + margin.

---

### [BUG] Evaluator keeps the results of deleted nodes (and of replaced recipes) alive forever
Severity: medium · Status: VERIFIED (`t9_leak_robust.py` L1: two 40 MB results of nodes that no longer exist are still alive after the graph changed)
Evidence: `engine/graph/core.py:497` `_cache` is only ever overwritten per node id; `release_memory` (`:512-520`) only pops `mesh.*` /
`output.*` nodes; `Session._run_job` does `self.evaluator.graph = graph` without pruning.
```
cache keys after first graph: ['big1', 'big2']
cache keys after switching to a graph without them: ['big1', 'big2', 'other']
40 MB results still alive: 2 of 2
```
Why it matters: STEP meshes, Voronoi foam arrays (`Shared`), CAD distance fields etc. of nodes the user deleted, or of the previous
recipe (nodes with ids not reused by the next recipe), stay in RAM until the server restarts. FINDINGS notes an idle editor holding 10.6 GB.
Suggestion: in `_run` (or when `Evaluator.graph` is set) drop `_cache` entries whose node id is not in `self.graph.nodes` or whose node type
changed; also drop the entries of nodes that were not needed by the last run after N runs (hidden `__view__` nodes of hidden bodies).

---

### [BUG] One failing or half-wired visible node stops every other visible body from refreshing
Severity: medium · Status: VERIFIED (`t4_build.py` B4, B5)
Evidence: `engine/server.py:411-434`: views are only published after `self.evaluator.run(targets)` returned for ALL targets; a GraphError
anywhere ends the job with `run_done ok=False`. `build_run_graph` (`:131-158`) adds every visible node as a target without the `_wired`
check that preview nodes get. Output:
```
visible half-wired node 'half' + good node 'ok':  run_done ok=False "invalid graph: half: required input 'body' is not set"  (no view_ready)
visible order ['ok','bad']: node_done ok, node_error bad, run_done           (no view_ready for ok)
visible order ['bad','ok']: node_error bad, run_done                         ('ok' is not even computed)
```
Why it matters: you toggle the eye on a new node you have not wired yet (or one node's parameters are invalid) and the whole 3D view
freezes on the previous result for all other bodies, with no sign in the viewport that it is stale (the message sits on the failing node
card only). Also the frontend keeps `views[id]` of the failed node as it was.
Suggestion: (a) treat visible nodes like preview nodes: skip those where `_wired()` is false (they still get their problem marker from
the `problems` event); (b) evaluate targets one by one (or catch the GraphError per target) so independent visible bodies still get their
new view, and mark a failed node's old view as stale (dim / "outdated" badge) instead of leaving it looking current.

---

### [BUG] Draft / Medium quality does not scale `voxel` when it comes from a link (Number node or group setting)
Severity: medium (low for the current recipes: none links voxel) · Status: VERIFIED (`t4_build.py` B1, B2)
Evidence: `engine/server.py:108-112`: `if not _is_link(voxel): inputs["voxel"] = float(voxel) * multiplier`.
```
draft   literal voxel -> 2.0; linked voxel -> {'link': 'vox.out'} (Number node value 0.5)
medium  literal voxel -> 1.0; linked voxel -> {'link': 'vox.out'} (Number node value 0.5)
group with a Number setting feeding the inner mesher's voxel: link kept as is (not multiplied by 4)
```
Why it matters: the Mesh from Implicit Body doc and the Draft button promise "Draft = 4x coarser", and the Number node / group
"setting" is the documented way to have one voxel size in one place. A linked voxel silently makes the Draft preview full resolution (slow, memory heavy, the very
case Draft exists to avoid). Nothing tells the user.
Suggestion: when voxel is a link, insert a hidden `math.multiply`-like scaling in `build_run_graph` (or give `Evaluator` a `preview_scale` that
`mesh.from_implicit` reads), so linked and literal voxels behave the same. Same for `keep` / `tolerance` already forced at non-final quality (those are OK).

---

### [BUG] An exception after the evaluation (or while checking problems) kills the run thread without `run_done`: the editor shows "Updating..." forever
Severity: medium · Status: VERIFIED (`t9_leak_robust.py` R1, R2)
Evidence: `engine/server.py:416` (`graph.problems()` is outside any try/except; `_split` raises GraphError for a link without a port) and
`:443-466` (`pack_meshes`, `_paint` view packing has no try/except around `pack_meshes`). Output:
```
R1  Exception in thread: engine/server.py line 416 ... GraphError: link 'n1' must look like 'node.port'    events: run_start, queue
R2  thread exception AttributeError: 'int' object has no attribute 'faces'   events: run_start, view_ready(ok)   run_done sent? False
```
The editor's `run.running` stays true (spinner, Cancel button instead of Update, node statuses frozen) until the next run starts.
Why it matters: any unexpected bug in view packing (empty/odd mesh, decimation dependency missing, a Many with an odd item) turns into a hang with
only a traceback in the console window the user does not look at.
Suggestion: wrap the body of `_run_job` in `try/except Exception` that publishes `run_done ok=False error=...` (the same fallback the
`evaluator.run` branch already has), and wrap each view/slice publication so one bad view does not drop the rest.

---

### [BUG] An unwired Export (or Preview Image) node blocks the whole export
Severity: medium · Status: VERIFIED (`t9_leak_robust.py` E1)
Evidence: `engine/server.py:126-127`: export targets are all side-effect nodes with no `_wired` filter (preview nodes at `:129` do get it).
```
run_done ok=False error="invalid graph:\n  w2: required input 'mesh' is not set"   (w1, which is wired, is not written either)
```
Why it matters: a second Export Mesh node placed for later, or a Preview Image whose mesh link was removed, makes the Export button
fail for everything. The message is clear, but the wired export is still not written.
Suggestion: filter export targets with `_wired` like preview nodes; log "skipped Export Mesh 'w2': not connected" instead of failing.

---

### [BUG] Picture / Text fields: Field.key ignores `angle`, so two of them that differ only in angle are the same sub-expression
Severity: medium (silent wrong result, rare setup) · Status: VERIFIED (`t8_keys.py`) · Area: patterns (found while checking key collisions)
Evidence: `engine/picture.py:173-176` digest = (key, image hash, width, ox, oy, invert, band, mapping, centre) - no angle, no placement axes.
```
p0 key == p45 key (angle 0 vs 45): True
independent |p0 - p45| max: 1.0     diff node (math.subtract p0 p45) max: 0.0
```
`Context.eval` memoises by `field.key` (`engine/fields.py:86-104`), so `p0 - p45` (or any graph that uses both) evaluates the second as a copy
of the first. A single picture is fine (the evaluator re-runs the node and the field is not stored in ArrayCache).
Why it matters: a logo placed twice on one face with different rotations, or a rotated copy used as mask, comes out wrong without any message.
Suggestion: add the rounded placement axes (normal/right/up or axis/radius direction) and `angle` to the digest; audit other `_digest(...)`
calls for parameters used by `fn` but missing from the key (grep `_digest(` in engine/).

---

### [BUG] A queued Optimize job silently drops the preview that was waiting
Severity: low (mitigated by the `optimized` event) · Status: VERIFIED (`t3_session.py` scenario 2)
Evidence: `engine/server.py:328-329`: `_launch` clears `self.pending` for every job whose mode is not "export", including "optimize" (a LONG_JOB).
```
queue: E1 running, E2 queued, O1 (optimize) queued, preview P pending
... E2 done -> run_start O1 (queue shows preview_waiting=False)  -> P never runs ("preview P ever ran? False")
```
The editor does bump `graphVersion` on `optimized` (store.ts:391), so a new preview follows, but only if the browser is connected, and the
"preview waiting" hint disappears while the optimization is still running.
Suggestion: `if job["mode"] not in LONG_JOBS: self.pending = None`.

---

### [UX] Undo folds visibility toggles and node moves into the neighbouring undo step
Severity: low-medium · Status: VERIFIED (`f1_store.ts` F3, with 70 ms pauses between actions)
Evidence: `store.ts:355` (`toggleVisible`) and node-position changes (`onNodesChange`, `:283-284`) never call `remember`, but snapshots store `visible`
and node positions. Output: setInput(a=5) -> hide m -> move a -> saveSelection gives `past=2`; the second undo restores `a=3`, `visible=[m]`
and `a.pos=(0,0)` in one go (three user actions in one step). Also `addGroupPort` and `setPicking` are not undoable at all.
Why it matters: Ctrl+Z "does too much" (the eye comes back, nodes jump back) or the user cannot undo showing / hiding a body.
Suggestion: call `remember(s, 'visible')` in `toggleVisible` and `remember(s, 'move')` on drag end (coalesced), or at least do not restore `visible`/positions from a snapshot taken for another kind of edit.

---

### [IMPROVEMENT] Literal values are never checked against a port's `choices` or type in `Graph.problems()`
Severity: low-medium · Status: VERIFIED (`t7_choices.py`)
Evidence: `engine/graph/core.py:397-435` checks links, unknown ports and required inputs only. Most nodes validate their own enum, but with
different wording and sometimes the wrong name, and three accept anything:
```
math.remap auto='maybe'                OK (silently)
field.cylindrical component='bogus'    OK (silently)
mesh.merge fuse='maybe'                OK (treated as "no")
lattice.tpms pattern='bogus'           "type must be one of gyroid, ..."   (the port is called Pattern, not type)
lattice.tpms mode='bogus'              "mode must be one of sheet, solid, solid (other side)"
implicit.union edge='bogus'            "Edge must be one of: sharp, round, chamfer"
mesh.from_implicit voxel='abc'         "'<=' not supported between instances of 'str' and 'int'"   (raw Python error)
math.remap from_low == from_high       "float division by zero"
```
Suggestion: in `problems()`: for `port.choices` check the literal is one of them ("Mode: 'bogus' is not one of sheet, solid, ..."), for `number` / `int` / `scalar`
ports check literals are finite numbers, and report it on the node before any run; recipes edited by hand or saved before an enum was renamed then fail with a clear message.

---

### [IMPROVEMENT] STEP replaced on disk keeps its stale result only if both mtime AND size match (evaluator signature uses mtime only)
Severity: low · Status: VERIFIED limitation (`t2_stale.py` F), not realistic in practice
Evidence: `core.py:477-480` `_literal_signature` = (path, mtime); the on-disk STEP cache key (`cache.py:195`) uses size + mtime_ns. A file replaced with
a new mtime is re-imported correctly (checked); a file copied over with the old mtime restored is not.
Suggestion: use `(mtime_ns, size)` in `_literal_signature` too (same as the disk cache key); also covers Picture files (`pattern.picture` path port).

---

### Checked and fine (no finding)
- Signatures: literal change upstream recomputes exactly the downstream nodes; a rewired link recomputes the target and everything below; 0.1 vs 0.10000001 differ;
  `1` vs `1.0` recompute (harmless); group node settings (override, default, edited group definition, replaced inner node) recompute the right copies only;
  redefined face selection name changes the resolved literal and so the signature; STEP file changes recompute; Many length changes are handled (`t1`, `t2`).
- A failing node: the error names the right node (also inside groups, shown on the group node as `inner: message`), downstream nodes do not run, and the next
  run retries after the fix (`t1`, `t4` B6).
- Cancel: cancelling mid-run and re-running left no half-cached state; the "cancel lost before `Evaluator.run` resets the flag" race could not be reproduced
  in 30 trials (`t3` scenario 3).
- Preview vs export queue ordering is right: a preview during an export waits and only the newest runs afterwards; previews never cancel exports (`t3` scenario 1).
- Group round trip in the store (create, edit/finish, ungroup) keeps all links and the recipe JSON identical (`f1_store.ts` F2); copy/paste/duplicate ids and links are right (F5);
  deleting a node clears its view/slice/paint/marker entries (F4).

---

### [BUG] The editor drops unknown keys on save (CLAUDE.md says they are preserved) and can keep a removed port name that it cannot show or delete
Severity: low-medium · Status: VERIFIED (`f2_store.ts`, real store code)
Evidence: `graphConvert.ts:63-77` reads only `type / inputs / pos / visible` and `:82-96` rebuilds the node from those four; `store.ts:247-262` keeps only
`groups` and `selections` from the top level. CLAUDE.md ("Unknown keys are preserved, so UI data lives in the same file") is true for the engine
(`Graph.to_dict`) but not for the editor. Output:
```
top-level keys in : version,meta,notes,nodes,groups      out: version,nodes,groups
node a keys in    : type,inputs,pos,comment,color        out: type,inputs,pos
node b inputs out : {"b":2,"bogus_old_port":5,"a":{"link":"a.out"}}   (a port that no longer exists is carried along)
```
No current recipe or preset has extra keys, so nothing is lost today. Two consequences: (1) any note / colour / doc field added to a recipe by hand or by
Claude is erased by the first Save from the editor; (2) after a node's port is renamed or removed in the code, an old recipe keeps the old name in
`inputs`; the engine then reports `<node type> has no input 'old_name'` on that node (`core.py:410`, verified in `t7_choices.py`), which blocks every run that
needs the node, and the editor has no widget for that input, so it cannot be cleared except by deleting and re-adding the node.
Suggestion: keep the whole node object in `data` (spread the unknown keys back in `flowToRecipe`) and the unknown top-level keys in the store; in the
engine either ignore literal inputs that do not exist (log a warning once per node) or drop them in `recipeToFlow` when the spec has no such port.

---

### [IMPROVEMENT] Recipe save is not atomic
Severity: low · Status: SUSPECTED (from code)
Evidence: `engine/server.py:619` `p.write_text(...)` writes the recipe in place (the autosave right above uses tmp + `os.replace`, `:635-638`). The recipes live in the repo and
partly on OneDrive; a crash, a full disk or OneDrive locking the file mid-write leaves a truncated JSON, and the editor then cannot open it (`/api/file` returns 400).
Suggestion: same tmp + `os.replace` pattern, and keep one `.bak` of the previous version.

---

### [BUG] The server keeps the packed 3D view of every node that was ever shown (hidden or deleted nodes too)
Severity: medium (memory, grows during a session) · Status: VERIFIED (`t11_views_kept.py`)
Evidence: `engine/server.py:244-245`, `:262`, `:452`, `:456`, `:485`, `:517`: `Session.views` / `view_faces` are only ever overwritten, never pruned; neither
`release_memory()` nor `watch_idle_memory` touches them. Output: after showing a, then hiding a and showing b, then opening a recipe with only c, the dict still holds
`{'a': 1474592 bytes, 'b': 368672, 'c': 92192}`. A view is up to `DISPLAY_MAX_FACES` = 15M triangles, about 270 MB packed; `view_faces` also holds the CAD
body objects and `paint:*` / `slice:*` entries hold per-vertex arrays.
Why it matters: together with the Evaluator cache above, this is the reason an idle editor can sit on gigabytes after a design session; on this PC (page file on an HDD) that is
what freezes the machine.
Suggestion: in `_run_job`, after publishing, drop `views` / `view_faces` / `paint:` / `slice:` entries whose node is not in the job's `views` (or not in the graph) and
call the same cleanup from `watch_idle_memory`.

---

### [BUG] A superseded preview that is already publishing its views is not interrupted: the next request waits for it
Severity: medium (latency after an edit, worst on big parts) · Status: VERIFIED (`t12_publish_cancel.py`, packing slowed to 2 s to stand in for a 12M-triangle view)
Evidence: `engine/server.py:366-369` `cancel()` only sets the evaluator flag and joins the thread; after `evaluator.run` returned, `_run_job` (`:443-489`) never looks at the
flag, and its `if run_id != self.run_id: return` guards (`:444`, `:470`) are dead in practice because `run_id` only changes in `_launch`, after the join. Output:
```
second start() (P2) returned after 1.56 s although P1 was already obsolete
 0.00s run_start P1 ... 2.00s view_ready P1 ... 2.06s run_start P2 ... 4.07s run_done P2
```
Why it matters: with the lamp (a 15M-triangle display copy is decimated for ~24 s per FINDINGS) an edit made while the previous preview is still packing waits that whole time before
the new run even starts, and the HTTP request of `/api/run` is blocked meanwhile.
Suggestion: in the view / slice / paint loops check `self.evaluator._cancel` (or a per-job flag set by `cancel()`) and return early; do the same inside `display_arrays` per mesh.

---

### [UX] Cancel gives no feedback and can take as long as the step it interrupts
Severity: low · Status: SUSPECTED (from code)
Evidence: `engine/server.py:366-369`, `/api/cancel` (`:754`) blocks in `thread.join()` until the running node reaches a `check_cancel` / `report_progress` call; nodes in
non-interruptible steps (a block's marching cubes, fuse, pyfqmr) delay it. The Cancel button stays as is, the status line keeps saying "Updating".
Suggestion: publish a `cancelling` event right away and show "Cancelling..." on the toolbar; return from `/api/cancel` without joining.

---

### [IMPROVEMENT] Parallel (worker) steps: a failed item is reported only after all items finish, and the message does not say which item
Severity: low-medium · Status: SUSPECTED (from code; the 2-worker test was killed by the memory guard at 3.0-3.6 GB, so not run)
Evidence: `engine/graph/workers.py:122-146`: `failure` keeps the first error message, the loop goes on until `done == len(arg_list)`, then raises `WorkerError(failure)` with no
item index; `Evaluator._run` wraps it as `node 'mesh' (...) failed: GraphError: ...`.
Why it matters: one bad petal out of 144 costs the whole run time (an hour) before the error appears, and then you do not know which body.
Suggestion: on the first failure set `self._current.value = -token` (skip queued items), return early, and prefix the message with the item number ("body 17 of 144").
Note for coordination: two spawned workers plus the main process exceeded 3 GB even for tiny bodies, so any audit run with workers>1 needs a higher cap than the 3 GB rule allows.

---

### [IMPROVEMENT] Preview details that do not follow the "Draft = coarse, no decimation" rule
Severity: low · Status: SUSPECTED (from code)
Evidence: `engine/server.py:106-118` only adjusts `mesh.from_implicit` (voxel, keep, tolerance) and `mesh.merge` (fuse). `mesh.reduce` (Reduce Mesh, 700 bytes per triangle, `parallel=True`)
runs at full tolerance in every Draft preview.
Suggestion: pass-through `mesh.reduce` below Final (return the mesh unchanged), like `keep`/`tolerance`.

---

### [IMPROVEMENT] Export runs also build the 3D view of every visible implicit body at 512 cells
Severity: low-medium · Status: VERIFIED (`t4_build.py` B3)
Evidence: `engine/server.py:131-158` handles `visible` in every mode; for an export with `sdf` and `m` visible the targets are `['exp', '__view__sdf', 'm']`. Each hidden `mesh.view`
is meshed at `cells = 512` along the longest side (up to 134M voxels) in a run that already is the heavy final-quality job, and the result is packed and published too.
Suggestion: in export mode skip `views` / `slices` for implicit bodies (keep nodes that are already needed for the export, like mesh outputs).

---

### [UX] Smaller things
- `Group editing`: while a group is open the canvas shows inner nodes, but status / error / range events are mapped onto the group node (`server.py:268-275`), so inner cards never show
  their own errors or timings; an inner node with the same id as an outer node shows the outer node's status. The group card's time is that of its last output node only.
  Suggestion: send inner events with the inner id too and let the store keep `status["group/inner"]`.
- `picking` (store.ts:66): deleting the node a face-pick is set on leaves `picking.source` pointing at a node that no longer exists; the viewport shows "loading <id>..." forever until Esc.
- `skip_duplicates` (nodes.py:73-78) decides "sits exactly on top of another" from the rounded bounding box alone: two different bodies with identical bounds (a tube and its core, a plate and the same plate with
  a pocket) are silently dropped, and face selections "body:face" renumber with it. SUSPECTED; add volume / face count to the key.
- Number fields commit on every keystroke (EngineNode.tsx:29-31, 46): retyping a voxel size passes through values like 0 or 0.1; if you pause more than 450 ms the auto-update runs with it
  (voxel 0 gives a clear error, but 0.01 starts a huge job). Suggestion: commit on Enter / blur for settings that start heavy jobs, or ask before a run with a voxel far below the previous one.
