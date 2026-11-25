# Organic optimized parts: plan

2026-11-11. Status: steps 1 (Extrude along), 2 (polished body), 3 (fillet and exact bosses) and 4 (even-out
flow) are built, and after them Edges (round or chamfer), Walls and a sample shaped like the reference, see
"Status" at the end;
step 5 (planar frames, optional) is not started.

## Short answer

The reference is a 2.5D part: an exact CAD plate and bosses, a web whose members all run straight through its depth (flat sides), and a polished surface with round fillets where web meets plate. Our Topology Optimization is fully 3D and keeps the voxel look, so it makes bone-like 3D shapes instead. Four additions close most of the gap, in this order:

1. **Extrude constraint** in Topology Optimization: densities equal along one direction, so members become flat-sided prisms. This alone gives the reference's character.
2. **Polished body**: the filtered field plus the cubic smooth field, already prototyped and measured on 2026-11-03, never built.
3. **Fillet to Keep and smooth skins**: a rounded union where the optimized shape meets Keep, and bosses as true cylinders instead of voxel skins.
4. **Even-out flow**: uniform member widths and filleted joints (prototyped, needs the GPU and a loose-piece check).

A planar mode for Frame Optimization is a fifth, optional route that gives the most literal truss look. The plate needs no new "plates" feature: it is a CAD Keep body, not something to optimize.

## Where the engine stands

| Trait in the reference | Engine today | Gap |
| --- | --- | --- |
| Web members run through the whole depth, flat sides | Fully 3D densities; only Symmetry ties elements together | Large: no extrude constraint |
| Flat plate, countersunk holes, crisp edges | Keep body is unioned exactly (`nodes_fea.py:700`); crop gives exact CAD faces; dual contouring keeps sharp edges at export | Small: works if the plate is modelled as Keep |
| Fillets where web meets plate | Plain `F.union(out, keep_body)`: a sharp inside corner | Medium: needs a rounded union |
| Round bosses around holes | Skin is whole voxel elements, smoothed; FINDINGS says the bosses stay stepped | Medium: needs an exact skin distance |
| Polished, stripe-free surface | Body = distance to an h/2 marching-cubes mesh: element stripes and facets show | Medium: smooth field prototyped, not built |
| Even member widths, rounded opening corners | Smoothing 2 rounds broad faces; thin members stay beaded | Medium: even-out flow prototyped, not built |
| Truss layout, triangular openings | Topology finds it in 2D-like cases; Frame Optimization makes trusses but 3D, round bars by default | Optional: planar frame mode |

## The plan

Each step stands alone and is judged by an analysis of the shown body, not by a picture (the FINDINGS lesson from Smoothing 2). Test parts: the wall and bolted brackets, plus a new recipe shaped like the reference (a 200 x 57 x 28 mm bar, plate on top as Keep, screws as supports, a load under it).

1. **Extrude constraint (size M).** New Topology Optimization input *Extrude along*: none, x, y, z (later a picked face's normal).
   - Sensitivities dc, dv and dg are averaged along each grid line through the free elements, in the same hook as `Problem.symmetric`, so free densities stay equal along the line from the uniform start.
   - The design space can still be tapered: the crop by its CAD faces gives the reference's sloped web sides.
   - Max thickness and Max section then act in the plane only; the local-density neighbourhood becomes 2D.
   - In the key only when set. Judge: every column of free elements identical, stiffness vs the 3D result at the same mass.
2. **Polished body (size M).** Build the 2026-11-03 prototypes as the node's body.
   - Mesh the filtered field xt instead of the near-binary densities (same solid elements, surface placed between them).
   - Use density\_mesh's final samples as a cubic B-spline field (`spline_filter` once, `map_coordinates` order 3, CuPy on the GPU) instead of the distance to its h/2 mesh. This removes the stripes and facets.
   - Push near-solid pieces apart from the part out before interpolating (islands of 0.01-0.1 mm3). Bump `TOPO_VERSION` only if the state changes.
   - Judge: same mass and deflection within 1%, zero loose pieces in the Final view.
3. **Fillet to Keep and smooth skins (size S-M).** New input *Fillet to Keep* (mm, 0 = sharp).
   - Replace `F.union(out, keep_body)` with the rounded union the Booleans and frames already use (`union_edged`, G2 `bulge`).
   - Make the skin around supports and loads its own exact distance (skin minus the distance to the faces), unioned with the same fillet, so bosses come out as true cylinders.
   - Judge: no sharp inside corner along the plate, bosses round in a section view, stress peak at the joint not higher.
4. **Even-out flow (size M).** Move the level set inward by its smaller principal curvature after the blur and before `thicken` (soft min, gated where |grad| is under 0.5, about 4 h2 of time).
   - Struts and plates keep their thickness; bulges, necks and stubs even out and joints get fillets.
   - Needs the GPU (200 s on the CPU at fine) and a loose-piece test before shipping.
5. **Planar frames (size M-L, optional).** Frame Optimization with *In plane*: nodes on the design space's mid-plane, rectangle bars whose depth is the web's and whose width is sized, out-of-plane buckling in the beam check.
   - Join Frames already fillets a frame into a CAD body, so the plate stays a CAD part.
   - Gives the most literal truss look; worth doing only if steps 1-4 do not satisfy.

Later, by need: the stress-driven growth stage from FINDINGS (CAO), which evened the bolted bracket's stress and made it 15-19% stiffer at the same mass.

## What works today

You can get partway there with existing nodes:

- Model the plate, countersinks and bosses in SolidWorks as one body and give it to Topology Optimization as **Keep**; make the design space only the web region below it.
- Make the web space thin (the reference is about 8-10 mm at the web) so members have little room to wander in depth, and set **Symmetry** about its mid-plane.
- Use **Members: plates and struts** (struts only never settles, FINDINGS 2026-11-03), **Smoothness 0.5** for cleaner edges, **Smoothing 2**, and **Fine** detail for the final.
- Export with **Mesh from Implicit Body, method dual contouring** so the plate's edges and countersinks stay sharp.

What this cannot give yet: flat web sides, the fillet into the plate, and round bosses. Steps 1 and 3 fix those.

## Status

- **Step 1, Extrude along: built 2026-11-11** (world axes x, y, z; a picked face's normal is not done).
  Topology Optimization's input *Extrude along*, `topology.EXTRUDE`, `Problem.along`; tests in
  `tests/test_topology.py` and `tests/test_gpu.py`; sample `recipes/topo_hanger_extruded.json`
  (`samples/hanger_web.step` from `tools/make_hanger_sample.py`). Measurements and pictures: docs/FINDINGS.md,
  "Extrude along", and docs/organic_parts/. Differences from the plan above: the filter is in the plane too
  (with the 3D cone a line's densities still differed at the two faces and the eroded design lost those
  layers); Members and Max section are turned off with a note instead of acting in the plane (a wall through
  the depth has no section to limit; Max thickness is its width); the sample is held by its two end screws,
  since held by all six the loads go straight up to the plate and no web forms.
- **Step 2, polished body: built 2026-11-12.** The node's body is `fields.sampled_body` of
  `topology.density_field`'s samples: a cubic B-spline of the smoothed samples, found in the optimizer's filtered
  field, which new states save as `field` (no `TOPO_VERSION` bump: older states have none and use their
  densities, so they get the smooth body but their truest surface only when optimized again). Tests in
  `tests/test_fields.py`, `tests/test_topology.py` and `tests/test_gpu.py`. Judged as planned: one piece on the
  hanger and both brackets, mass within 0.9% and deflection within 1.7% (the bolted bracket; the others within
  0.3% and 0.65%). Measurements: docs/FINDINGS.md, "The polished body"; picture docs/organic_parts/polished_body.png.
  Differences from the plan above: the samples are the spline's control values instead of a spline through them, so
  there are no ringing islands to push out; the values are rescaled to mm near the surface (the blurred distance
  ran at 0.64 per mm across thin members, which would have made textures and offsets too large there).
- **Step 3, fillet and exact bosses: built 2026-11-12.** Input *Fillet* (mm, 0 = sharp, not in the key).
  The body is the members (state key `members`, `topology.members_field`), the skin as an exact distance
  (`topology.skin_body`) and the Keep body, joined with a curvature-continuous blend and then cropped. Tests in
  `tests/test_topology.py` and `tests/test_gpu.py`; the sample recipe has Fillet 1.5. Judged as planned: bosses
  0.02-0.04 mm off their cylinder (0.29 before), stress on the hanger 4% lower with a 2 mm fillet, one piece on all
  three parts. Measurements: docs/FINDINGS.md, "Exact bosses and a fillet"; picture
  docs/organic_parts/fillet_and_bosses.png. Differences from the plan above: the frames' G2 blend instead of
  `union_edged` (which pitted on the sampled field); a hole's boss is a ring all round also under a bearing load
  (10% more mass on the bolted bracket, whose pin eyes were open before); corners between two faces of the design
  space itself are not filleted.
- **Step 4, even-out flow: built 2026-11-12.** Input *Even out* (x, default 1, 0 = off, not in the key):
  `topology.even_out`, the flow by the surface's smaller curvature, after `thicken`. Tests in
  `tests/test_topology.py` and `tests/test_gpu.py`. Judged on the hanger (extruded and free) and both brackets:
  one piece each, mass within 0.6%, deflection +1.5% / +1.0% / +0.1% and -7.3% (bolted bracket). Measurements:
  docs/FINDINGS.md, "Even out"; picture docs/organic_parts/even_out.png. Differences from the plan above: with
  Extrude along the flow is in the plane (the 3D one cannot even a flat side's bulges); the gate is on ridges
  themselves, not on the slope (which froze thin members); holes that shrink to a sample close all the way; half
  the prototype's time by default (4 elements^2 halved the hanger's small openings); it runs after `thicken`, on
  the GPU in about a second, with no kernel of its own.
- **After step 4, 2026-11-12: Round edges and a new sample.** Put beside the reference
  (docs/organic_parts/reference_look.png) the sample still did not look like it, and not because of its surface:
  the reference's edges are all round, its web is a truss of thick members, and the part is a wedge with sloped
  ends and countersunk holes. Built: input *Round edges* (mm, 0 = sharp, not in the key; `fields.intersect_edged`:
  the rims where members meet the design space's faces), and the sample again: the reference's part
  (`tools/make_hanger_sample.py`), held by its four outer screws, each shaft loaded alone three ways (nine load
  cases: that is what makes a truss), Keep 50%, Min thickness 5.5, Fillet 4, Round edges 2. Measurements and the
  five load trials: docs/FINDINGS.md, "Round edges, and a hanger like the reference".
- **The reference is not 2.5D** (the user, 2026-11-12): its members lie at different depths and it has an opening
  through the wedge seen from the side, so "Short answer" above is wrong about that, and Extrude along is an option
  for parts that should be extrusions, not the way to this look. `recipes/topo_hanger.json` is the sample free in
  3D, its loads 12 mm in front of the web: a flat-faced truss in front, struts behind it. The extruded recipe is
  kept beside it. FINDINGS, "The reference is not an extrusion".
- **Square profiles with chamfers** (the user, 2026-11-12, of the free 3D sample: "the picture has a more
  refined shape with square profiles with nice chamfers"). The reference is a hollow wedge whose walls are cut
  into trusses. Built: input *Walls* (mm: the design space hollow, members strips of its walls) and *Edge shape*
  round / chamfer for *Edges* (was Round edges). `recipes/topo_hanger.json` now uses them (Walls 6, Edges 1.5
  chamfer). docs/FINDINGS.md, "Walls and chamfers".
- **Bars along the optimized members** (the user's idea, 2026-11-13: let the optimization say where the bars
  go and draw the bars ourselves). Built: *Members as: bars* and *Bar size* (`engine/fea/bars.py`): the members'
  centre lines traced into straight bars with a square section, flat faces, filleted joints and broken rims.
  It does what step 5 was meant for, from the topology result instead of a truss layout. docs/FINDINGS.md,
  "Bars along the optimized members". Then (2026-11-14, the user: the bars should follow the optimized shape's
  size and form better) *Bar form: follow*, the default: a bar runs along its member's own curve and is as wide
  as the member along it. `recipes/topo_hanger.json` uses it: 4% lighter and 10% less stiff than the optimized
  shape, where straight 7 mm bars were 25% and 60%. FINDINGS, "Bars that follow their members".
- Step 5 (planar frames): not started, and after the remark above not on the way to the reference. What is left of
  the reference's look: five openings where it has eight
  (the loads decide), a Keep plate's lower edge over an opening stays sharp, waves longer than about 8 elements on
  members' sides (Even out 2-3 takes them, and small openings with them).

## Open questions

- Extrude along a picked face's normal too (for parts not lined up with x, y or z)?
- Is a planar frame route (step 5) wanted at all, or is the topology route enough?
