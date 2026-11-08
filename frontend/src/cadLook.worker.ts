// The CAD look of a shown CAD body, worked out off the main thread (a 2M-triangle body takes about a
// second): vertices split where the CAD face changes, so shading is smooth within a face and crisp at
// its borders, and the lines along the faces' borders (sharp ones, and tangent ones where two faces
// meet smoothly). Triangles keep their order, so picking and the right-click menus (triangle -> CAD
// face) work on the new geometry as on the view's.
import type { EdgeSet, LookRequest, LookResult } from './cadLook'

/** Two CAD faces meet smoothly when most triangle pairs along their border bend less than this (the
 *  engine's `facemask.tangent_groups` takes the median the same way). */
const COS_TANGENT = Math.cos((30 * Math.PI) / 180)
/** Without CAD faces (a decimated view, a list item while it is made): edges that bend more are sharp. */
const COS_CREASE = Math.cos((35 * Math.PI) / 180)
/** A face is flat when its triangles' normals agree this well; two flat faces meet smoothly only when
 *  they bend less than COS_COPLANAR across their border. */
const COS_FLAT = Math.cos((0.5 * Math.PI) / 180)
const COS_COPLANAR = Math.cos((1 * Math.PI) / 180)

// per half-edge (corner c of a triangle: the edge from corner c to the next one)
const SMOOTH = 1, TANGENT = 2, SHARP = 3  // 0: nothing (the edge is handled at its mate)

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<LookRequest>) => void) | null
  postMessage(message: unknown, transfer: Transferable[]): void
}

ctx.onmessage = (e) => {
  const t0 = performance.now()
  const r = computeLook(e.data)
  r.ms = performance.now() - t0
  const edges = (s: EdgeSet) => [s.positions.buffer, s.a.buffer, s.b.buffer, s.parts.buffer]
  ctx.postMessage(r, [r.positions.buffer, r.normals.buffer, r.index.buffer, r.vface.buffer, r.faceIds.buffer,
    r.faceBodies.buffer, ...edges(r.sharp), ...edges(r.tangent)])
}

const next = (he: number) => (he % 3 === 2 ? he - 2 : he + 1)

function computeLook({ id, positions, indices, faces, bodies, parts }: LookRequest): LookResult {
  const T = (indices.length / 3) | 0
  const V = (positions.length / 3) | 0
  const C = T * 3
  const nParts = parts && parts.length ? parts.length : 1
  const partEnd = new Uint32Array(nParts)  // first triangle after each list item
  if (parts && parts.length) { let s = 0; parts.forEach((n, p) => { s += n; partEnd[p] = s }) } else partEnd[0] = T

  // the list item of every triangle and vertex: vertices are only welded within an item (bodies that
  // touch must not share edges)
  const triPart = nParts > 1 ? new Uint32Array(T) : null
  const vPart = nParts > 1 ? new Uint32Array(V) : null
  if (triPart && vPart) {
    let p = 0
    for (let t = 0; t < T; t++) {
      while (t >= partEnd[p] && p < nParts - 1) p++
      triPart[t] = p
      vPart[indices[t * 3]] = vPart[indices[t * 3 + 1]] = vPart[indices[t * 3 + 2]] = p
    }
  }

  // 1. weld points at exactly the same position (the views' CAD meshes are welded already; this keeps
  // a triangle soup from turning every edge into a border)
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, V * 3)
  const weld = new Int32Array(V)
  {
    let size = 1
    while (size < V * 2) size <<= 1
    const table = new Int32Array(size).fill(-1)
    const mask = size - 1
    for (let v = 0; v < V; v++) {
      const x = bits[v * 3], y = bits[v * 3 + 1], z = bits[v * 3 + 2], p = vPart ? vPart[v] : 0
      let h = (Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca77) ^ Math.imul(z, 0xc2b2ae3d) ^ Math.imul(p, 0x27d4eb2f)) >>> 0
      h = (h ^ (h >>> 15)) & mask
      for (;;) {
        const u = table[h]
        if (u < 0) { table[h] = v; weld[v] = v; break }
        if (bits[u * 3] === x && bits[u * 3 + 1] === y && bits[u * 3 + 2] === z && (!vPart || vPart[u] === p)) { weld[v] = u; break }
        h = (h + 1) & mask
      }
    }
  }

  // 2. unit normal of every triangle (0 for a degenerate one) and the angle at each corner (the
  // normals' weights: CAD tessellations are full of long thin triangles, which area weights would favour)
  const tn = new Float32Array(C)
  const ang = new Float32Array(C)
  for (let t = 0; t < T; t++) {
    const i0 = indices[t * 3] * 3, i1 = indices[t * 3 + 1] * 3, i2 = indices[t * 3 + 2] * 3
    const ax = positions[i1] - positions[i0], ay = positions[i1 + 1] - positions[i0 + 1], az = positions[i1 + 2] - positions[i0 + 2]
    const bx = positions[i2] - positions[i0], by = positions[i2 + 1] - positions[i0 + 1], bz = positions[i2 + 2] - positions[i0 + 2]
    const cx = positions[i2] - positions[i1], cy = positions[i2 + 1] - positions[i1 + 1], cz = positions[i2 + 2] - positions[i1 + 2]
    const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
    if (len > 0) { tn[t * 3] = nx / len; tn[t * 3 + 1] = ny / len; tn[t * 3 + 2] = nz / len }
    // |a x b| is the same at every corner, so each angle is atan2(|cross|, dot of its two edges)
    const a0 = Math.atan2(len, ax * bx + ay * by + az * bz)
    const a1 = Math.atan2(len, -(ax * cx + ay * cy + az * cz))
    ang[t * 3] = a0
    ang[t * 3 + 1] = a1
    ang[t * 3 + 2] = Math.max(0, Math.PI - a0 - a1)
  }

  // 3. the other half of every edge: half-edges bucketed by their lower point, matched by the higher
  // one (mate -1: a border, -2: more than two triangles on the edge, -3: a degenerate edge)
  const mate = new Int32Array(C).fill(-1)
  const bucket = new Int32Array(C)
  {
    const start = new Int32Array(V + 1)
    for (let he = 0; he < C; he++) {
      const a = weld[indices[he]], b = weld[indices[next(he)]]
      if (a === b) mate[he] = -3
      else start[Math.min(a, b) + 1]++
    }
    for (let v = 0; v < V; v++) start[v + 1] += start[v]
    const fill = start.slice(0, V)
    for (let he = 0; he < C; he++) {
      if (mate[he] === -3) continue
      const a = weld[indices[he]], b = weld[indices[next(he)]]
      bucket[fill[Math.min(a, b)]++] = he
    }
    let most = 0
    for (let v = 0; v < V; v++) most = Math.max(most, start[v + 1] - start[v])
    const keys = new Float64Array(most)
    const higher = (he: number) => Math.max(weld[indices[he]], weld[indices[next(he)]])
    for (let v = 0; v < V; v++) {
      const s = start[v], n = start[v + 1] - s
      if (n < 2) continue
      for (let k = 0; k < n; k++) keys[k] = higher(bucket[s + k]) * C + bucket[s + k]
      if (n <= 32) {
        for (let i = 1; i < n; i++) {  // insertion sort: buckets are a handful of half-edges
          const key = keys[i]
          let j = i - 1
          while (j >= 0 && keys[j] > key) { keys[j + 1] = keys[j]; j-- }
          keys[j + 1] = key
        }
      } else keys.subarray(0, n).sort()
      for (let i = 0; i < n;) {
        const hi = Math.floor(keys[i] / C)
        let j = i + 1
        while (j < n && Math.floor(keys[j] / C) === hi) j++
        if (j - i === 2) {
          const h1 = keys[i] - hi * C, h2 = keys[i + 1] - hi * C
          mate[h1] = h2
          mate[h2] = h1
        } else if (j - i > 2) {
          for (let k = i; k < j; k++) mate[keys[k] - hi * C] = -2
        }
        i = j
      }
    }
  }

  // 4. dense face numbers (body and CAD face); without CAD faces each list item is one "face"
  const byFaces = !!faces && faces.length === T
  const fk = new Uint32Array(T)
  let faceIds: Uint32Array, faceBodies: Uint32Array
  if (byFaces) {
    const dense = new Map<number, number>()
    const ids: number[] = [], bods: number[] = []
    let lastKey = -1, last = 0
    for (let t = 0; t < T; t++) {
      const body = bodies ? bodies[t] : 0
      const key = body * 16777216 + faces![t]
      if (key !== lastKey) {
        let f = dense.get(key)
        if (f === undefined) { f = ids.length; dense.set(key, f); ids.push(faces![t]); bods.push(body) }
        lastKey = key
        last = f
      }
      fk[t] = last
    }
    faceIds = Uint32Array.from(ids)
    faceBodies = Uint32Array.from(bods)
  } else {
    if (triPart) fk.set(triPart)
    faceIds = new Uint32Array(nParts).map((_, p) => p)
    faceBodies = faceIds.slice()
  }
  const F = faceIds.length

  // 5. what each edge is: smooth within a face, a border between two faces (tangent when most of
  // the triangle pairs along it bend under 30°, as the engine's tangent groups), or sharp
  const cosAt = (t1: number, t2: number) => {
    const ax = tn[t1 * 3], ay = tn[t1 * 3 + 1], az = tn[t1 * 3 + 2], bx = tn[t2 * 3], by = tn[t2 * 3 + 1], bz = tn[t2 * 3 + 2]
    if ((ax === 0 && ay === 0 && az === 0) || (bx === 0 && by === 0 && bz === 0)) return NaN
    return ax * bx + ay * by + az * bz
  }
  const pairKey = (f1: number, f2: number) => (f1 < f2 ? f1 * F + f2 : f2 * F + f1)
  // flat faces (every triangle's normal within ~0.5° of the face's first): two of them are only tangent
  // when they lie in one plane, so a gentle crease (a 16-sided prism's 22.5°) stays a sharp edge
  const flat = new Uint8Array(F)
  if (byFaces) {
    const first = new Float32Array(F * 3)
    const seen = new Uint8Array(F)
    flat.fill(1)
    for (let t = 0; t < T; t++) {
      const f = fk[t], x = tn[t * 3], y = tn[t * 3 + 1], z = tn[t * 3 + 2]
      if (x === 0 && y === 0 && z === 0) continue
      if (!seen[f]) { seen[f] = 1; first[f * 3] = x; first[f * 3 + 1] = y; first[f * 3 + 2] = z }
      else if (x * first[f * 3] + y * first[f * 3 + 1] + z * first[f * 3 + 2] < COS_FLAT) flat[f] = 0
    }
  }
  const votes = new Map<number, number>()
  if (byFaces) {
    for (let he = 0; he < C; he++) {
      const m = mate[he]
      if (m <= he) continue
      const t1 = (he / 3) | 0, t2 = (m / 3) | 0
      if (fk[t1] === fk[t2]) continue
      const cos = cosAt(t1, t2)
      if (Number.isNaN(cos)) continue
      const key = pairKey(fk[t1], fk[t2])
      const smooth = flat[fk[t1]] && flat[fk[t2]] ? cos > COS_COPLANAR : cos > COS_TANGENT
      votes.set(key, (votes.get(key) ?? 0) + (smooth ? 1 : -1))
    }
  }
  const kind = new Uint8Array(C)
  let nSharp = 0, nTangent = 0
  for (let he = 0; he < C; he++) {
    const m = mate[he]
    if (m === -3 || (m >= 0 && m < he)) continue
    let k: number
    if (m < 0) k = SHARP
    else {
      const t1 = (he / 3) | 0, t2 = (m / 3) | 0
      if (byFaces) k = fk[t1] === fk[t2] ? SMOOTH : (votes.get(pairKey(fk[t1], fk[t2])) ?? 0) > 0 ? TANGENT : SHARP
      else {
        const cos = cosAt(t1, t2)
        k = Number.isNaN(cos) || cos > COS_CREASE ? SMOOTH : SHARP
      }
    }
    kind[he] = k
    if (k === SHARP) nSharp++
    else if (k === TANGENT) nTangent++
  }

  // the corners at either end of an edge he (with its mate m), pairwise: the corners of m's triangle
  // at he's start and at he's end
  const across = (he: number, m: number): [number, number] => {
    const same = weld[indices[m]] === weld[indices[he]]  // m runs the same way (a flipped neighbour)
    return same ? [m, next(m)] : [next(m), m]
  }

  // 6. one vertex per group of corners joined by smooth edges (union-find over corners)
  const parent = new Int32Array(C)
  for (let c = 0; c < C; c++) parent[c] = c
  const find = (arr: Int32Array, i: number) => {
    while (arr[i] !== i) { arr[i] = arr[arr[i]]; i = arr[i] }
    return i
  }
  const union = (arr: Int32Array, a: number, b: number) => {
    const ra = find(arr, a), rb = find(arr, b)
    if (ra !== rb) arr[Math.max(ra, rb)] = Math.min(ra, rb)
  }
  for (let he = 0; he < C; he++) {
    if (kind[he] !== SMOOTH) continue
    const [s2, e2] = across(he, mate[he])
    union(parent, he, s2)
    union(parent, next(he), e2)
  }
  const index = new Uint32Array(C)
  const rootOut = bucket  // reused
  rootOut.fill(-1)
  let Vo = 0
  for (let c = 0; c < C; c++) {
    const r = find(parent, c)
    if (rootOut[r] < 0) rootOut[r] = Vo++
    index[c] = rootOut[r]
  }
  const outPos = new Float32Array(Vo * 3)
  const vface = new Uint32Array(Vo)
  const acc = new Float32Array(Vo * 3)
  for (let c = 0; c < C; c++) {
    const o = index[c], v = indices[c], t = (c / 3) | 0, w = ang[c]
    outPos[o * 3] = positions[v * 3]
    outPos[o * 3 + 1] = positions[v * 3 + 1]
    outPos[o * 3 + 2] = positions[v * 3 + 2]
    vface[o] = fk[t]
    acc[o * 3] += tn[t * 3] * w
    acc[o * 3 + 1] += tn[t * 3 + 1] * w
    acc[o * 3 + 2] += tn[t * 3 + 2] * w
  }
  // across tangent borders the normals are shared (each side keeps its own vertex, for face tints)
  const normals = new Float32Array(Vo * 3)
  let sums = acc
  let rootOf: ((o: number) => number) | null = null
  if (nTangent) {
    const group = new Int32Array(Vo)
    for (let o = 0; o < Vo; o++) group[o] = o
    for (let he = 0; he < C; he++) {
      if (kind[he] !== TANGENT) continue
      const [s2, e2] = across(he, mate[he])
      union(group, index[he], index[s2])
      union(group, index[next(he)], index[e2])
    }
    sums = new Float32Array(Vo * 3)
    for (let o = 0; o < Vo; o++) {
      const r = find(group, o)
      sums[r * 3] += acc[o * 3]; sums[r * 3 + 1] += acc[o * 3 + 1]; sums[r * 3 + 2] += acc[o * 3 + 2]
    }
    rootOf = (o) => find(group, o)
  }
  for (let o = 0; o < Vo; o++) {
    const r = rootOf ? rootOf(o) : o
    const x = sums[r * 3], y = sums[r * 3 + 1], z = sums[r * 3 + 2]
    const len = Math.sqrt(x * x + y * y + z * z)
    if (len > 0) { normals[o * 3] = x / len; normals[o * 3 + 1] = y / len; normals[o * 3 + 2] = z / len } else normals[o * 3 + 2] = 1
  }

  // 7. the lines, in triangle order (so each list item's are together), each end with the normals of
  // the two triangles beside it (the drawing hides lines whose both sides face away: seen through a
  // section cut, they would lie on the cut face)
  const edgeSet = (n: number): EdgeSet => ({
    positions: new Float32Array(n * 6), a: new Float32Array(n * 6), b: new Float32Array(n * 6), parts: new Uint32Array(nParts),
  })
  const sharp = edgeSet(nSharp), tangent = edgeSet(nTangent)
  let ks = 0, kt = 0
  for (let he = 0; he < C; he++) {
    const k = kind[he]
    if (k !== SHARP && k !== TANGENT) continue
    const set = k === SHARP ? sharp : tangent
    const at = k === SHARP ? ks++ : kt++
    const t1 = (he / 3) | 0, m = mate[he]
    set.parts[triPart ? triPart[t1] : 0]++
    const ends = [indices[he], indices[next(he)]]
    for (let e = 0; e < 2; e++) {
      const o = at * 6 + e * 3
      for (let x = 0; x < 3; x++) {
        set.positions[o + x] = positions[ends[e] * 3 + x]
        set.a[o + x] = tn[t1 * 3 + x]
        // a border has one side: its other normal points the other way, so it is never hidden
        set.b[o + x] = m >= 0 ? tn[((m / 3) | 0) * 3 + x] : -tn[t1 * 3 + x]
      }
    }
  }

  return { id, positions: outPos, normals, index, vface, faceIds, faceBodies, sharp, tangent, byFaces, ms: 0 }
}
