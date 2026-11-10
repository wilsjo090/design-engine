import * as THREE from 'three'

/** A Pipe node's shape as the 3D view draws it (engine/pipe.py makes the field: the same curve and sections). */
export interface PipeDraw {
  points: THREE.Vector3[]
  radius: number
  endRadius: number      // 0 = the same as radius
  teardrop: boolean
  up: THREE.Vector3      // unit
  roof: number           // degrees from horizontal
  smooth: boolean
  flat: boolean
}

/** The points of a Pipe's Points input ("x, y, z; x, y, z; ..."), or null when one is not three numbers. */
export function parsePoints(value: unknown): THREE.Vector3[] | null {
  const parts = String(value ?? '').split(/[;\n]+/).map((s) => s.trim()).filter(Boolean)
  const out: THREE.Vector3[] = []
  for (const part of parts) {
    const v = part.split(/[\s,]+/).filter(Boolean).map(Number)
    if (v.length !== 3 || !v.every(Number.isFinite)) return null
    out.push(new THREE.Vector3(...v))
  }
  return out.length ? out : null
}

const fmt = (x: number) => String(Number(x.toFixed(3)))
export const pointsText = (points: THREE.Vector3[]) => points.map((p) => `${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)}`).join('; ')

const SAME = 1e-6

function distinct(points: THREE.Vector3[]) {
  const out: THREE.Vector3[] = []
  for (const p of points) if (!out.length || p.distanceTo(out[out.length - 1]) > SAME) out.push(p)
  return out
}

/** One unit tangent per point, as engine/pipe.py `_tangents`: the bisector of the chords, the ends' making
 *  their spans parabolas. */
function tangents(P: THREE.Vector3[]) {
  const u = P.slice(1).map((p, i) => p.clone().sub(P[i]).normalize())
  const t = P.map(() => new THREE.Vector3())
  for (let i = 1; i < P.length - 1; i++) {
    const b = u[i - 1].clone().add(u[i])
    t[i] = b.length() > 1e-9 ? b.normalize() : u[i].clone()
  }
  const reflect = (tn: THREE.Vector3, ui: THREE.Vector3) => {
    const v = ui.clone().multiplyScalar(2 * tn.dot(ui)).sub(tn)
    return v.length() > 1e-9 ? v.normalize() : ui.clone()
  }
  t[0] = reflect(t[1], u[0])
  t[P.length - 1] = reflect(t[P.length - 2], u[u.length - 1])
  return t
}

interface Sample { at: THREE.Vector3; tangent: THREE.Vector3; length: number }

/** The path as runs of samples (one run for a smooth curve, one per straight piece otherwise). */
function runs(P: THREE.Vector3[], smooth: boolean, spacing: number): Sample[][] {
  if (!smooth || P.length < 3) {
    let length = 0
    return P.slice(1).map((b, i) => {
      const a = P[i]
      const tangent = b.clone().sub(a).normalize()
      const run = [{ at: a.clone(), tangent, length }, { at: b.clone(), tangent, length: length + a.distanceTo(b) }]
      length += a.distanceTo(b)
      return run
    })
  }
  const t = tangents(P)
  const run: Sample[] = []
  let length = 0
  for (let i = 0; i < P.length - 1; i++) {
    const chord = P[i].distanceTo(P[i + 1])
    const m0 = t[i].clone().multiplyScalar(chord)
    const m1 = t[i + 1].clone().multiplyScalar(chord)
    const n = Math.min(256, Math.max(8, Math.ceil((1.6 * chord) / spacing)))
    for (let j = i === 0 ? 0 : 1; j <= n; j++) {
      const s = j / n, s2 = s * s, s3 = s2 * s
      const at = P[i].clone().multiplyScalar(2 * s3 - 3 * s2 + 1).addScaledVector(m0, s3 - 2 * s2 + s)
        .addScaledVector(P[i + 1], -2 * s3 + 3 * s2).addScaledVector(m1, s3 - s2)
      const d = P[i].clone().multiplyScalar(6 * s2 - 6 * s).addScaledVector(m0, 3 * s2 - 4 * s + 1)
        .addScaledVector(P[i + 1], -6 * s2 + 6 * s).addScaledVector(m1, 3 * s2 - 2 * s)
      if (run.length) length += at.distanceTo(run[run.length - 1].at)
      run.push({ at, tangent: d.length() > 1e-12 ? d.normalize() : t[i].clone(), length })
    }
  }
  return [run]
}

const RING = 32

class Builder {
  positions: number[] = []
  index: number[] = []
  vertex(v: THREE.Vector3) { this.positions.push(v.x, v.y, v.z); return this.positions.length / 3 - 1 }
  quad(a: number, b: number, c: number, d: number) { this.index.push(a, b, c, a, c, d) }
  geometry() {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3))
    g.setIndex(this.index)
    g.computeVertexNormals()
    return g
  }
}

/** The section of a straight sweep at one sample: a circle, or a teardrop (the 3D section seen along the
 *  path) whose tip points along Up's part square to the path. RING + 1 vertices, the tip first. */
function ring(b: Builder, s: Sample, r: number, frame: { n: THREE.Vector3; b: THREE.Vector3 }, p: PipeDraw,
  last: { angle: number }) {
  const { n, b: bn } = frame
  let reach = r, half = 0
  if (p.teardrop) {
    const across = p.up.clone().addScaledVector(s.tangent, -p.up.dot(s.tangent))
    if (across.length() > 1e-3) last.angle = Math.atan2(across.dot(bn), across.dot(n))
    const tip = (across.length() * r) / Math.cos(THREE.MathUtils.degToRad(p.roof))
    if (tip > r * (1 + 1e-6)) { reach = tip; half = Math.acos(r / tip) }
  }
  const dir = (angle: number, len: number) =>
    s.at.clone().addScaledVector(n, Math.cos(angle) * len).addScaledVector(bn, Math.sin(angle) * len)
  const first = b.vertex(dir(last.angle, reach))
  for (let j = 0; j < RING; j++) b.vertex(dir(last.angle + half + ((2 * Math.PI - 2 * half) * j) / (RING - 1), r))
  return first
}

/** The section itself at a point (a ball, or the teardrop of revolution about Up). */
function section(b: Builder, at: THREE.Vector3, r: number, p: PipeDraw) {
  const axis = p.teardrop ? p.up : new THREE.Vector3(0, 0, 1)
  const e1 = Math.abs(axis.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)
  e1.addScaledVector(axis, -e1.dot(axis)).normalize()
  const e2 = axis.clone().cross(e1)
  const roof = THREE.MathUtils.degToRad(p.roof)
  const top = p.teardrop ? Math.PI / 2 - roof : Math.PI / 2
  const profile: [number, number][] = []
  for (let k = 0; k <= 16; k++) {
    const a = -Math.PI / 2 + ((top + Math.PI / 2) * k) / 16
    profile.push([r * Math.cos(a), r * Math.sin(a)])
  }
  if (p.teardrop) profile.push([0, r / Math.cos(roof)])
  const rows = profile.map(([rho, y]) => {
    const start = b.positions.length / 3
    for (let j = 0; j < RING; j++) {
      const psi = (2 * Math.PI * j) / RING
      b.vertex(at.clone().addScaledVector(e1, rho * Math.cos(psi)).addScaledVector(e2, rho * Math.sin(psi)).addScaledVector(axis, y))
    }
    return start
  })
  for (let k = 0; k + 1 < rows.length; k++) {
    for (let j = 0; j < RING; j++) {
      const j2 = (j + 1) % RING
      b.quad(rows[k] + j, rows[k] + j2, rows[k + 1] + j2, rows[k + 1] + j)
    }
  }
}

/** The pipe as triangles: the swept sections, the end sections (round ends), the joints of a straight
 *  path, flat caps. */
export function pipeGeometry(p: PipeDraw): THREE.BufferGeometry {
  const b = new Builder()
  const P = distinct(p.points)
  const end = p.endRadius > 0 ? p.endRadius : p.radius
  if (P.length === 1) {
    section(b, P[0], p.radius, p)
    return b.geometry()
  }
  const all = runs(P, p.smooth, Math.max(p.radius * 0.5, 0.2))
  const lastRun = all[all.length - 1]
  const total = lastRun[lastRun.length - 1].length || 1
  const radius = (length: number) => p.radius + ((end - p.radius) * length) / total
  for (const run of all) {
    // a frame carried along the run without turning about the path (no twist)
    let n = new THREE.Vector3().crossVectors(run[0].tangent, Math.abs(run[0].tangent.z) < 0.9
      ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0)).normalize()
    const last = { angle: 0 }
    let previous = -1
    run.forEach((s, i) => {
      if (i > 0) {
        const q = new THREE.Quaternion().setFromUnitVectors(run[i - 1].tangent, s.tangent)
        n = n.applyQuaternion(q)
        n.addScaledVector(s.tangent, -n.dot(s.tangent)).normalize()
      }
      const frame = { n, b: s.tangent.clone().cross(n) }
      const first = ring(b, s, radius(s.length), frame, p, last)
      if (previous >= 0) {
        for (let j = 0; j <= RING; j++) {
          const j2 = (j + 1) % (RING + 1)
          b.quad(previous + j, previous + j2, first + j2, first + j)
        }
      }
      if (p.flat && ((run === all[0] && i === 0) || (run === all[all.length - 1] && i === run.length - 1))) {
        const centre = b.vertex(s.at)
        for (let j = 0; j <= RING; j++) b.index.push(centre, first + j, first + ((j + 1) % (RING + 1)))
      }
      previous = first
    })
  }
  // the sections at the joints of a straight path, and at the ends when they are round
  all.forEach((run, k) => {
    if (k > 0) section(b, run[0].at, radius(run[0].length), p)
  })
  if (!p.flat) {
    section(b, P[0], p.radius, p)
    section(b, P[P.length - 1], end, p)
  }
  return b.geometry()
}
