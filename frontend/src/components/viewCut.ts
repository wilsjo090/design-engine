// Shared by the 3D view's drawings (Viewport.tsx, CadBody.tsx): cutting a material at a box (details
// over views), and hiding list items whose pieces are drawn instead.
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'

/** inside of a part cut by the section plane */
export const CAP_COLOR = '#c8643f'

/** A box a view is cut at, in world coordinates: 'outside' keeps what is outside it (a view whose
 *  part on screen has a finer detail), 'inside' what is inside (that detail). Cut per pixel, so the
 *  two meet exactly on the box's faces. */
export interface CutBox { lo: [number, number, number]; hi: [number, number, number]; keep: 'outside' | 'inside' }

const NO_BOX = { lo: new THREE.Vector3(1e9, 1e9, 1e9), hi: new THREE.Vector3(-1e9, -1e9, -1e9) }

export interface BoxCutUniforms { uBoxLo: { value: THREE.Vector3 }; uBoxHi: { value: THREE.Vector3 } }

/** The box's uniforms, shared by every shader patched with them, so a new box needs no new shader. */
export function useBoxCutUniforms(box: CutBox | null | undefined): BoxCutUniforms {
  const uniforms = useMemo(() => ({ uBoxLo: { value: NO_BOX.lo.clone() }, uBoxHi: { value: NO_BOX.hi.clone() } }), [])
  useEffect(() => {
    uniforms.uBoxLo.value.copy(box ? new THREE.Vector3(...box.lo) : NO_BOX.lo)
    uniforms.uBoxHi.value.copy(box ? new THREE.Vector3(...box.hi) : NO_BOX.hi)
  }, [box, uniforms])
  return uniforms
}

/** Patches a built-in material's shader (any with `begin_vertex` and `clipping_planes_fragment`: the
 *  standard, basic and line materials) to discard what the box does not keep. */
export function patchBoxCut(shader: THREE.WebGLProgramParametersWithUniforms, uniforms: BoxCutUniforms, keep: CutBox['keep']) {
  shader.uniforms.uBoxLo = uniforms.uBoxLo
  shader.uniforms.uBoxHi = uniforms.uBoxHi
  shader.vertexShader = 'varying vec3 vBoxWorld;\n' + shader.vertexShader.replace('#include <begin_vertex>',
    '#include <begin_vertex>\nvBoxWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;')
  shader.fragmentShader = 'uniform vec3 uBoxLo;\nuniform vec3 uBoxHi;\nvarying vec3 vBoxWorld;\n' +
    shader.fragmentShader.replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\n' +
      `if (${keep === 'inside' ? '!' : ''}(all(greaterThanEqual(vBoxWorld, uBoxLo)) && all(lessThanEqual(vBoxWorld, uBoxHi)))) discard;`)
}

/** Material props that cut it at a box (see CutBox). */
export function useBoxCut(box: CutBox | null | undefined) {
  const keep = box?.keep ?? 'outside'
  const uniforms = useBoxCutUniforms(box)
  return useMemo(() => ({
    onBeforeCompile: (shader: THREE.WebGLProgramParametersWithUniforms) => patchBoxCut(shader, uniforms, keep),
    customProgramCacheKey: () => `box-${keep}`,
  }), [keep, uniforms])
}

/** Collapses the elements of list items `hide` of a geometry's index (`per` index entries per triangle
 *  or line, `counts` of them per item, `source` the whole index) to a point, and gives back those of
 *  items no longer hidden. Only the GPU's copy of those ranges is updated; the geometry gets its own
 *  index array on the first change, so `source` (picking, menus, face data) stays whole. */
export function hideRanges(geometry: THREE.BufferGeometry, source: Uint32Array, counts: ArrayLike<number> | null | undefined,
  per: number, hide: number[] | undefined) {
  const hidden: Set<number> = geometry.userData.hidden ?? new Set<number>()
  const want = new Set(counts ? hide ?? [] : [])
  const index = geometry.index
  if (!counts || !index || (!want.size && !hidden.size)) return
  if (index.array === source) index.array = source.slice()
  const array = index.array as Uint32Array
  const starts = [0]
  for (let p = 0; p < counts.length; p++) starts.push(starts[starts.length - 1] + counts[p] * per)
  let changed = false
  for (let p = 0; p < counts.length; p++) {
    if (want.has(p) === hidden.has(p)) continue
    const from = starts[p], to = starts[p + 1]
    if (to === from) continue
    if (want.has(p)) array.fill(0, from, to)
    else array.set(source.subarray(from, to), from)
    index.addUpdateRange(from, to - from)
    changed = true
  }
  geometry.userData.hidden = want
  if (changed) index.needsUpdate = true
}
