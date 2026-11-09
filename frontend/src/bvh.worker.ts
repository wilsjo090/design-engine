// A shown body's search tree for raycasts (three-mesh-bvh), built off the main thread (bvhBuild.ts): 3.7 s
// for 12M triangles froze the 3D view on its first right-click. Indirect, as the main thread builds it
// (Viewport's ensureBoundsTree): the tree sorts its own list of triangle numbers, never the index, so the
// view's triangle order (CAD face numbers, list items) stays as it is. The positions and index are copies;
// only the tree goes back (its node buffers and that list).
import { BufferAttribute, BufferGeometry } from 'three'
import { MeshBVH, type MeshBVHOptions } from 'three-mesh-bvh'
import type { BvhReply, BvhRequest } from './bvhBuild'

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<BvhRequest>) => void) | null
  postMessage(message: BvhReply, transfer: Transferable[]): void
}

ctx.onmessage = (e) => {
  const { id, positions, index } = e.data
  try {
    const t0 = performance.now()
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new BufferAttribute(positions, 3))
    geometry.setIndex(new BufferAttribute(index, 1))
    // (`indirect` exists in three-mesh-bvh 0.7.8 but is missing from its type definitions)
    const bvh = new MeshBVH(geometry, { indirect: true } as MeshBVHOptions)
    const tree = MeshBVH.serialize(bvh, { cloneBuffers: false }) as unknown as
      { roots: ArrayBuffer[]; indirectBuffer: Uint32Array | Uint16Array | null }
    if (!tree.indirectBuffer) throw new Error('the tree was not built indirect')
    ctx.postMessage({ id, roots: tree.roots, indirect: tree.indirectBuffer, ms: performance.now() - t0 },
      [...tree.roots, tree.indirectBuffer.buffer as ArrayBuffer])
  } catch (err) {
    ctx.postMessage({ id, error: String(err) }, [])
  }
}
