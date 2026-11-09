// Search trees for raycasting the 3D view's bodies (right-click menus, a click selecting a node, Measure),
// built in a Web Worker (bvh.worker.ts) as soon as a body is drawn: on the main thread the first right-click
// on a 12M-triangle body froze the view for 3.7 s. One build at a time, the newest body first, started when
// the browser is idle; a body replaced or hidden before its tree is done drops its build. A right-click before
// the tree is there builds it on the main thread as before (Viewport's ensureBoundsTree).
import * as THREE from 'three'
import { MeshBVH, type SerializedBVH } from 'three-mesh-bvh'

export interface BvhRequest { id: number; positions: Float32Array; index: Uint32Array }
export type BvhReply =
  | { id: number; roots: ArrayBuffer[]; indirect: Uint32Array | Uint16Array; ms: number; error?: undefined }
  | { id: number; error: string }

interface Job { id: number; geometry: THREE.BufferGeometry; source: Uint32Array }

/** waiting builds, the newest last (taken from the end) */
const queue: Job[] = []
const jobs = new WeakMap<THREE.BufferGeometry, Job>()
let current: Job | null = null
let worker: Worker | null = null
let nextId = 1
let waitingIdle = false
let idleTimer: ReturnType<typeof setTimeout> | null = null
/** workers that failed in a row (after a few, trees are only built on the main thread) */
let failures = 0
/** the worker is let go after this long without work (a big build leaves hundreds of MB in its heap) */
const IDLE_STOP = 5000

/** Build `geometry`'s search tree in the worker, from its positions and `source` (its whole index: list items
 *  hidden by hideRanges are collapsed in the geometry's own copy, and a tree built from that would not find
 *  them once shown again; raycasts read the geometry's index as it is, so hidden ones are never hit). */
export function scheduleBoundsTree(geometry: THREE.BufferGeometry, source: Uint32Array) {
  if (geometry.boundsTree || jobs.has(geometry) || failures >= 3 || !geometry.index) return
  if (source.length !== geometry.index.count) source = geometry.index.array as Uint32Array
  const job: Job = { id: nextId++, geometry, source }
  jobs.set(geometry, job)
  queue.push(job)
  pump()
}

/** `geometry` is going (its view replaced, its body hidden) or its tree is being built on the main thread:
 *  its build is dropped, and stopped if it is running. */
export function cancelBoundsTree(geometry: THREE.BufferGeometry) {
  const job = jobs.get(geometry)
  if (!job) return
  jobs.delete(geometry)
  const i = queue.indexOf(job)
  if (i >= 0) queue.splice(i, 1)
  if (current === job) {
    worker?.terminate()
    worker = null
    current = null
    pump()
  }
}

function pump() {
  if (current || !queue.length || waitingIdle) return
  waitingIdle = true
  const go = () => { waitingIdle = false; start() }
  if (typeof requestIdleCallback === 'function') requestIdleCallback(go, { timeout: 1000 })
  else setTimeout(go, 50)
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./bvh.worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (e: MessageEvent<BvhReply>) => finished(e.data)
    worker.onerror = (e) => {
      console.warn('Search tree worker failed:', e.message)
      failures++
      const job = current
      worker?.terminate()
      worker = null
      current = null
      if (job) jobs.delete(job.geometry)   // (a right-click builds it on the main thread)
      pump()
    }
  }
  return worker
}

function start() {
  if (current) return
  if (failures >= 3) {   // the worker cannot run here: right-clicks build the trees as before
    for (const j of queue) jobs.delete(j.geometry)
    queue.length = 0
    return
  }
  let job = queue.pop()
  while (job && (job.geometry.boundsTree || jobs.get(job.geometry) !== job)) {   // (built meanwhile)
    if (jobs.get(job.geometry) === job) jobs.delete(job.geometry)
    job = queue.pop()
  }
  if (!job) return
  if (idleTimer) clearTimeout(idleTimer)
  idleTimer = null
  current = job
  const positions = (job.geometry.getAttribute('position').array as Float32Array).slice()
  const index = job.source.slice()
  const request: BvhRequest = { id: job.id, positions, index }
  getWorker().postMessage(request, [positions.buffer, index.buffer])
}

function finished(reply: BvhReply) {
  const job = current
  if (!job || reply.id !== job.id) return
  current = null
  if (reply.error !== undefined) {
    console.warn(`Search tree for ${job.source.length / 3} triangles failed in the worker: ${reply.error}`)
    jobs.delete(job.geometry)
  } else if (jobs.get(job.geometry) === job) {
    jobs.delete(job.geometry)
    const geometry = job.geometry
    if (!geometry.boundsTree && geometry.index && geometry.index.count === job.source.length) {
      failures = 0
      const bvh = MeshBVH.deserialize({ roots: reply.roots, index: null, indirectBuffer: reply.indirect } as unknown as
        SerializedBVH, geometry, { setIndex: false })
      // (as the main thread's build does)
      if (!geometry.boundingBox) geometry.boundingBox = bvh.getBoundingBox(new THREE.Box3())
      geometry.boundsTree = bvh
      console.debug(`Search tree: ${job.source.length / 3} triangles in ${reply.ms.toFixed(0)} ms (worker)`)
    }
  }
  if (queue.length) pump()
  else {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => {
      idleTimer = null
      if (!current && !queue.length) {
        worker?.terminate()
        worker = null
      }
    }, IDLE_STOP)
  }
}
