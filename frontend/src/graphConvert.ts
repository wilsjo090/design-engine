// Recipe JSON (engine format) <-> React Flow nodes and edges.
import type { EngineEdge, EngineNode, GroupDef, NodeSpec, Recipe } from './types'
import { GROUP_IN, GROUP_OUT, GROUP_PREFIX, SETTING_TYPE, TYPE_COLORS, isLink } from './types'

export function edgeId(source: string, sourceHandle: string, target: string, targetHandle: string) {
  return `${source}.${sourceHandle}->${target}.${targetHandle}`
}

export function makeEdge(
  source: string, sourceHandle: string, target: string, targetHandle: string, type?: string,
): EngineEdge {
  return {
    id: edgeId(source, sourceHandle, target, targetHandle),
    source, sourceHandle, target, targetHandle,
    style: { stroke: TYPE_COLORS[type ?? ''] ?? '#888', strokeWidth: 2 },
  }
}

/** Simple layered layout for recipes without saved positions: columns by dependency depth. */
function autoLayout(recipe: Recipe): Record<string, [number, number]> {
  const ids = Object.keys(recipe.nodes)
  const depth: Record<string, number> = {}
  const visit = (id: string, stack: Set<string>): number => {
    if (depth[id] !== undefined) return depth[id]
    if (stack.has(id)) return 0
    stack.add(id)
    let d = 0
    for (const v of Object.values(recipe.nodes[id]?.inputs ?? {})) {
      if (isLink(v)) {
        const src = v.link.split('.')[0]
        if (recipe.nodes[src]) d = Math.max(d, visit(src, stack) + 1)
      }
    }
    stack.delete(id)
    depth[id] = d
    return d
  }
  ids.forEach((id) => visit(id, new Set()))
  // Long chains wrap into bands of COLS columns so the graph stays roughly screen-shaped.
  const COLS = 5, DX = 320, DY = 200
  const rows: Record<number, number> = {}
  for (const id of ids) rows[depth[id]] = (rows[depth[id]] ?? 0) + 1
  const bandHeight: number[] = []
  for (const [d, n] of Object.entries(rows)) {
    const band = Math.floor(Number(d) / COLS)
    bandHeight[band] = Math.max(bandHeight[band] ?? 0, n)
  }
  const bandTop = (band: number) => bandHeight.slice(0, band).reduce((a, n) => a + (n ?? 0) * DY + 80, 0)
  const used: Record<number, number> = {}
  const pos: Record<string, [number, number]> = {}
  for (const id of ids) {
    const d = depth[id]
    const row = (used[d] = (used[d] ?? 0) + 1) - 1
    pos[id] = [(d % COLS) * DX, bandTop(Math.floor(d / COLS)) + row * DY]
  }
  return pos
}

export function recipeToFlow(recipe: Recipe, specs: Record<string, NodeSpec>) {
  const layout = autoLayout(recipe)
  const nodes: EngineNode[] = []
  const edges: EngineEdge[] = []
  for (const [id, spec] of Object.entries(recipe.nodes)) {
    const literals: Record<string, unknown> = {}
    for (const [port, value] of Object.entries(spec.inputs ?? {})) {
      if (isLink(value)) {
        const [src, ...rest] = value.link.split('.')
        const srcPort = rest.join('.')
        const srcType = specs[recipe.nodes[src]?.type]?.outputs.find((o) => o.name === srcPort)?.type
        edges.push(makeEdge(src, srcPort, id, port, srcType))
      } else {
        literals[port] = value
      }
    }
    const [x, y] = spec.pos ?? layout[id]
    nodes.push({ id, type: 'engine', position: { x, y }, data: { typeId: spec.type, inputs: literals } })
  }
  const visible = Object.entries(recipe.nodes).filter(([, s]) => s.visible).map(([id]) => id)
  return { nodes, edges, visible }
}

export function flowToRecipe(nodes: EngineNode[], edges: EngineEdge[], visible: string[]): Recipe {
  const out: Recipe = { version: 1, nodes: {} }
  for (const n of nodes) {
    const inputs: Record<string, unknown> = { ...n.data.inputs }
    for (const e of edges) {
      if (e.target === n.id && e.targetHandle) inputs[e.targetHandle] = { link: `${e.source}.${e.sourceHandle}` }
    }
    out.nodes[n.id] = {
      type: n.data.typeId,
      inputs,
      pos: [Math.round(n.position.x), Math.round(n.position.y)],
      ...(visible.includes(n.id) ? { visible: true } : {}),
    }
  }
  return out
}

/** A readable unique id for a new node, e.g. "offset_2". */
export function newNodeId(typeId: string, existing: Set<string>) {
  const base = typeId.split('.').pop()!.replace(/[^a-z0-9_]/gi, '_')
  for (let i = 1; ; i++) {
    const id = `${base}_${i}`
    if (!existing.has(id)) return id
  }
}

// --- groups ---------------------------------------------------------------------------------
/** Ids of the input and output nodes shown while editing a group. */
export const GROUP_IN_ID = '__group_in'
export const GROUP_OUT_ID = '__group_out'

/** The node spec of a group: its inputs, one setting per Number node inside, its outputs. */
export function groupSpec(name: string, def: GroupDef): NodeSpec {
  const settings = Object.entries(def.nodes)
    .filter(([, n]) => n.type === SETTING_TYPE)
    .map(([id, n]) => ({
      name: id, type: 'number', label: id, default: (n.inputs?.value as number | undefined) ?? 0,
      doc: `Setting: the value of the Number node "${id}" inside the group.`,
    }))
  const count = Object.keys(def.nodes).length
  return {
    type: GROUP_PREFIX + name,
    title: name,
    category: 'Groups',
    doc: def.doc || `Your group of ${count} node${count === 1 ? '' : 's'}. Edit opens it (changes apply to every copy); ` +
      'Ungroup puts its nodes back. Number nodes inside become its settings.',
    inputs: [
      ...def.inputs.map((i) => ({ name: i.name, type: i.type, label: i.name, optional: true,
        doc: 'Group input: what the nodes inside get from outside.' })),
      ...settings,
    ],
    outputs: def.outputs.map((o) => ({ name: o.name, type: o.type, label: o.name, doc: `Group output (inside: ${o.link}).` })),
  }
}

/** A group's nodes for editing, plus its input node (left) and output node (right). */
export function groupToFlow(def: GroupDef, specs: Record<string, NodeSpec>) {
  const { nodes, edges } = recipeToFlow({ version: 1, nodes: def.nodes }, specs)
  const xs = nodes.map((n) => n.position.x), ys = nodes.map((n) => n.position.y)
  const midY = ys.length ? (Math.min(...ys) + Math.max(...ys)) / 2 : 0
  const left = (xs.length ? Math.min(...xs) : 0) - 360, right = (xs.length ? Math.max(...xs) : 0) + 360
  const flowEdges = edges.map((e) => {
    if (e.source !== GROUP_IN) return e
    const type = def.inputs.find((i) => i.name === e.sourceHandle)?.type
    return makeEdge(GROUP_IN_ID, e.sourceHandle!, e.target, e.targetHandle!, type)
  })
  for (const o of def.outputs) {
    const [src, ...rest] = o.link.split('.')
    if (def.nodes[src]) flowEdges.push(makeEdge(src, rest.join('.'), GROUP_OUT_ID, o.name, o.type))
  }
  const io: EngineNode[] = [
    { id: GROUP_IN_ID, type: 'groupIn', position: { x: left, y: midY }, deletable: false, data: { typeId: GROUP_IN, inputs: {} } },
    { id: GROUP_OUT_ID, type: 'groupOut', position: { x: right, y: midY }, deletable: false, data: { typeId: GROUP_OUT, inputs: {} } },
  ]
  return { nodes: [...io, ...nodes], edges: flowEdges }
}

/** The group definition from the nodes being edited (inputs and outputs come from `prev`). */
export function flowToGroup(nodes: EngineNode[], edges: EngineEdge[], prev: GroupDef): GroupDef {
  const inner = nodes.filter((n) => n.type === 'engine')
  const ids = new Set(inner.map((n) => n.id))
  const recipe = flowToRecipe(inner, edges.filter((e) => e.target !== GROUP_OUT_ID), [])
  for (const spec of Object.values(recipe.nodes)) {
    for (const [port, value] of Object.entries(spec.inputs ?? {})) {
      if (isLink(value) && value.link.startsWith(`${GROUP_IN_ID}.`)) {
        spec.inputs![port] = { link: `${GROUP_IN}.${value.link.slice(GROUP_IN_ID.length + 1)}` }
      }
    }
  }
  const outputs = prev.outputs.flatMap((o) => {
    const e = edges.find((x) => x.target === GROUP_OUT_ID && x.targetHandle === o.name)
    if (e) return [{ ...o, link: `${e.source}.${e.sourceHandle}` }]
    return ids.has(o.link.split('.')[0]) ? [o] : []
  })
  return { ...prev, nodes: recipe.nodes, outputs }
}
