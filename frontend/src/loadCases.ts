import type { EngineEdge, EngineNode } from './types'

/** Load cases (Load Case nodes): which loads belong to which case, and the case's colour on the node
 *  cards, the arrows in the 3D view and its legend. Loads in no case (and supports) act in every case and
 *  keep their usual colours. */
export const LOAD_CASE = 'sim.load_case'
/** One colour per case in the order of their names (as the engine lists them: Case 2 before Case 10);
 *  none is the loads' red or the supports' green. */
export const CASE_COLORS = ['#ff9f1c', '#4d96ff', '#c77dff', '#f15bb5', '#2ec4b6', '#ffca3a', '#8ac926', '#ff7f6b']
const DEFAULT_NAME = 'Case 1'

export interface LoadCase {
  name: string
  color: string
  /** its Load Case nodes (two with the same name are one case) */
  nodes: string[]
  /** the Load nodes in the chains feeding them */
  loads: string[]
}

export interface Cases {
  list: LoadCase[]
  /** node id -> its case: the Load Case nodes and their loads */
  of: Record<string, LoadCase>
}

const NONE: Cases = { list: [], of: {} }

export const compareNames = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })

export function caseName(n: EngineNode) {
  return String(n.data.inputs.name ?? DEFAULT_NAME).trim() || DEFAULT_NAME
}

let last: { nodes: EngineNode[]; edges: EngineEdge[]; cases: Cases } | null = null

/** The load cases of a graph (worked out once per change of its nodes or edges). A case's loads are the
 *  Load nodes up its Loads input through their Add to, up to another Load Case. */
export function loadCases(nodes: EngineNode[], edges: EngineEdge[]): Cases {
  if (last && last.nodes === nodes && last.edges === edges) return last.cases
  const caseNodes = nodes.filter((n) => n.data.typeId === LOAD_CASE)
  let cases = NONE
  if (caseNodes.length) {
    const byId = new Map(nodes.map((n) => [n.id, n]))
    const into = (id: string, port: string) => edges.find((e) => e.target === id && e.targetHandle === port)?.source
    const names = [...new Set(caseNodes.map(caseName))].sort(compareNames)
    const list = names.map((name, i): LoadCase => ({
      name, color: CASE_COLORS[i % CASE_COLORS.length], loads: [],
      nodes: caseNodes.filter((n) => caseName(n) === name).map((n) => n.id),
    }))
    const of: Record<string, LoadCase> = {}
    for (const c of list) {
      for (const id of c.nodes) {
        of[id] = c
        const seen = new Set<string>()
        for (let cur = into(id, 'loads'); cur && !seen.has(cur); cur = into(cur, 'add_to')) {
          seen.add(cur)
          const type = byId.get(cur)?.data.typeId
          if (type !== 'sim.load' && type !== 'sim.fixed') break   // (another Load Case keeps its own)
          if (type === 'sim.load' && !of[cur]) {
            of[cur] = c
            c.loads.push(cur)
          }
        }
      }
    }
    cases = { list, of }
  }
  last = { nodes, edges, cases }
  return cases
}

/** `loadCases(...).list` as JSON: a string, so a selector using it changes only when the cases do, not
 *  on every move of a node (`parseCases` turns it back). */
export function casesKey(nodes: EngineNode[], edges: EngineEdge[]) {
  return JSON.stringify(loadCases(nodes, edges).list)
}

export function parseCases(key: string): Cases {
  const list = JSON.parse(key) as LoadCase[]
  return { list, of: Object.fromEntries(list.flatMap((c) => [...c.nodes, ...c.loads].map((id) => [id, c]))) }
}

/** A name for another case: the next number after the highest "<text> <n>" taken ("Case 3"), else
 *  "<name> 2", "<name> 3", ... */
export function nextCaseName(name: string, taken: Set<string>) {
  const m = /^(.*?)(\d+)\s*$/.exec(name)
  const stem = m ? m[1] : `${name.trimEnd()} `
  let n = m ? Number(m[2]) : 1
  for (const t of taken) {
    const k = t.startsWith(stem) ? Number(t.slice(stem.length)) : NaN
    if (Number.isInteger(k) && k > n) n = k
  }
  let candidate = `${stem}${n + 1}`
  while (taken.has(candidate)) candidate = `${stem}${++n + 1}`
  return candidate
}
