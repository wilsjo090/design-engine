import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { useStore } from '../store'

/** The box of the body hovered in a card's body list (store.bodyHint, PickLists.BodyPick), drawn through
 *  everything with its name, so the body is found in the 3D view without looking for it. */
export function BodyHint() {
  const hint = useStore((s) => s.bodyHint)
  const box = useMemo(() => {
    if (!hint) return null
    const lo = new THREE.Vector3(...hint.lo)
    const hi = new THREE.Vector3(...hint.hi)
    const size = hi.clone().sub(lo).max(new THREE.Vector3(0.01, 0.01, 0.01))
    const centre = lo.clone().add(hi).multiplyScalar(0.5)
    return { centre, size, top: new THREE.Vector3(centre.x, centre.y, hi.z) }
  }, [hint])
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), [])
  if (!hint || !box) return null
  return (
    <group position={box.centre}>
      <mesh scale={box.size} renderOrder={30} raycast={() => null}>
        <boxGeometry />
        <meshBasicMaterial color="#5cc8ff" transparent opacity={0.12} depthTest={false} depthWrite={false} />
      </mesh>
      <lineSegments geometry={edges} scale={box.size} renderOrder={31} raycast={() => null}>
        <lineBasicMaterial color="#5cc8ff" depthTest={false} transparent opacity={0.95} />
      </lineSegments>
      <Html position={[0, 0, box.size.z / 2]} center style={{ pointerEvents: 'none' }}>
        <div className="body-hint-label">{hint.label}</div>
      </Html>
    </group>
  )
}
