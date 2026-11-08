// The wizards' panels share the room beside the 3D view: opening one closes the others.
const closers = new Map<string, () => void>()

/** A wizard's way to close its panel (stopping its picking). */
export function registerWizard(name: string, close: () => void) {
  closers.set(name, close)
}

export function closeOtherWizards(name: string) {
  for (const [other, close] of closers) if (other !== name) close()
}
