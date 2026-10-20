"""Materials: stiffness, Poisson's ratio, density and strength. Units: MPa, g/cm^3."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    E: float         # Young's modulus (stiffness), MPa
    nu: float        # Poisson's ratio
    density: float   # g/cm^3
    strength: float  # yield (metals) or tensile strength (plastics), MPa

    @property
    def key(self):
        return (self.E, self.nu, self.density, self.strength)

    def describe(self):
        return (f"{self.name} (E {self.E:g} MPa, Poisson {self.nu:g}, {self.density:g} g/cm3, "
                f"strength {self.strength:g} MPa)")


# Typical datasheet values for printed (plastics) or wrought (metals) material. Printed parts are
# weaker between layers than along them.
PRESETS = {m.name: m for m in (
    Material("PLA", 3500.0, 0.36, 1.24, 50.0),
    Material("PETG", 2000.0, 0.38, 1.27, 45.0),
    Material("ABS", 2200.0, 0.35, 1.04, 40.0),
    Material("Nylon PA12", 1700.0, 0.39, 1.01, 48.0),
    Material("Aluminium 6061-T6", 69000.0, 0.33, 2.70, 276.0),
    Material("Steel S355", 210000.0, 0.30, 7.85, 355.0),
)}


def custom(E, nu, density, strength):
    if E <= 0:
        raise ValueError("stiffness E must be greater than 0")
    if not -0.99 < nu < 0.499:
        raise ValueError("Poisson's ratio must be below 0.5 (rubber-like) and above -1")
    if density <= 0 or strength <= 0:
        raise ValueError("density and strength must be greater than 0")
    return Material("custom", float(E), float(nu), float(density), float(strength))
