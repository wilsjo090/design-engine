"""Bar profiles: the cross-section of a frame's bars and its section properties.

A profile is a kind plus proportions; a bar's *size* is the profile's width in mm (a round bar's
diameter, a square's side, a rectangle's or an I's width, an angle's leg). Every property is
computed from the shape, so any size works without a table. Units: mm, mm^2, mm^4, mm^3.

Local axes of a bar: x along it, y across its width, z across its height. Iy = integral of z^2
(bending that curves the bar in its x-z plane), Iz = integral of y^2. Shapes other than round are
built from rectangles (holes as negative ones), so area, centroid, second moments and the extreme
fibres come out exactly; an angle's properties are about its principal axes (at 45 degrees).
"""
import math
from dataclasses import dataclass

import numpy as np

KINDS = ("round", "round tube", "square", "square tube", "rectangle", "rect tube", "I", "angle")
HOLLOW = ("round tube", "square tube", "rect tube")


@dataclass(frozen=True)
class Section:
    """Section properties of one size of a profile."""
    size: float
    A: float          # area
    Iy: float         # second moments about the principal axes through the centroid
    Iz: float
    J: float          # torsion constant (St Venant, or Bredt for closed tubes)
    Wy: float         # section moduli: bending stress = M / W at the extreme fibre
    Wz: float
    Wt: float         # torsion: shear stress = T / Wt (conservative for solid rectangles)
    width: float      # outer width (y) and height (z) as drawn, for drawing and fitting
    height: float
    centroid: tuple = (0.0, 0.0)  # in the drawing's (y, z) from its lower left corner (round: centre)
    turn: float = 0.0  # radians the drawing is turned so its strong axis lies along the bar's z

    @property
    def I_min(self):
        return min(self.Iy, self.Iz)

    @property
    def r_min(self):
        """Smallest radius of gyration (buckling)."""
        return math.sqrt(self.I_min / self.A)


@dataclass(frozen=True)
class Profile:
    kind: str = "round"
    ratio: float = 1.0      # rectangle, rect tube, I: height / width
    wall: float = 0.15      # tubes and angle: wall thickness as a share of the width ...
    min_wall: float = 0.8   # ... but at least this, mm (SLS prints walls down to ~0.8 mm)
    flange: float = 0.12    # I: flange thickness as a share of the height
    web: float = 0.12       # I: web thickness as a share of the width

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"profile must be one of: {', '.join(KINDS)}")
        if self.ratio <= 0:
            raise ValueError("the height / width ratio must be greater than 0")
        if not 0 < self.wall < 0.5 or not 0 < self.flange < 0.5 or not 0 < self.web < 1:
            raise ValueError("wall and flange shares must be between 0 and 0.5, web between 0 and 1")

    @property
    def hollow(self):
        return self.kind in HOLLOW

    @property
    def round(self):
        return self.kind in ("round", "round tube")

    def thickness(self, size):
        """Wall thickness of a tube or angle of this size, mm."""
        return min(max(self.wall * size, self.min_wall), size / 2)

    def section(self, size):
        size = float(size)
        if size <= 0:
            raise ValueError("a bar's size must be greater than 0")
        if self.round:
            d = size
            di = d - 2 * self.thickness(d) if self.kind == "round tube" else 0.0
            A = math.pi * (d ** 2 - di ** 2) / 4
            I = math.pi * (d ** 4 - di ** 4) / 64
            return Section(size, A, I, I, 2 * I, I / (d / 2), I / (d / 2), 2 * I / (d / 2), d, d, (d / 2, d / 2))
        rects, J, Wt = self._rectangles(size)
        return _from_rectangles(size, rects, J, Wt)

    def distance(self, size, y, z, xp=np):
        """Signed distance in the section's plane (mm, negative inside) at local (y, z): about the
        centroid, with the strong axis along z, as the frame check uses the section."""
        if self.round:
            rho = xp.sqrt(y * y + z * z)
            if self.kind == "round":
                return rho - size / 2
            t = self.thickness(size)
            return xp.abs(rho - (size - t) / 2) - t / 2
        sec = self.section(size)
        c, s = np.cos(sec.turn), np.sin(sec.turn)
        qy = sec.centroid[0] + y * c - z * s        # into the drawing's coordinates
        qz = sec.centroid[1] + y * s + z * c
        rects, _, _ = self._rectangles(size)
        solid = None
        for y0, y1, z0, z1, sign in rects:
            d = _box2(qy, qz, y0, y1, z0, z1, xp)
            if sign > 0:
                solid = d if solid is None else xp.minimum(solid, d)
            else:
                solid = xp.maximum(solid, -d)
        return solid

    def _rectangles(self, b):
        """(rectangles (y0, y1, z0, z1, sign), torsion constant, torsion modulus) of width b."""
        k = self.kind
        h = b * self.ratio if k in ("rectangle", "rect tube", "I") else b
        if k in ("square", "rectangle"):
            a, c = max(b, h), min(b, h)
            J = a * c ** 3 * (1 / 3 - 0.21 * (c / a) * (1 - c ** 4 / (12 * a ** 4)))
            return [(0, b, 0, h, 1)], J, J / c
        if k in ("square tube", "rect tube"):
            t = min(self.thickness(b), h / 2)
            Am = (b - t) * (h - t)               # Bredt: area inside the wall's middle line
            J = 4 * Am ** 2 * t / (2 * (b - t + h - t))
            rects = [(0, b, 0, h, 1)]
            if b - 2 * t > 0 and h - 2 * t > 0:
                rects.append((t, b - t, t, h - t, -1))
            return rects, J, 2 * Am * t
        if k == "I":   # flanges and web at least the thinnest wall too (0.3 mm at 2.5 mm broke the mesh)
            tf = min(max(self.flange * h, self.min_wall), h / 2)
            tw = min(max(self.web * b, self.min_wall), b)
            rects = [(0, b, 0, tf, 1), (0, b, h - tf, h, 1), ((b - tw) / 2, (b + tw) / 2, tf, h - tf, 1)]
            J = (2 * b * tf ** 3 + (h - 2 * tf) * tw ** 3) / 3
            return rects, J, J / max(tf, tw)
        t = self.thickness(b)  # angle: equal legs b, thickness t
        rects = [(0, b, 0, t, 1), (0, t, t, b, 1)]
        J = (b + (b - t)) * t ** 3 / 3
        return rects, J, J / t


def _box2(y, z, y0, y1, z0, z1, xp=np):
    """Signed distance to the rectangle [y0, y1] x [z0, z1]."""
    dy = xp.abs(y - (y0 + y1) / 2) - (y1 - y0) / 2
    dz = xp.abs(z - (z0 + z1) / 2) - (z1 - z0) / 2
    return xp.sqrt(xp.maximum(dy, 0) ** 2 + xp.maximum(dz, 0) ** 2) + xp.minimum(xp.maximum(dy, dz), 0)


def _from_rectangles(size, rects, J, Wt):
    r = np.asarray(rects, dtype=np.float64)
    y0, y1, z0, z1, s = r.T
    a = s * (y1 - y0) * (z1 - z0)
    A = a.sum()
    cy = (a * (y0 + y1) / 2).sum() / A
    cz = (a * (z0 + z1) / 2).sum() / A
    # second moments about the centroid (each rectangle: its own plus the parallel-axis term)
    Iyy = (s * (y1 - y0) * ((z1 - cz) ** 3 - (z0 - cz) ** 3) / 3).sum()   # integral of z^2
    Izz = (s * (z1 - z0) * ((y1 - cy) ** 3 - (y0 - cy) ** 3) / 3).sum()   # integral of y^2
    Iyz = (s * ((y1 - cy) ** 2 - (y0 - cy) ** 2) * ((z1 - cz) ** 2 - (z0 - cz) ** 2) / 4).sum()
    # principal directions u_k: w_k = integral of (p . u_k)^2, the second moment for bending that
    # strains the section along u_k (fibres measured along u_k too). The strong one is the bar's
    # local z (Iy = integral of z^2), the weak one its local y.
    w, v = np.linalg.eigh(np.array([[Izz, Iyz], [Iyz, Iyy]]))
    corners = np.array([(yy - cy, zz - cz) for y0_, y1_, z0_, z1_, sg in rects if sg > 0
                        for yy in (y0_, y1_) for zz in (z0_, z1_)])
    along = corners @ v                    # column k: each corner's distance along u_k
    I_weak, I_strong = float(w[0]), float(w[1])
    c_weak = float(np.abs(along[:, 0]).max())
    c_strong = float(np.abs(along[:, 1]).max())
    width = float(max(r[r[:, 4] > 0, 1]) - min(r[r[:, 4] > 0, 0]))
    height = float(max(r[r[:, 4] > 0, 3]) - min(r[r[:, 4] > 0, 2]))
    strong = v[:, 1]  # (y, z) of the strong direction: the drawing is turned by this to the bar's z
    turn = math.atan2(-strong[0], strong[1]) if abs(Iyz) > 1e-12 * max(abs(Izz), abs(Iyy)) else (
        0.0 if Iyy >= Izz else math.pi / 2)
    return Section(size, float(A), I_strong, I_weak, float(J), I_strong / c_strong, I_weak / c_weak, float(Wt),
                   width, height, (float(cy), float(cz)), float(turn))
