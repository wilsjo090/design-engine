"""Pictures and text on CAD faces: a grayscale image, or text drawn in a font, laid onto picked
faces as a field 0..1 (dark = 1). Feed it to a pattern's Size scale (a halftone of the picture),
multiply it by a height for Offset Body (an embossed logo or lettering), or cut with it.

The picture is projected, not unwrapped: onto a flat face straight along its normal, onto a round
face (a cylinder or a cone) around its axis, so the picture keeps its proportions on the
surface. Within Band of the face only (a picture projected straight through a box would also land
on its far side). A pure function of position, so block meshing stays exact.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from .fields import Field, _digest
from .shared import Shared

MAPPINGS = ("auto", "flat", "around")
MAX_PIXELS = 1600     # the longer side is scaled down to this (finer than any voxel at picture sizes)
ROUND_SURFACES = {1, 2}   # GeomAbs: cylinder, cone
FONTS = Path(r"C:\Windows\Fonts")


class PictureError(ValueError):
    pass


def load_image(path):
    """Darkness 0..1 of a picture file (float32, rows from the top), scaled down to MAX_PIXELS."""
    from PIL import Image
    try:
        image = Image.open(path)
    except (OSError, ValueError) as err:
        raise PictureError(f"cannot read the picture {path}: {err}") from err
    if image.mode in ("RGBA", "LA", "P"):   # transparent parts count as white
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    image = image.convert("L")
    if max(image.size) > MAX_PIXELS:
        k = MAX_PIXELS / max(image.size)
        image = image.resize((max(1, round(image.width * k)), max(1, round(image.height * k))), Image.LANCZOS)
    return 1.0 - np.asarray(image, dtype=np.float32) / 255.0


def render_text(text, font="arial", pixels=240):
    """Darkness 0..1 of `text` drawn in a TrueType font (a file in C:\\Windows\\Fonts by name, e.g.
    arial, arialbd, times, consola, segoeui, or a full path), and how many pixels one letter height
    (the font size) is."""
    from PIL import Image, ImageDraw, ImageFont
    if not str(text).strip():
        raise PictureError("there is no text")
    candidates = [Path(font)] if any(c in str(font) for c in "/\\") else \
        [FONTS / f"{font}.ttf", FONTS / f"{font}.otf", FONTS / str(font)]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise PictureError(f"no font '{font}' (try arial, arialbd, times, consola, segoeui, or a full path)")
    face = ImageFont.truetype(str(path), pixels)
    left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=face)
    margin = pixels // 8
    image = Image.new("L", (right - left + 2 * margin, bottom - top + 2 * margin), 255)
    ImageDraw.Draw(image).text((margin - left, margin - top), text, font=face, fill=0)
    return 1.0 - np.asarray(image, dtype=np.float32) / 255.0, pixels


class Placement:
    """Where a picture sits: flat (centre, normal, right, up) or around an axis (point on the axis,
    axis, radius, the direction to the picture's centre)."""

    def __init__(self, mesh, chosen, mapping="auto", angle=0.0):
        ids = np.asarray(mesh.face_attributes.get("cad_face", []))
        if len(ids) != len(mesh.faces):
            raise PictureError("this body has no CAD faces: connect a body from Import STEP")
        tri = np.isin(ids, list(chosen))
        if not tri.any():
            raise PictureError("pick the face to put it on")
        V, F = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
        P = V[F[tri]]
        cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        area = 0.5 * np.linalg.norm(cross, axis=1)
        normals = cross / np.maximum(2 * area, 1e-300)[:, None]
        self.centre = (area[:, None] * P.mean(axis=1)).sum(axis=0) / area.sum()
        if mapping == "auto":
            kinds = np.asarray(mesh.metadata.get("cad_surfaces", np.zeros((0, 3))))
            round_face = any(f < len(kinds) and int(kinds[f][0]) in ROUND_SURFACES for f in chosen)
            mapping = "around" if round_face else "flat"
        self.mapping = mapping
        a = math.radians(angle)
        if mapping == "flat":
            n = (area[:, None] * normals).sum(axis=0)
            if np.linalg.norm(n) < 1e-9:
                raise PictureError("these faces point every way: pick one flat face (or use Mapping around)")
            n /= np.linalg.norm(n)
            up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            up -= (up @ n) * n
            up /= np.linalg.norm(up)
            right = np.cross(up, n)            # seen from outside, looking at the face
            self.normal = n
            self.right = right * math.cos(a) + up * math.sin(a)
            self.up = up * math.cos(a) - right * math.sin(a)
        else:
            # the axis: the direction the normals of a cylinder or cone are (nearly) square to
            w, vecs = np.linalg.eigh((normals * area[:, None]).T @ normals)
            axis = vecs[:, 0]
            if axis[np.argmax(np.abs(axis))] < 0:
                axis = -axis
            if axis @ np.array([0.0, 0.0, 1.0]) < 0:
                axis = -axis
            # a point on it and the radius: circle fit to the faces' corners, across the axis
            pts = P.reshape(-1, 3)
            e1 = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            q = np.column_stack([pts @ e1, pts @ e2])
            A = np.column_stack([2 * q, np.ones(len(q))])
            (cx, cy, k), *_ = np.linalg.lstsq(A, (q ** 2).sum(axis=1), rcond=None)
            self.axis = axis
            self.origin = cx * e1 + cy * e2
            self.radius = float(math.sqrt(max(k + cx * cx + cy * cy, 1e-12)))
            toward = self.centre - self.origin
            toward -= (toward @ axis) * axis
            if np.linalg.norm(toward) < 0.05 * self.radius:   # a face all the way round: face the front (-y)
                for front in ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0]):
                    toward = np.array(front) - (np.array(front) @ axis) * axis
                    if np.linalg.norm(toward) > 0.1:
                        break
            self.toward = toward / np.linalg.norm(toward)
            self.height = float((self.centre - self.origin) @ axis)
            self.turn = (math.cos(a), math.sin(a))   # the picture turned on the (unrolled) surface

    def coordinates(self, x, y, z):
        """(across, up, off the face) in mm for points (broadcastable arrays)."""
        if self.mapping == "flat":
            dx, dy, dz = x - self.centre[0], y - self.centre[1], z - self.centre[2]
            dot = lambda v: dx * v[0] + dy * v[1] + dz * v[2]
            return dot(self.right), dot(self.up), dot(self.normal)
        dx, dy, dz = x - self.origin[0], y - self.origin[1], z - self.origin[2]
        dot = lambda v: dx * v[0] + dy * v[1] + dz * v[2]
        h = dot(self.axis)
        side = np.cross(self.axis, self.toward)
        a, b = dot(self.toward), dot(side)
        s, t = self.radius * np.arctan2(b, a), h - self.height   # unrolled: along the circumference, up
        c, k = self.turn
        return s * c + t * k, t * c - s * k, np.sqrt(a * a + b * b) - self.radius


def picture_field(placement, darkness, width, offset=(0.0, 0.0), invert=False, band=2.0, key=""):
    """Field 0..1: the picture (darkness, rows from the top) `width` mm wide (its height follows its
    proportions), centred on the placement plus `offset`, within `band` mm of the face; 0 elsewhere."""
    rows, cols = darkness.shape
    height = width * rows / cols
    image = Shared(np.ascontiguousarray(darkness, dtype=np.float32))
    ox, oy, band, width = float(offset[0]), float(offset[1]), float(band), float(width)

    def fn(ctx):
        grid = ctx.grid
        X, Y, Z = (a.reshape([-1 if d == k else 1 for d in range(3)]) for k, a in enumerate(grid.axes()))
        s, t, off = placement.coordinates(X, Y, Z)
        s, t, off = np.broadcast_arrays(s, t, off)
        u = ((s - ox) / width + 0.5) * (cols - 1)            # pixel column, row (row 0 at the top)
        v = (0.5 - (t - oy) / height) * (rows - 1)
        inside = (u >= 0) & (u <= cols - 1) & (v >= 0) & (v <= rows - 1) & (np.abs(off) <= band)
        u0 = np.clip(np.floor(u), 0, cols - 2).astype(np.int64)
        v0 = np.clip(np.floor(v), 0, rows - 2).astype(np.int64)
        fu, fv = np.clip(u - u0, 0, 1), np.clip(v - v0, 0, 1)
        img = image.get()
        value = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
                 + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
        if invert:
            value = 1.0 - value
        return ctx.xp.asarray(np.where(inside, value, 0.0).astype(np.float32))
    digest = hashlib.sha1(np.ascontiguousarray(darkness).tobytes()).hexdigest()
    field = Field(fn, _digest("picture", key, digest, width, ox, oy, bool(invert), band,
                              placement.mapping, tuple(np.round(placement.centre, 9))),
                  "picture", range=(0.0, 1.0))
    reach = math.hypot(width, height) / 2 + abs(ox) + abs(oy) + band
    field.extent = (placement.centre - reach, placement.centre + reach)
    return field
