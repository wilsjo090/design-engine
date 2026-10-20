import numpy as np
import trimesh

from engine import fields as F
from engine.slice import field_slice


def test_z_slice_orientation_and_values():
    # f = x + 10*y: columns run along x, rows along y
    img, rect = field_slice(F.X + 10 * F.Y, "z", 3.0, (0, 0, 0), (10, 5, 8), res=100)
    assert rect == (0.0, 0.0, 10.0, 5.0)
    assert img.shape == (51, 101)                      # 0.1 mm pixels
    assert np.isclose(img[0, 0], 0) and np.isclose(img[0, -1], 10)    # along x
    assert np.isclose(img[-1, 0], 50)                                  # along y


def test_y_and_x_slices_use_the_right_axes():
    img, rect = field_slice(F.Z, "y", 1.0, (0, 0, 0), (4, 9, 2), res=40)   # columns x, rows z
    assert np.isclose(img[-1, 0], 2) and np.isclose(img[0, -1], 0)
    img, rect = field_slice(F.Y, "x", 1.0, (0, 0, 0), (4, 6, 2), res=60)   # columns y, rows z
    assert np.isclose(img[0, -1], 6) and np.isclose(img[-1, 0], 0)


def test_list_of_body_fields_is_composited_inside_their_boxes():
    boxes = [trimesh.creation.box((4, 4, 4)).apply_translation((x, 0, 0)) for x in (-5, 5)]
    bodies = [F.body_from_mesh(b, band=1.0) for b in boxes]
    img, rect = field_slice(bodies, "z", 0.0, (-10, -5, -5), (10, 5, 5), res=200)
    centre_col = lambda x: int(round((x - rect[0]) / (rect[2] - rect[0]) * (img.shape[1] - 1)))
    mid_row = img.shape[0] // 2
    assert np.isclose(img[mid_row, centre_col(-5)], -2, atol=0.05)   # inside the left box
    assert np.isclose(img[mid_row, centre_col(5)], -2, atol=0.05)    # inside the right box
    assert np.isnan(img[mid_row, centre_col(0)])                     # between boxes: not covered
