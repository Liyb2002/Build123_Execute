import numpy as np
import copy
import random
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection


def do_perturb(stroke_node_features, perturb_factor=0.002):
    """
    Step 0 (skeleton):
    - Walk each stroke, detect its type (row[9]).
    - For now, do nothing (pass) in every case.
    - Return [] per your instruction.

    Final (future) output format per stroke:
      a list of 10 points, each point is [x, y, z],
      i.e. [[x1,y1,z1], ..., [x10,y10,z10]]
    """
    result = []  # For now, empty as all functions are pass

    for stroke in stroke_node_features:
        t = stroke[9]

        if t == 1:
            # Straight Line
            result.append(perturb_straight_line(stroke))

        elif t == 2:
            # Circle
            result.append(perturb_circle(stroke))

        elif t == 3:
            # Cylinder face
            pass
        elif t == 4:
            # Arc
            pass
        elif t == 5:
            # Spline
            pass
        elif t == 6:
            # Sphere
            pass
        else:
            # Unknown type
            pass

    return result



def perturb_straight_line(stroke, rng=None):
    """
    Adapted from your reference logic for straight lines.

    Input:
        stroke: [x1,y1,z1, x2,y2,z2, 0,0,0, 1]

    Output:
        list of 10 points, each [x,y,z]
    """
    if rng is None:
        rng = random

    # ------ vector helpers ------
    def v_add(a, b):   return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def v_sub(a, b):   return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
    def v_mul(a, s):   return [a[0]*s, a[1]*s, a[2]*s]
    def v_len(a):      return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    def v_norm(a):
        L = v_len(a)
        return [1.0, 0.0, 0.0] if L < 1e-12 else [a[0]/L, a[1]/L, a[2]/L]
    def v_lerp(a, b, t):
        return [a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t, a[2]*(1-t)+b[2]*t]
    def gauss3(scale):
        return [rng.gauss(0.0, scale), rng.gauss(0.0, scale), rng.gauss(0.0, scale)]
    def uniform3(a, b):
        return [rng.uniform(a, b), rng.uniform(a, b), rng.uniform(a, b)]

    # ------ parse endpoints ------
    p1 = [float(stroke[0]), float(stroke[1]), float(stroke[2])]
    p2 = [float(stroke[3]), float(stroke[4]), float(stroke[5])]
    L = v_len(v_sub(p2, p1))
    if L < 1e-12:
        # Degenerate: just return 10 identical points
        return [p1[:] for _ in range(10)]

    # ------ cad2sketch-style random strengths ------
    point_jitter_ratio   = rng.uniform(0.0002, 0.001)
    endpoint_shift_ratio = rng.uniform(0.005, 0.01)
    overdraw_ratio       = rng.uniform(0.03, 0.06)

    point_jitter   = point_jitter_ratio * L
    endpoint_shift = endpoint_shift_ratio * L
    overdraw       = overdraw_ratio * L

    # ------ create a small polyline (5 evenly spaced points) ------
    base_ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    pts = [v_lerp(p1, p2, t) for t in base_ts]

    # ------ perturb original geometry ------
    # endpoints: uniform cube shift; interior: Gaussian per axis
    for i in range(len(pts)):
        if i == 0 or i == len(pts)-1:
            shift = uniform3(-endpoint_shift, endpoint_shift)
        else:
            shift = gauss3(point_jitter)
        pts[i] = v_add(pts[i], shift)

    # ------ overdraw at both ends ------
    v_start = v_sub(pts[1], pts[0])
    v_end   = v_sub(pts[-2], pts[-1])
    v_start = v_norm(v_start)
    v_end   = v_norm(v_end)
    pts[0]  = v_sub(pts[0],  v_mul(v_start, overdraw))
    pts[-1] = v_sub(pts[-1], v_mul(v_end,   overdraw))

    # ------ resample 10 evenly spaced points between new endpoints ------
    start = pts[0]
    end   = pts[-1]
    ts = [i/9.0 for i in range(10)]
    resampled = [v_lerp(start, end, t) for t in ts]

    # ------ jitter interior points only ------
    for i in range(1, len(resampled)-1):
        resampled[i] = v_add(resampled[i], gauss3(point_jitter))

    return resampled



def perturb_circle(stroke, start_angle=0.0, rng=None):
    """
    Perturb/synthesize a hand-drawn-looking CLOSED circle polyline (10 points).
    Input stroke: [cx,cy,cz, nx,ny,nz, 0, radius, 0, 2]  (radius at index 7)
    Returns: list of 10 points [x,y,z], with pts[-1] == pts[0].
    """
    if rng is None:
        rng = random

    # --- parse ---
    cx, cy, cz = float(stroke[0]), float(stroke[1]), float(stroke[2])
    nx, ny, nz = float(stroke[3]), float(stroke[4]), float(stroke[5])
    R = float(stroke[7])

    # --- vec utils ---
    def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    def add(a, b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def mul(a, s): return [a[0]*s, a[1]*s, a[2]*s]
    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def norm(a):
        L2 = dot(a, a)
        if L2 <= 0.0:
            return [0.0, 0.0, 1.0]
        L = math.sqrt(L2)
        return [a[0]/L, a[1]/L, a[2]/L]

    center = [cx, cy, cz]
    n = norm([nx, ny, nz])
    if abs(n[0]) + abs(n[1]) + abs(n[2]) < 1e-12:
        n = [0.0, 0.0, 1.0]

    # --- build in-plane orthonormal basis (right-handed) ---
    ref = [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = norm(cross(n, ref))
    if abs(u[0]) + abs(u[1]) + abs(u[2]) < 1e-12:
        ref = [0.0, 1.0, 0.0]
        u = norm(cross(n, ref))
    v = cross(n, u)

    if R <= 1e-12:
        # Degenerate: return closed “point circle”
        p = center[:]
        return [p, p, p, p, p, p, p, p, p, p]

    # --- ellipse-ish params & jitter (non-uniform angles) ---
    rx = R * rng.uniform(0.9, 1.1)
    ry = R * rng.uniform(0.9, 1.1)
    phi = rng.uniform(0.0, 2.0 * math.pi)             # in-plane rotation
    jitter_2d = rng.uniform(0.001, 0.004) * R         # wobble in plane
    angle_jitter = 0.15                                # radians; < (2π/9)/2 so order stays increasing

    cphi, sphi = math.cos(phi), math.sin(phi)

    # --- 9 slightly non-uniform angles (then close with first) ---
    pts = []
    for i in range(9):  # 0..8
        base = start_angle + 2.0 * math.pi * (i / 9.0)
        t = base + rng.uniform(-angle_jitter, angle_jitter)

        # ellipse in local 2D (then rotate by phi)
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        xr = cphi * x - sphi * y
        yr = sphi * x + cphi * y

        # in-plane jitter
        xr += rng.gauss(0.0, jitter_2d)
        yr += rng.gauss(0.0, jitter_2d)

        # back to 3D
        p = add(center, add(mul(u, xr), mul(v, yr)))
        pts.append(p)

    # optional gentle seam smoothing before closing
    seam_shift_len = rng.uniform(0.05, 0.1) * R
    seam_shift = [rng.gauss(0.0, seam_shift_len),
                  rng.gauss(0.0, seam_shift_len),
                  rng.gauss(0.0, seam_shift_len)]
    for idx, w in zip([8, 7, 6], [1.0, 0.8, 0.6]):
        pts[idx] = add(pts[idx], mul(seam_shift, w))

    # close the loop
    pts.append(pts[0][:])
    return pts






# ---------------------------------------------------------------------------------------- #

def vis_perturbed_strokes(lines, *, color="black", linewidth=0.8, show=True):
    """
    Visualize perturbed strokes with equal scaling across x/y/z.

    Parameters
    ----------
    lines : list[list[list[float]]]
        Iterable of strokes; each stroke is a list of [x, y, z] points.
    color : str
        Line color for all strokes.
    linewidth : float
        Width of the stroke lines.
    show : bool
        If True, calls plt.show() at the end.
    """
    if not lines or not any(lines):
        raise ValueError("`lines` must contain at least one stroke with points.")

    # Gather mins/maxes across all points
    x_min = y_min = z_min = float("inf")
    x_max = y_max = z_max = float("-inf")

    for pts in lines:
        for x, y, z in pts:
            if x < x_min: x_min = x
            if y < y_min: y_min = y
            if z < z_min: z_min = z
            if x > x_max: x_max = x
            if y > y_max: y_max = y
            if z > z_max: z_max = z

    # Compute the center and a uniform half-size (max extent)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    # Handle the degenerate case where all points coincide
    if max_diff == 0:
        max_diff = 1.0  # arbitrary unit cube

    half = max_diff / 2.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot strokes
    for pts in lines:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth)

    # Equalize axes: set symmetric limits around the center with uniform range
    ax.set_xlim([x_center - half, x_center + half])
    ax.set_ylim([y_center - half, y_center + half])
    ax.set_zlim([z_center - half, z_center + half])

    # Force a 1:1:1 box aspect if available (Matplotlib 3.3+)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        # Older Matplotlib versions won't have set_box_aspect; limits above still help.
        pass

    # Optional aesthetics
    ax.set_axis_off()
    ax.grid(False)

    if show:
        plt.show()

    return fig, ax
