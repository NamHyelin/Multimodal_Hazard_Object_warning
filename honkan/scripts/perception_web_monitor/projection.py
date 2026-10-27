"""Shared projection geometry for the viewer's 2D envelope / 3D wireframe switch."""
import cv2
import numpy as np

EDGES = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]


def box_geometry(corners, k, d, width, height):
    """Project near-plane clipped edges. Return screen segments and a clipped 2D envelope.

    Both displays use the same geometry; no calibration, object size or pose changes.
    Envelope bounds are computed BEFORE screen clipping, so off-screen corners do
    not shrink the enclosing rectangle to the surviving wireframe segments.
    """
    if not np.isfinite(corners).all():
        return [], None
    points, segments = [], []
    for i, j in EDGES:
        a, b = corners[i].copy(), corners[j].copy()
        if a[2] < .1 and b[2] < .1:
            continue
        if a[2] < .1:
            a += (b-a)*((.1-a[2])/(b[2]-a[2]))
        if b[2] < .1:
            b += (a-b)*((.1-b[2])/(a[2]-b[2]))
        uv = cv2.projectPoints(np.array([a,b]), np.zeros(3), np.zeros(3), k, d)[0].reshape(2,2)
        if not np.isfinite(uv).all() or np.abs(uv).max() > 1e7:
            continue
        points.extend(uv)
        ok, p, q = cv2.clipLine((0,0,width,height), tuple(uv[0].astype(int)), tuple(uv[1].astype(int)))
        if ok:
            segments.append((p,q))
    if not points:
        return segments, None
    points = np.asarray(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    if hi[0] < 0 or hi[1] < 0 or lo[0] > width-1 or lo[1] > height-1:
        return segments, None
    lo = np.maximum(0, np.floor(lo)).astype(int)
    hi = np.minimum([width-1,height-1], np.ceil(hi)).astype(int)
    if np.any(hi <= lo):
        return segments, None
    return segments, (tuple(lo), tuple(hi))
