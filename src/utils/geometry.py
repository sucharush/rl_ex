import numpy as np
from dataclasses import dataclass
from src.environments.points import PointSet

def cross2d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def prependicular(segments: np.ndarray) -> np.ndarray:
    """outward normals for CCW: (dy, -dx)"""
    normals = np.stack([segments[:, 1], -segments[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals

def point_inside_segments(o: np.ndarray, segments: np.ndarray) -> bool:
    # segments in ccw!!!
    edges = segments[:, 1] - segments[:, 0]  # (N,2)
    # outward = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    # outward /= np.linalg.norm(outward, axis=1, keepdims=True)
    outward = prependicular(edges)
    # inward = -outward
    vecs = o - segments[:, 0]
    dots = np.einsum("ij,ij->i", vecs, outward)
    return np.all(dots <= 1e-12)  # allow small tolerance


def rays_segments_intersection(
    o: np.ndarray, dirs: np.ndarray, p1s: np.ndarray, p2s: np.ndarray
):
    """
    Vectorized ray-segments intersection.
    o: (2,) ray origin
    dirs: (M,2) array of ray directions (normalized)
    p1s, p2s: (N,2) arrays of segment endpoints
    Returns: (M,) array with min positive t for each ray, or np.inf if no hit.
    """
    # Reshape for broadcasting
    d = dirs[:, None, :]  # (M,1,2)
    v = (p2s - p1s)[None, :, :]  # (1,N,2)
    w = (p1s - o)[None, :, :]  # (1,N,2)

    denom = np.cross(d, v)  # (M,N)
    t = np.cross(w, v) / denom  # (M,N)
    u = np.cross(w, d) / denom  # (M,N)

    # Valid mask
    mask = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)

    # Fill invalid with +inf so min works
    t_valid = np.where(mask, t, np.inf)

    # Min over segments axis -> (M,)
    t_min = np.min(t_valid, axis=1)
    return t_min

def rays_segments_span(o, dirs, p1s, p2s):
    """
    Vectorized ray-segments intersection.
    Returns (t_min, t_max) for each ray.
    If no hit, return (+inf, -inf).
    """
    d = dirs[:, None, :]  # (M,1,2)
    v = (p2s - p1s)[None, :, :]  # (1,N,2)
    w = (p1s - o)[None, :, :]  # (1,N,2)

    denom = np.cross(d, v)  # (M,N)
    t = np.cross(w, v) / denom  # (M,N)
    u = np.cross(w, d) / denom  # (M,N)

    mask = (np.abs(denom) > 1e-12) & (u >= 0) & (u <= 1)
    t_valid = np.where(mask, t, np.nan)

    # Compute min/max over valid intersections
    t_min = np.nanmin(t_valid, axis=1)
    t_max = np.nanmax(t_valid, axis=1)

    # Fallback for rays with no hits
    t_min[np.isnan(t_min)] = np.inf
    t_max[np.isnan(t_max)] = -np.inf

    return t_min, t_max

def rays_segments_span(o, dirs, p1s, p2s, tol=1e-12):
    d = dirs[:, None, :]        # (M,1,2) ray directions
    v = (p2s - p1s)[None, :, :] # (1,N,2)
    w = (p1s - o)[None, :, :]   # (1,N,2)

    denom = cross2d(d, v)      # (M,N)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross2d(w, v) / denom
        u = cross2d(w, d) / denom

    # --- normal case ---
    mask = (np.abs(denom) > tol) & (u >= 0) & (u <= 1)
    t_valid = np.where(mask, t, np.nan)

    # --- collinear case ---
    mask_col = np.abs(denom) <= tol
    if np.any(mask_col):
        d_unit = d / np.linalg.norm(d, axis=-1, keepdims=True)
        # project both endpoints
        t1 = np.einsum("mnj,mnj->mn", (p1s-o)[None,:,:], d_unit)
        t2 = np.einsum("mnj,mnj->mn", (p2s-o)[None,:,:], d_unit)
        # keep only when truly collinear
        t1 = np.where(mask_col, t1, np.nan)
        t2 = np.where(mask_col, t2, np.nan)

        # use both endpoints as candidate intersections
        t_valid = np.concatenate([t_valid, t1, t2], axis=1)

    # reduce to min/max per ray
    t_min = np.nanmin(t_valid, axis=1)
    t_max = np.nanmax(t_valid, axis=1)

    # rays with no hits
    t_min[np.isnan(t_min)] = np.inf
    t_max[np.isnan(t_max)] = -np.inf
    # print(t_min, t_max)
    return t_min, t_max


def normalize_angle(theta):
    return np.mod(theta, 2 * np.pi)
