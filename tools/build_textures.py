#!/usr/bin/env python3
"""Authors the game's tileable surface textures and their normal maps.

    python3 tools/build_textures.py assets

Everything is synthesised here (no source images), so the results are free to
use. Each surface gets an albedo map and a tangent-space normal map
(+R = +u, +G = +v with v running down the image, +B = out of the surface).

    ground_*   packed dirt with pebbles and cracks        (tile about 4 m)
    wood_*     weathered bare wood, grain along u         (tile about 2 m)
    cloth_*    coarse plain-weave cloth, neutral grey     (tile about 12 cm)
    leather_*  pebbled leather with creases, neutral grey (tile about 20 cm)

The wood, cloth and leather albedos average mid-grey so the game can tint
them with each object's own colour. Replace any file with a photo texture of
the same name to upgrade it.
"""
import os, sys
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

OUT = os.path.join(sys.argv[1], 'textures')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1887)

def spectral(n, beta, sx=1.0, sy=1.0, lo=0.0):
    """Periodic noise with a 1/f^beta spectrum; sx/sy stretch it along x/y. Returns values in 0..1."""
    fy = np.fft.fftfreq(n)[:, None] * n * sy
    fx = np.fft.fftfreq(n)[None, :] * n * sx
    f = np.sqrt(fx * fx + fy * fy); f[0, 0] = 1
    amp = f ** (-beta / 2); amp[f < lo] = 0; amp[0, 0] = 0
    x = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * amp).real
    x -= x.min(); return x / x.max()

def blur(a, s): return ndimage.gaussian_filter(a, s, mode='wrap')
def norm01(a): a = a - a.min(); return a / max(a.max(), 1e-9)
def smooth(e0, e1, x): t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

def normal_map(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    n = np.dstack([-dx, -dy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return n * 0.5 + 0.5

def save(name, rgb, q=88):
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, name), quality=q)
    print('wrote', name, os.path.getsize(os.path.join(OUT, name)) // 1024, 'KB')

def worley(n, count):
    """Distances to the nearest and second-nearest of `count` random points, with wrap-around."""
    pts = rng.random((count, 2)) * n
    yy, xx = np.mgrid[0:n, 0:n]
    d, idx = cKDTree(pts, boxsize=n).query(np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5]), k=2)
    return d[:, 0].reshape(n, n), d[:, 1].reshape(n, n), idx[:, 0].reshape(n, n)

# ---------------------------------------------------------------- ground
def ground(n=1024):
    base = spectral(n, 2.4)
    mid = spectral(n, 1.6, lo=6)
    fine = spectral(n, 0.6, lo=40)
    h = 0.55 * base + 0.3 * mid + 0.1 * fine
    # pebbles of mixed size sitting on the surface
    peb = np.zeros((n, n)); tint = np.zeros((n, n)); mask = np.zeros((n, n))
    for count, r in ((260, 11.0), (900, 5.5), (2600, 2.6)):
        d1, _, idx = worley(n, count)
        rad = r * (0.55 + 0.9 * rng.random(count))[idx]
        present = (rng.random(count) < 0.3)[idx]
        p = np.clip(1 - (d1 / rad) ** 2, 0, 1) ** 0.6 * present
        peb = np.maximum(peb, p * (0.3 + r / 22))
        k = p > 0.02
        tint[k] = (rng.random(count)[idx])[k]; mask = np.maximum(mask, smooth(0.02, 0.25, p))
    # dried cracks
    cr = np.abs(spectral(n, 1.9, lo=3) - 0.5)
    cracks = (1 - smooth(0.0, 0.012, cr)) * smooth(0.45, 0.6, spectral(n, 2.2))
    h = h + peb * 0.55 - cracks * 0.25
    dirt_a, dirt_b = np.array([0.50, 0.36, 0.25]), np.array([0.70, 0.55, 0.40])
    col = dirt_a + (dirt_b - dirt_a) * (0.5 * base + 0.5 * mid)[..., None]
    col *= (0.82 + 0.36 * fine)[..., None]
    stone = np.array([0.50, 0.44, 0.38])[None, None, :] * (0.55 + 0.6 * tint)[..., None]
    col = col * (1 - mask[..., None] * 0.85) + stone * mask[..., None] * 0.85
    occ = smooth(-0.02, 0.05, h - blur(h, 6))
    col *= (0.72 + 0.28 * occ)[..., None]
    col *= (1 - 0.45 * cracks)[..., None]
    save('ground_albedo.jpg', col)
    save('ground_normal.jpg', normal_map(blur(h, 0.7), 26.0), 90)

# ---------------------------------------------------------------- wood
def wood(n=1024):
    # long streaky grain: noise that varies quickly across v and slowly along u
    g1 = spectral(n, 1.5, sx=14, sy=1, lo=3)
    g2 = spectral(n, 1.2, sx=30, sy=1, lo=10)
    warp = (spectral(n, 2.6, sx=3, sy=1) - 0.5) * 60
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # a few knots bend the grain around themselves
    for _ in range(5):
        cx, cy, r = rng.random() * n, rng.random() * n, 26 + rng.random() * 30
        dx = np.minimum(np.abs(xx - cx), n - np.abs(xx - cx)); dy = (yy - cy + n / 2) % n - n / 2
        warp += np.exp(-(dx * dx / (r * r * 4) + dy * dy / (r * r))) * np.sign(dy) * r * 0.9
    rows = (yy + warp) % n
    ring = 0.5 + 0.5 * np.sin(rows / n * 2 * np.pi * 26 + g1 * 9)
    sample = lambda a: ndimage.map_coordinates(a, [rows, xx], order=1, mode='wrap')
    grain = 0.45 * sample(g1) + 0.3 * sample(g2) + 0.25 * ring
    # weathering checks: thin dark splits along the grain
    ck = sample(spectral(n, 1.0, sx=40, sy=1, lo=14))
    checks = smooth(0.74, 0.86, ck) * smooth(0.4, 0.7, spectral(n, 2.0, sx=6))
    pores = spectral(n, 0.4, sx=8, sy=1, lo=60)
    h = grain * 0.7 + pores * 0.12 - checks * 0.5
    v = 0.38 + 0.42 * norm01(grain) + 0.08 * pores - 0.3 * checks
    v = v / v.mean() * 0.5                                    # average mid-grey: used as a tint
    col = np.dstack([v * 1.05, v * 1.0, v * 0.94])
    save('wood_albedo.jpg', col)
    save('wood_normal.jpg', normal_map(blur(h, 0.6), 18.0), 90)

# ---------------------------------------------------------------- cloth
def cloth(n=512, threads=44):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    wob = (spectral(n, 2.0, lo=2) - 0.5) * 0.35
    u = xx / n * threads + wob; v = yy / n * threads + (spectral(n, 2.0, lo=2) - 0.5) * 0.35
    fu, fv, iu, iv = u % 1, v % 1, np.floor(u), np.floor(v)
    across = lambda f: np.clip(np.cos((f - 0.5) * np.pi), 0, 1) ** 0.6
    warp = across(fu) * (0.5 + 0.5 * np.cos(np.pi * v + np.pi * iu))       # vertical threads
    weft = across(fv) * (0.5 - 0.5 * np.cos(np.pi * u + np.pi * iv))       # horizontal threads
    h = np.maximum(warp, weft)
    fib = spectral(n, 0.5, lo=30)
    h = h * (0.85 + 0.3 * fib) + 0.1 * spectral(n, 1.8, lo=3)
    v0 = 0.3 + 0.55 * h + 0.12 * fib
    v0 = v0 / v0.mean() * 0.5
    save('cloth_albedo.jpg', np.dstack([v0, v0, v0]))
    save('cloth_normal.jpg', normal_map(blur(h, 0.8), 14.0), 90)

# ---------------------------------------------------------------- leather
def leather(n=512):
    d1, d2, _ = worley(n, 2600)
    cells = blur(smooth(0.0, 2.6, d2 - d1), 1.1)             # raised pebbles between grooves
    d1b, d2b, _ = worley(n, 300)
    big = blur(smooth(0.0, 6.0, d2b - d1b), 2.0)
    cr = np.abs(spectral(n, 2.2, lo=2) - 0.5)
    creases = blur((1 - smooth(0.0, 0.012, cr)) * smooth(0.5, 0.7, spectral(n, 2.4)), 0.8)
    h = 0.6 * cells + 0.2 * big - 0.6 * creases + 0.08 * spectral(n, 0.6, lo=30)
    v = 0.62 + 0.16 * cells + 0.06 * big - 0.2 * creases + 0.1 * spectral(n, 1.8, lo=2)
    v = v / v.mean() * 0.5
    save('leather_albedo.jpg', np.dstack([v, v, v]))
    save('leather_normal.jpg', normal_map(blur(h, 0.9), 9.0), 90)

ground(); wood(); cloth(); leather()
