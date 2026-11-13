#!/usr/bin/env python3
"""Builds the game's head and hand models from MakeHuman's base mesh.

MakeHuman's base mesh, morph targets, skeleton and eye assets were released
as CC0 (see https://github.com/makehumancommunity/makehuman, LICENSE.md).
This script downloads nothing: point it at a folder holding the source files.

    python3 tools/build_models.py <makehuman-data-dir> assets

Needed in <makehuman-data-dir> (all from makehuman/data in that repository):
    base.obj  default.mhskel  default_weights.mhw
    caucasian-male-young.target  caucasian-male-old.target
    high-poly.obj  high-poly.mhclo  brown_eye.png

An optional third argument is a folder holding Khronos' CC0 "Lantern" sample
(Lantern.gltf, Lantern.bin and its four textures); it is repacked as
assets/models/lantern.glb with smaller textures.

Output (glTF binary, metres, character facing +z, y up):
    assets/models/head_a.glb, head_b.glb   two faces, with eyes
    assets/models/hand_open.glb, hand_grip.glb   in hand space (see below)
    assets/textures/eye_brown.jpg, eye_blue.jpg

The game's skeleton is left-handed, so MakeHuman's ".L" side becomes the
character's right. Hands are written in "hand space": origin at the wrist,
x toward the thumb, y toward the fingers, z out of the back of the hand.
"""
import json, math, os, struct, sys, colorsys
import numpy as np
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'textures'), exist_ok=True)
src = lambda n: os.path.join(SRC, n)

# ---------------------------------------------------------------- loading
def load_obj(path):
    V, VT, F = [], [], []
    grp = None
    for line in open(path):
        p = line.split()
        if not p: continue
        if p[0] == 'v': V.append([float(x) for x in p[1:4]])
        elif p[0] == 'vt': VT.append([float(x) for x in p[1:3]])
        elif p[0] == 'g': grp = p[1]
        elif p[0] == 'f':
            vi = [int(t.split('/')[0]) - 1 for t in p[1:]]
            ti = [int(t.split('/')[1]) - 1 if '/' in t and t.split('/')[1] else 0 for t in p[1:]]
            F.append((vi, ti, grp))
    return np.array(V), np.array(VT), F

def load_target(path):
    idx, off = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip(): continue
        p = line.split()
        idx.append(int(p[0])); off.append([float(x) for x in p[1:4]])
    return np.array(idx, dtype=int), np.array(off).reshape(-1, 3)

BASE, _, BASE_F = load_obj(src('base.obj'))
SKEL = json.load(open(src('default.mhskel')))
WEIGHTS = json.load(open(src('default_weights.mhw')))['weights']
TARGETS = {n: load_target(src(n + '.target')) for n in ['caucasian-male-young', 'caucasian-male-old']}
BODY_F = [f[0] for f in BASE_F if f[2] == 'body']
NV = len(BASE)

def weight_of(names):
    """Summed skin weight per vertex for a list of bones."""
    w = np.zeros(NV)
    for n in names:
        for vi, wt in WEIGHTS.get(n, []): w[vi] += wt
    return w

def morph(mix):
    V = BASE.copy()
    for name, k in mix.items():
        idx, off = TARGETS[name]
        V[idx] += off * k
    return V

def joints(V):
    return {k: V[v].mean(0) for k, v in SKEL['joints'].items()}
def bone(J, name):
    b = SKEL['bones'][name]
    return J[b['head']], J[b['tail']]

# ---------------------------------------------------------------- mesh utilities
def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)

def compact(V, faces, attrs=()):
    """Keep only the vertices the faces use."""
    used = sorted({i for f in faces for i in f})
    remap = {o: n for n, o in enumerate(used)}
    return V[used], [[remap[i] for i in f] for f in faces], [a[used] for a in attrs]

def subdivide(V, faces, attrs):
    """One round of Catmull-Clark on a quad mesh. Attributes are carried along linearly."""
    nv = len(V)
    fpts = np.array([V[f].mean(0) for f in faces])
    edges = {}
    efaces = {}
    for fi, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            k = (min(a, b), max(a, b))
            edges.setdefault(k, len(edges)); efaces.setdefault(k, []).append(fi)
    ekeys = sorted(edges, key=lambda k: edges[k])
    epts = np.zeros((len(ekeys), 3))
    for i, k in enumerate(ekeys):
        fs = efaces[k]
        epts[i] = (V[k[0]] + V[k[1]] + fpts[fs].sum(0)) / (2 + len(fs)) if len(fs) == 2 else (V[k[0]] + V[k[1]]) / 2
    # original points
    fsum = np.zeros((nv, 3)); fcnt = np.zeros(nv); esum = np.zeros((nv, 3)); ecnt = np.zeros(nv)
    bsum = np.zeros((nv, 3)); bcnt = np.zeros(nv)
    for fi, f in enumerate(faces):
        for i in f: fsum[i] += fpts[fi]; fcnt[i] += 1
    for k in ekeys:
        mid = (V[k[0]] + V[k[1]]) / 2
        for i in k: esum[i] += mid; ecnt[i] += 1
        if len(efaces[k]) == 1:
            for i in k: bsum[i] += mid; bcnt[i] += 1
    newV = V.copy()
    for i in range(nv):
        if fcnt[i] == 0: continue
        if bcnt[i] > 0: newV[i] = 0.5 * V[i] + 0.5 * bsum[i] / bcnt[i]        # boundary rule
        else:
            n = fcnt[i]
            newV[i] = (fsum[i] / n + 2 * esum[i] / ecnt[i] + (n - 3) * V[i]) / n
    outV = np.vstack([newV, epts, fpts])
    outA = []
    for a in attrs:
        ea = np.array([(a[k[0]] + a[k[1]]) / 2 for k in ekeys])
        fa = np.array([a[f].mean(0) for f in faces])
        outA.append(np.concatenate([a, ea, fa]))
    eo, fo = nv, nv + len(ekeys)
    outF = []
    for fi, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            a, b, c = f[j - 1], f[j], f[(j + 1) % n]
            outF.append([b, eo + edges[(min(b, c), max(b, c))], fo + fi, eo + edges[(min(a, b), max(a, b))]])
    return outV, outF, outA

def vertex_normals(V, faces):
    N = np.zeros_like(V)
    for f in faces:
        p = V[f]
        n = np.cross(p[1] - p[0], p[2] - p[0]) if len(f) == 3 else np.cross(p[2] - p[0], p[3] - p[1])
        for i in f: N[i] += n
    return normalize(N)

def cavity(V, faces, N, scale):
    """Cheap ambient occlusion: how far a vertex sits below the average of its neighbours."""
    acc = np.zeros_like(V); cnt = np.zeros(len(V))
    for f in faces:
        c = V[f].mean(0)
        for i in f: acc[i] += c; cnt[i] += 1
    d = ((acc / np.maximum(cnt, 1)[:, None] - V) * N).sum(1)
    return np.clip(1.0 - np.clip(d / scale, 0, 1) * 0.75, 0.25, 1.0)

def triangles(faces):
    out = []
    for f in faces:
        out.append([f[0], f[1], f[2]])
        if len(f) == 4: out.append([f[0], f[2], f[3]])
    return np.array(out, dtype=np.uint32)

lin = lambda c: np.array(c, dtype=float) ** 2.2        # sRGB (0-1) -> linear

# ---------------------------------------------------------------- glTF binary writer
def write_glb(path, prims, images=(), extras=None):
    """prims: dicts with pos, nor, idx and optional col, uv, name, texture (index into images)."""
    bin_ = bytearray(); views = []; accs = []
    def view(data, target=None):
        while len(bin_) % 4: bin_.append(0)
        v = {'buffer': 0, 'byteOffset': len(bin_), 'byteLength': len(data)}
        if target: v['target'] = target
        bin_.extend(data); views.append(v); return len(views) - 1
    def acc(arr, ctype, typ, target=None, minmax=False):
        a = {'bufferView': view(arr.tobytes(), target), 'componentType': ctype, 'count': len(arr), 'type': typ}
        if minmax: a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
        accs.append(a); return len(accs) - 1
    mats, meshes_prims = [], []
    for p in prims:
        p = dict(p)
        at = {'POSITION': acc(p['pos'].astype('<f4'), 5126, 'VEC3', 34962, True), 'NORMAL': acc(p['nor'].astype('<f4'), 5126, 'VEC3', 34962)}
        if 'col' in p: at['COLOR_0'] = acc(np.hstack([p['col'], np.ones((len(p['col']), 1))]).astype('<f4'), 5126, 'VEC4', 34962)
        if 'uv' in p: at['TEXCOORD_0'] = acc(p['uv'].astype('<f4'), 5126, 'VEC2', 34962)
        m = {'name': p.get('name', 'material'), 'pbrMetallicRoughness': {'metallicFactor': 0, 'roughnessFactor': p.get('rough', 0.7)}}
        if 'texture' in p: m['pbrMetallicRoughness']['baseColorTexture'] = {'index': p['texture']}
        if 'normalTex' in p: m['normalTexture'] = {'index': p['normalTex']}
        if 'emissiveTex' in p: m['emissiveTexture'] = {'index': p['emissiveTex']}; m['emissiveFactor'] = [1, 1, 1]
        mats.append(m)
        meshes_prims.append({'attributes': at, 'indices': acc(p['idx'].astype('<u4').reshape(-1), 5125, 'SCALAR', 34963), 'material': len(mats) - 1})
    g = {'asset': {'version': '2.0', 'generator': 'sundown-duel build_models.py', 'copyright': 'Derived from MakeHuman assets, CC0'},
         'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}], 'meshes': [{'primitives': meshes_prims}],
         'materials': mats, 'accessors': accs, 'bufferViews': views}
    if images:
        g['images'] = [{'uri': u} for u in images]
        g['textures'] = [{'source': i} for i in range(len(images))]
    if extras: g['asset']['extras'] = extras
    while len(bin_) % 4: bin_.append(0)
    g['buffers'] = [{'byteLength': len(bin_)}]
    js = json.dumps(g, separators=(',', ':')).encode()
    while len(js) % 4: js += b' '
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack('<II', len(bin_), 0x004E4942)); f.write(bin_)
    print('wrote', path, os.path.getsize(path) // 1024, 'KB')

# ---------------------------------------------------------------- eyes
def build_eye_textures():
    im = Image.open(src('brown_eye.png')).convert('RGBA').resize((512, 512), Image.LANCZOS)
    a = np.asarray(im).astype(float) / 255
    rgb, alpha = a[..., :3], a[..., 3]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(OUT, 'textures', 'eye_brown.jpg'), quality=90)
    # steel-blue variant: recolour the saturated iris pixels
    h = np.zeros(rgb.shape[:2]); s = np.zeros_like(h); v = np.zeros_like(h)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            h[y, x], s[y, x], v[y, x] = colorsys.rgb_to_hsv(*rgb[y, x])
    iris = np.clip((s - 0.35) / 0.25, 0, 1)
    out = rgb.copy()
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            if iris[y, x] > 0:
                c = np.array(colorsys.hsv_to_rgb(0.58, min(1, s[y, x] * 0.55), min(1, v[y, x] * 1.5 + 0.08)))
                out[y, x] = rgb[y, x] * (1 - iris[y, x]) + c * iris[y, x]
    Image.fromarray((out * 255).astype(np.uint8)).save(os.path.join(OUT, 'textures', 'eye_blue.jpg'), quality=90)
    return alpha

def fit_eyes(V, alpha):
    """Place MakeHuman's high-poly eyes on a morphed base mesh, dropping the transparent cornea shell."""
    EV, EVT, EF = load_obj(src('high-poly.obj'))
    scale = {}; refs = []; reading = False
    for line in open(src('high-poly.mhclo')):
        p = line.split()
        if not p or p[0].startswith('#'): continue
        if p[0] in ('x_scale', 'y_scale', 'z_scale'):
            ax = 'xyz'.index(p[0][0]); scale[ax] = abs(V[int(p[1])][ax] - V[int(p[2])][ax]) / float(p[3])
        elif p[0] == 'verts': reading = True
        elif reading:
            if len(p) == 9: refs.append([float(x) for x in p])
            elif len(p) == 1: refs.append([float(p[0]), 0, 0, 1, 0, 0, 0, 0, 0])
            else: reading = False
    refs = np.array(refs[:len(EV)])
    sc = np.array([scale[0], scale[1], scale[2]])
    P = (V[refs[:, 0].astype(int)] * refs[:, 3:4] + V[refs[:, 1].astype(int)] * refs[:, 4:5] + V[refs[:, 2].astype(int)] * refs[:, 5:6] + refs[:, 6:9] * sc)
    # un-share vertices so each corner has its own uv; skip faces whose texture is transparent
    pos, uv, faces = [], [], []
    for vi, ti, _ in EF:
        t = EVT[ti]; c = t.mean(0)
        if alpha[int((1 - c[1]) * 511.99) % 512, int(c[0] * 511.99) % 512] < 0.5: continue
        base = len(pos)
        for a, b in zip(vi, ti): pos.append(P[a]); uv.append([EVT[b][0], 1 - EVT[b][1]])
        faces.append(list(range(base, base + len(vi))))
    return np.array(pos), np.array(uv), faces

# ---------------------------------------------------------------- heads
HEAD_TOP, HEAD_HEIGHT, HEAD_Z = 1.757, 0.238, 0.006      # where the game's skeleton expects the head

def build_head(name, mix, skin, eye_tex, alpha, squint):
    V = morph(mix); J = joints(V)
    neck_y = bone(J, 'neck01')[0][1]
    axis = bone(J, 'neck02')[0]
    keep = np.zeros(NV, bool)
    for i in range(NV):
        y = V[i][1]
        keep[i] = y > neck_y - 0.05 or (y > neck_y - 0.55 and math.hypot(V[i][0] - axis[0], V[i][2] - axis[2]) < 0.62)
    faces = [f for f in BODY_F if all(keep[i] for i in f)]
    # masks from the facial bones
    lips = weight_of(['oris01', 'oris02', 'oris03.L', 'oris03.R', 'oris04.L', 'oris04.R', 'oris05', 'oris06', 'oris06.L', 'oris06.R', 'oris07.L', 'oris07.R'])
    lidU = {s: weight_of(['orbicularis03.' + s]) for s in 'LR'}
    lidD = {s: weight_of(['orbicularis04.' + s]) for s in 'LR'}
    jaw = weight_of(['jaw', 'special04'])
    # squint: swing the lids over the eyeballs
    for s in 'LR':
        c = bone(J, 'eye.' + s)[0]
        for w, ang in ((lidU[s], squint), (lidD[s], -squint * 0.45)):
            for i in np.nonzero(w > 0.01)[0]:
                a = ang * min(1.0, w[i] * 1.4); d = V[i] - c
                V[i] = c + np.array([d[0], d[1] * math.cos(a) - d[2] * math.sin(a), d[1] * math.sin(a) + d[2] * math.cos(a)])
    eye_pos, eye_uv, eye_faces = fit_eyes(morph(mix), alpha)

    Vh, Fh, (lip_m, jaw_m, lid_m) = compact(V, faces, (lips, jaw, lidU['L'] + lidU['R'] + lidD['L'] + lidD['R']))
    Vh, Fh, (lip_m, jaw_m, lid_m) = subdivide(Vh, Fh, (lip_m, jaw_m, lid_m))

    # into game space
    head_verts = Vh[Vh[:, 1] > bone(J, 'head')[0][1] - 0.6]
    top = Vh[:, 1].max()
    chin = min(v[1] for v in head_verts if v[2] > 0.9 and abs(v[0]) < 0.35 and v[1] < bone(J, 'head')[0][1])
    k = HEAD_HEIGHT / (top - chin)
    zc = (head_verts[:, 2].max() + head_verts[:, 2].min()) / 2
    T = lambda P: np.column_stack([P[:, 0] * k, HEAD_TOP - (top - P[:, 1]) * k, HEAD_Z + (P[:, 2] - zc) * k])
    pos = T(Vh); N = vertex_normals(pos, Fh)
    ao = cavity(pos, Fh, N, 0.0035)

    base = lin(skin)
    col = np.tile(base, (len(pos), 1))
    lipk = np.clip((lip_m - 0.55) / 0.3, 0, 1)[:, None]
    col = col * (1 - lipk * 0.55) + lin([0.50, 0.30, 0.26]) * lipk * 0.55
    stub = (np.clip((jaw_m - 0.25) / 0.5, 0, 1) * np.clip((1.62 - pos[:, 1]) / 0.05, 0, 1) * np.clip((pos[:, 2] + 0.02) / 0.05, 0, 1))[:, None]
    col = col * (1 - 0.42 * stub * (1 - lipk)) + lin([0.2, 0.17, 0.16]) * 0.1 * stub * (1 - lipk)
    lidk = np.clip(lid_m / 0.6, 0, 1)[:, None]
    col = col * (1 - 0.22 * lidk)
    col *= ao[:, None]

    ep = T(eye_pos); en = vertex_normals(ep, eye_faces)
    L = lambda p: [round(float(x), 4) for x in T(np.array([p]))[0]]
    extras = {
        'eyeR': L(bone(J, 'eye.L')[0]), 'eyeL': L(bone(J, 'eye.R')[0]),
        'browR': [L(bone(J, 'oculi01.L')[1]), L(bone(J, 'oculi01.L')[0]), L(bone(J, 'oculi02.L')[0])],
        'browL': [L(bone(J, 'oculi01.R')[1]), L(bone(J, 'oculi01.R')[0]), L(bone(J, 'oculi02.R')[0])],
        'mouth': L(bone(J, 'oris01')[1]), 'lipTop': L(bone(J, 'oris05')[1]) if 'oris05' in SKEL['bones'] else None,
        'nose': [round(float(x), 4) for x in pos[np.argmax(pos[:, 2] - 4 * abs(pos[:, 0]))]],
        'chin': [0, round(float(HEAD_TOP - HEAD_HEIGHT), 4), round(float(pos[:, 2].max() * 0.6), 4)],
        'top': round(float(pos[:, 1].max()), 4), 'back': round(float(head_verts[:, 2].min() * k - zc * k + HEAD_Z), 4),
        'halfWidth': round(float(np.abs(pos[pos[:, 1] > 1.6][:, 0]).max()), 4),
    }
    write_glb(os.path.join(OUT, 'models', name + '.glb'), [
        {'name': 'skin', 'pos': pos, 'nor': N, 'col': col, 'idx': triangles(Fh), 'rough': 0.62},
        {'name': 'eye', 'pos': ep, 'nor': en, 'uv': eye_uv, 'idx': triangles(eye_faces), 'texture': 0, 'rough': 0.15},
    ], images=['../textures/' + eye_tex], extras=extras)
    return extras

# ---------------------------------------------------------------- hands
def rot_about(axis, ang, pivot):
    a = axis / np.linalg.norm(axis); c, s = math.cos(ang), math.sin(ang); x, y, z = a
    Rm = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                   [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                   [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])
    M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = pivot - Rm @ pivot
    return M

def build_hand(name, mix, skin, curls, thumb):
    S = 'L'                                           # MakeHuman's +x side is the game's right hand
    V = morph(mix); J = joints(V)
    wrist = bone(J, 'wrist.' + S)[0]
    k2, k3, k5 = (bone(J, f'finger{i}-1.{S}')[0] for i in (2, 3, 5))
    y = normalize(k3 - wrist); xr = k2 - k5; x = normalize(xr - y * (xr @ y)); z = np.cross(x, y)
    tip = bone(J, f'finger3-3.{S}')[1]
    assert (tip - k3) @ z < 0, 'expected the relaxed fingers to curl toward the palm (-z)'
    # pose the fingers with linear-blend skinning
    Vp = V.copy(); acc = np.zeros((NV, 3)); wsum = np.zeros(NV)
    for f in range(1, 6):
        M = np.eye(4)
        for j in range(1, 4):
            bn = f'finger{f}-{j}.{S}'; h, t = bone(J, bn)
            if f == 1:
                d = normalize(t - h); target = normalize((k2 + k3) / 2 - z * 0.45 - h)
                ax = np.cross(d, target); ang = thumb[j - 1]
            else:
                ax = -x; ang = curls[f - 2][j - 1]          # rotating about -x curls toward the palm
            M = M @ rot_about(ax, ang, h)
            for vi, wt in WEIGHTS.get(bn, []):
                acc[vi] += wt * (M[:3, :3] @ V[vi] + M[:3, 3]); wsum[vi] += wt
    moved = wsum > 0
    Vp[moved] = acc[moved] + V[moved] * (1 - wsum[moved])[:, None]

    hand_bones = [f'wrist.{S}'] + [f'metacarpal{i}.{S}' for i in range(1, 5)] + [f'finger{f}-{j}.{S}' for f in range(1, 6) for j in range(1, 4)]
    hw = weight_of(hand_bones); aw = weight_of([f'lowerarm01.{S}', f'lowerarm02.{S}'])
    fore = normalize(wrist - bone(J, f'lowerarm02.{S}')[0])
    keep = np.array([hw[i] >= 0.5 or ((hw[i] + aw[i]) > 0.5 and (V[i] - wrist) @ fore > -0.75) for i in range(NV)])
    faces = [f for f in BODY_F if all(keep[i] for i in f)]
    Vh, Fh, _ = compact(Vp, faces)
    Vh, Fh, _ = subdivide(Vh, Fh, ())
    local = (Vh - wrist) @ np.column_stack([x, y, z]) * 0.1 * 1.04      # decimetres -> metres, a touch larger
    N = vertex_normals(local, Fh)
    ao = cavity(local, Fh, N, 0.002)
    col = np.tile(lin(skin), (len(local), 1)) * ao[:, None]
    write_glb(os.path.join(OUT, 'models', name + '.glb'), [{'name': 'skin', 'pos': local, 'nor': N, 'col': col, 'idx': triangles(Fh), 'rough': 0.62}])

# ---------------------------------------------------------------- main
alpha = build_eye_textures()
R = math.radians
build_head('head_a', {'caucasian-male-young': 0.5, 'caucasian-male-old': 0.5}, [0.80, 0.56, 0.42], 'eye_blue.jpg', alpha, R(10))
build_head('head_b', {'caucasian-male-young': 1.0}, [0.76, 0.54, 0.40], 'eye_brown.jpg', alpha, R(8))
mix = {'caucasian-male-young': 1.0}
build_hand('hand_open', mix, [0.78, 0.55, 0.41], [[R(14), R(20), R(12)]] * 4, [R(4), R(8), R(6)])
build_hand('hand_grip', mix, [0.78, 0.55, 0.41],
           [[R(38), R(50), R(28)], [R(74), R(86), R(52)], [R(78), R(88), R(56)], [R(80), R(90), R(58)]], [R(22), R(30), R(32)])

# ---------------------------------------------------------------- lantern prop
def build_lantern(d):
    g = json.load(open(os.path.join(d, 'Lantern.gltf'))); buf = open(os.path.join(d, 'Lantern.bin'), 'rb').read()
    def read(ai):
        a = g['accessors'][ai]; v = g['bufferViews'][a['bufferView']]
        n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
        dt = {5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']]
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.frombuffer(buf, dt, a['count'] * n, off).reshape(-1, n).astype(float if dt == '<f4' else np.uint32)
    P, N, UV, I = [], [], [], []
    for node in g['nodes']:
        if 'mesh' not in node: continue
        t = np.array(node.get('translation', [0, 0, 0]))
        for pr in g['meshes'][node['mesh']]['primitives']:
            base = sum(len(x) for x in P)
            p = read(pr['attributes']['POSITION']) + t
            P.append(p * [-1, 1, -1]); N.append(read(pr['attributes']['NORMAL']) * [-1, 1, -1])      # parent node turns it half round
            UV.append(read(pr['attributes']['TEXCOORD_0'])); I.append(read(pr['indices']).reshape(-1) + base)
    P, N, UV, I = np.vstack(P), np.vstack(N), np.vstack(UV), np.concatenate(I)
    pole = P[P[:, 1] < P[:, 1].min() + 1.0]
    P = (P - [pole[:, 0].mean(), P[:, 1].min(), pole[:, 2].mean()]) * 0.1                         # stand it on the origin, in metres
    for name, size, q in (('baseColor', 1024, 88), ('normal', 1024, 90), ('emissive', 512, 85)):
        Image.open(os.path.join(d, f'Lantern_{name}.png')).convert('RGB').resize((size, size), Image.LANCZOS).save(
            os.path.join(OUT, 'textures', f'lantern_{name}.jpg'), quality=q)
    write_glb(os.path.join(OUT, 'models', 'lantern.glb'),
              [{'name': 'lantern', 'pos': P, 'nor': N, 'uv': UV, 'idx': I.reshape(-1, 3), 'texture': 0, 'normalTex': 1, 'emissiveTex': 2, 'rough': 0.6}],
              images=['../textures/lantern_baseColor.jpg', '../textures/lantern_normal.jpg', '../textures/lantern_emissive.jpg'],
              extras={'size': [round(float(x), 3) for x in (P.max(0) - P.min(0))], 'min': [round(float(x), 3) for x in P.min(0)], 'max': [round(float(x), 3) for x in P.max(0)]})

if len(sys.argv) > 3: build_lantern(sys.argv[3])
