# Asset credits

Everything in this folder is free to use and redistribute.

| Files | Source | License |
| --- | --- | --- |
| `models/head_a.glb`, `models/head_b.glb`, `models/hand_open.glb`, `models/hand_grip.glb`, `textures/eye_*.jpg` | Built by `tools/build_models.py` from the MakeHuman base mesh, morph targets, skeleton and eye assets (https://github.com/makehumancommunity/makehuman) | CC0 1.0 |
| `models/lantern.glb`, `textures/lantern_*.jpg` | "Lantern" from the Khronos glTF Sample Assets (https://github.com/KhronosGroup/glTF-Sample-Assets), repacked with smaller textures | CC0 1.0 |
| `textures/ground_*`, `wood_*`, `cloth_*`, `leather_*` | Synthesised by `tools/build_textures.py` (no source images) | CC0 1.0 |

To upgrade a surface, replace a texture with a photo texture of the same file name.
The game loads these files over http(s); opened straight from disk it falls back to
its built-in models and procedural surfaces.
