# Avatar model and character attribution

`face.onnx` and `character.png` are derived from the `lambda_01` student model
and character image in
[pkhungurn/talking-head-anime-4-demo](https://github.com/pkhungurn/talking-head-anime-4-demo),
Copyright (c) 2024 pixiv Inc.

The THA4 models and images are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): **non-commercial
use only**. Commercial use needs permission from the rights holder or a
student model trained on a character you own (see the repo's `distiller_ui`).

`face.onnx` was produced by `tools/export_tha4_face.py` (face morpher only,
with the linear/premultiplied -> sRGB/straight-alpha conversion baked in).
