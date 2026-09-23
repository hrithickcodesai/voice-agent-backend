"""Export a THA4 student face morpher to ONNX with display conversion baked in.

Input:  pose  float32[1, 39]   (THA4 face pose params, see pose_parameters.py)
Output: rgba  float32[128, 128, 4]  sRGB, straight alpha, 0..1 (HWC for ImageData)
        (face64.onnx: same at 64x64, the cheaper wasm fallback)

Usage (from a scratch dir, needs torch, onnx, onnxruntime, omegaconf, pillow):
    git clone https://github.com/pkhungurn/talking-head-anime-4-demo tha4
    python export_tha4_face.py lambda_00
    cp out/lambda_00/{face.onnx,face64.onnx,face_mask.png,character.png} <repo>/static/avatar/

Also writes out/<char>/test_*.png (neutral/aaa/ooo/blink) for a visual check
and prints the onnx-vs-torch max error.
"""

import sys
import time

import numpy as np
import onnxruntime as ort
import PIL.Image
import torch
from torch.nn.functional import affine_grid

sys.path.insert(0, "tha4/src")
from tha4.poser.modes.mode_14 import load_face_morpher  # noqa: E402

CHAR = sys.argv[1] if len(sys.argv) > 1 else "lambda_00"
SRC = f"tha4/data/character_models/{CHAR}"
OUT = f"out/{CHAR}"


class FaceForBrowser(torch.nn.Module):
    # the face morpher is a siren (a per-pixel mlp over coordinates), so it can
    # be sampled on a coarser grid: size=64 is 4x cheaper, for the wasm fallback.
    def __init__(self, fm, size=128):
        super().__init__()
        self.fm = fm
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0)
        pos = affine_grid(identity, [1, 1, size, size], align_corners=False).view(1, size * size, 2)
        self.register_buffer("position", torch.transpose(pos, 1, 2).reshape(1, 2, size, size))

    def forward(self, pose):
        x = self.fm(pose, self.position)  # [1,4,size,size], linear premultiplied, -1..1
        v = (x + 1.0) * 0.5
        a = torch.clamp(v[:, 3:4], 0.0, 1.0)
        rgb = torch.where(a > 1e-5, v[:, 0:3] / torch.clamp(a, min=1e-5), torch.zeros_like(v[:, 0:3]))
        rgb = torch.clamp(rgb, 0.0, 1.0)
        srgb = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * torch.pow(torch.clamp(rgb, min=1e-8), 1.0 / 2.4) - 0.055)
        return torch.cat([srgb, a], dim=1)[0].permute(1, 2, 0)  # [128,128,4]


def main():
    import os

    os.makedirs(OUT, exist_ok=True)
    fm = load_face_morpher(f"{SRC}/face_morpher.pt").eval()
    model = FaceForBrowser(fm).eval()

    pose = torch.zeros(1, 39)
    torch.onnx.export(model, (pose,), f"{OUT}/face.onnx", input_names=["pose"],
                      output_names=["rgba"], opset_version=17, dynamo=False)
    print("onnx bytes:", os.path.getsize(f"{OUT}/face.onnx"))
    torch.onnx.export(FaceForBrowser(fm, size=64).eval(), (pose,), f"{OUT}/face64.onnx",
                      input_names=["pose"], output_names=["rgba"], opset_version=17, dynamo=False)

    sess = ort.InferenceSession(f"{OUT}/face.onnx")
    base = PIL.Image.open(f"{SRC}/character.png").convert("RGBA")
    base.save(f"{OUT}/character.png")

    # feathered eyes+mouth mask over the 128x128 face patch: the low-res
    # (face64) path blends the morph only where the face actually moves and
    # keeps the full-res original everywhere else.
    from PIL import ImageFilter

    mask = PIL.Image.open(f"tha4/data/images/{CHAR}_face_mask.png").convert("L")
    mask = mask.crop((256 - 64, 144 - 64, 256 + 64, 144 + 64)).filter(ImageFilter.GaussianBlur(3))
    white = PIL.Image.new("RGBA", (128, 128), (255, 255, 255, 0))
    white.putalpha(mask)
    white.save(f"{OUT}/face_mask.png")

    # mouth_aaa index: 12 eyebrow + 12 eye + 2 iris_small = 26; blink (eye_wink L/R) = 12,13
    tests = {"neutral": {}, "aaa": {26: 1.0}, "ooo": {30: 1.0}, "blink": {12: 1.0, 13: 1.0}}
    for name, vals in tests.items():
        p = torch.zeros(1, 39)
        for i, v in vals.items():
            p[0, i] = v
        with torch.no_grad():
            ref = model(p).numpy()
        got = sess.run(None, {"pose": p.numpy()})[0]
        img = base.copy()
        img.paste(PIL.Image.fromarray((got * 255).round().astype(np.uint8), "RGBA"), (256 - 64, 144 - 64))
        img.save(f"{OUT}/test_{name}.png")
        print(f"{name}: max |onnx - torch| = {np.abs(ref - got).max():.2e}")

    p = np.zeros((1, 39), np.float32)
    t = time.perf_counter()
    for _ in range(20):
        sess.run(None, {"pose": p})
    print(f"onnxruntime cpu: {(time.perf_counter() - t) / 20 * 1000:.1f} ms/frame")


if __name__ == "__main__":
    main()
