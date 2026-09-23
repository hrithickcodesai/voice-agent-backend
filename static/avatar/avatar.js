// Lip-synced talking-head avatar for FaceTime calls, rendered in the browser.
//
// Uses a THA4 student face morpher (github.com/pkhungurn/talking-head-anime-4-demo)
// exported to ONNX (face.onnx: pose[1,39] -> rgba[128,128,4], sRGB straight alpha)
// and run with onnxruntime-web - WebGPU when available, WASM otherwise. Only the
// 128x128 face patch is regenerated per frame and pasted onto the static
// 512x512 character image; the body stays put, which is all lip sync needs.
//
// Model + character: CC BY-NC 4.0 (pixiv Inc.) - non-commercial use only.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const BASE = new URL("./", import.meta.url);
const FACE_X = 256 - 64; // where the face patch sits in the 512x512 image
const FACE_Y = 144 - 64;

// indices into the 39-float face pose (tha4/poser/modes/pose_parameters.py)
const P = {
  eyeWinkL: 12, eyeWinkR: 13,
  mouthAaa: 26, mouthIii: 27, mouthUuu: 28, mouthEee: 29, mouthOoo: 30,
  cornerRaisedL: 34, cornerRaisedR: 35,
  irisX: 37, irisY: 38,
};

// head-and-shoulders portrait crop of the 512x512 character, shown full-bleed
const CROP = { x: 116, y: 0, w: 280, h: 440 };

let sessionPromise = null;
let baseImagePromise = null;

async function loadModel(file, providers) {
  const model = await (await fetch(new URL(file, BASE))).arrayBuffer();
  return ort.InferenceSession.create(model, { executionProviders: providers });
}

// Safe to call early (e.g. when the call starts ringing) so the ~6MB runtime
// and model are ready by the time the tutor picks up. WebGPU renders the full
// 128x128 face at ~50fps; without it, single-threaded WASM manages ~5fps at
// that size, so it gets the 64x64 export (4x cheaper) upscaled instead.
export function preloadAvatar() {
  sessionPromise ??= (async () => {
    if (navigator.gpu) {
      try {
        return { session: await loadModel("face.onnx", ["webgpu"]), size: 128 };
      } catch (e) {
        console.warn("webgpu unavailable for avatar, falling back to wasm", e);
      }
    }
    return { session: await loadModel("face64.onnx", ["wasm"]), size: 64 };
  })();
  baseImagePromise ??= Promise.all([loadImage("character.png"), loadImage("face_mask.png")]);
  return Promise.all([sessionPromise, baseImagePromise]);
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = new URL(file, BASE).href;
  });
}

export class TalkingHead {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    // full 512x512 character (face patch pasted in), cropped onto `canvas`
    this.full = document.createElement("canvas");
    this.full.width = this.full.height = 512;
    this.fullCtx = this.full.getContext("2d");
    this.faceCanvas = document.createElement("canvas");
    this.pose = new Float32Array(39);
    this.lastPose = new Float32Array(39).fill(NaN);
    this.running = false;
    this.analyser = null;
    this.audioCtx = null;
    this.source = null;
    this.mouthOpen = 0;
    this.nextBlinkAt = performance.now() + 1500;
    this.blinkStart = -1;
  }

  async start() {
    const [{ session, size }, [baseImage, maskImage]] = await preloadAvatar();
    this.session = session;
    this.baseImage = baseImage;
    this.faceCanvas.width = this.faceCanvas.height = size;
    this.faceCtx = this.faceCanvas.getContext("2d");
    this.faceData = this.faceCtx.createImageData(size, size);
    this.fullCtx.drawImage(baseImage, 0, 0, 512, 512);
    // a low-res face is only blended in through the feathered eyes+mouth
    // mask, so the rest of the face keeps the full-res original instead of
    // showing a blurry square
    if (size < 128) {
      this.maskImage = maskImage;
      this.maskedCanvas = document.createElement("canvas");
      this.maskedCanvas.width = this.maskedCanvas.height = 128;
      this.maskedCtx = this.maskedCanvas.getContext("2d");
    }
    this.canvas.width = CROP.w;
    this.canvas.height = CROP.h;
    this.running = true;
    this.loop();
  }

  // Drive the mouth from the remote (bot) audio track. audioCtx must have
  // been created/resumed inside a user gesture (the FaceTime tap) - iOS
  // Safari keeps a context made later suspended, and the analyser would
  // then only ever read silence.
  setAudioTrack(track, audioCtx) {
    this.source?.disconnect();
    this.audioCtx = audioCtx;
    this.source = audioCtx.createMediaStreamSource(new MediaStream([track]));
    this.analyser = audioCtx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.3;
    this.source.connect(this.analyser);
    this.freq = new Uint8Array(this.analyser.frequencyBinCount);
    this.wave = new Float32Array(this.analyser.fftSize);
  }

  stop() {
    this.running = false;
    this.source?.disconnect();
    this.source = null;
    this.analyser = null;
  }

  // Loudness -> how open the mouth is; rough spectral balance -> vowel shape
  // (bright "ee/ii" vs round "oo/uu" vs open "aa").
  updateMouth() {
    let open = 0, bright = 0.5;
    if (this.analyser) {
      this.analyser.getFloatTimeDomainData(this.wave);
      let sum = 0;
      for (const s of this.wave) sum += s * s;
      const rms = Math.sqrt(sum / this.wave.length);
      open = Math.min(1, Math.max(0, (rms - 0.01) * 9));

      this.analyser.getByteFrequencyData(this.freq);
      const hz = this.audioCtx.sampleRate / this.analyser.fftSize;
      const band = (lo, hi) => {
        let e = 0;
        for (let i = Math.floor(lo / hz); i < Math.ceil(hi / hz); i++) e += this.freq[i];
        return e;
      };
      const low = band(250, 900), high = band(1800, 3500);
      bright = high / (low + high + 1e-6);
    }
    // fast attack, slower release so the mouth doesn't flicker between syllables
    const k = open > this.mouthOpen ? 0.6 : 0.25;
    this.mouthOpen += (open - this.mouthOpen) * k;
    const m = this.mouthOpen;

    const p = this.pose;
    const ee = Math.max(0, (bright - 0.45) * 2.5);
    const oo = Math.max(0, (0.3 - bright) * 3);
    p[P.mouthIii] = m * Math.min(1, ee) * 0.6;
    p[P.mouthOoo] = m * Math.min(1, oo) * 0.7;
    p[P.mouthAaa] = m * (1 - Math.min(1, ee + oo) * 0.5);
    p[P.mouthEee] = 0;
    p[P.mouthUuu] = 0;
    // soft resting smile, fading while the mouth is open
    p[P.cornerRaisedL] = p[P.cornerRaisedR] = 0.35 * (1 - m);
  }

  updateEyes(now) {
    if (this.blinkStart < 0 && now >= this.nextBlinkAt) this.blinkStart = now;
    let blink = 0;
    if (this.blinkStart >= 0) {
      const t = (now - this.blinkStart) / 160; // ~160ms close+open
      blink = t < 0.5 ? t * 2 : Math.max(0, 2 - t * 2);
      if (t >= 1) {
        this.blinkStart = -1;
        this.nextBlinkAt = now + 2000 + Math.random() * 4000;
      }
    }
    this.pose[P.eyeWinkL] = this.pose[P.eyeWinkR] = blink;
    // slow, small gaze drift so the idle face feels alive
    this.pose[P.irisX] = Math.sin(now / 2300) * 0.12;
    this.pose[P.irisY] = Math.sin(now / 3100) * 0.08;
  }

  async loop() {
    while (this.running) {
      const now = performance.now();
      this.updateMouth();
      this.updateEyes(now);

      let changed = false;
      for (let i = 0; i < 39; i++) {
        if (Math.abs(this.pose[i] - this.lastPose[i]) > 0.01 || Number.isNaN(this.lastPose[i])) {
          changed = true;
          break;
        }
      }
      if (changed) {
        this.lastPose.set(this.pose);
        const out = await this.session.run({ pose: new ort.Tensor("float32", this.pose.slice(), [1, 39]) });
        const rgba = out.rgba.data;
        const px = this.faceData.data;
        for (let i = 0; i < px.length; i++) px[i] = rgba[i] * 255;
        out.rgba.dispose?.();
        this.faceCtx.putImageData(this.faceData, 0, 0);
        // the patch replaces the face region (incl. transparent pixels), so
        // clear before drawing rather than compositing over the old frame
        this.fullCtx.clearRect(FACE_X, FACE_Y, 128, 128);
        if (this.maskedCtx) {
          const m = this.maskedCtx;
          m.globalCompositeOperation = "copy";
          m.drawImage(this.faceCanvas, 0, 0, 128, 128);
          m.globalCompositeOperation = "destination-in";
          m.drawImage(this.maskImage, 0, 0);
          this.fullCtx.drawImage(this.baseImage, FACE_X, FACE_Y, 128, 128, FACE_X, FACE_Y, 128, 128);
          this.fullCtx.drawImage(this.maskedCanvas, FACE_X, FACE_Y);
        } else {
          this.fullCtx.drawImage(this.faceCanvas, FACE_X, FACE_Y, 128, 128);
        }
        this.ctx.clearRect(0, 0, CROP.w, CROP.h);
        this.ctx.drawImage(this.full, CROP.x, CROP.y, CROP.w, CROP.h, 0, 0, CROP.w, CROP.h);
      }
      await new Promise((r) => requestAnimationFrame(r));
    }
  }
}
