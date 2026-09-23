// Lip-synced talking-head avatar for FaceTime calls, rendered in the browser.
//
// Uses a THA4 student face morpher (github.com/pkhungurn/talking-head-anime-4-demo)
// exported to ONNX (face.onnx: pose[1,39] -> rgba[128,128,4], sRGB straight alpha)
// and run with onnxruntime-web - WebGPU when available, WASM otherwise. Only the
// 128x128 face patch is regenerated per frame and pasted onto the static
// 512x512 character image; head motion is a 2D pivot of the whole image
// around the neck (the model's body morpher is too heavy for phones).
//
// Model + character: CC BY-NC 4.0 (pixiv Inc.) - non-commercial use only.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const BASE = new URL("./", import.meta.url);
const FACE_X = 256 - 64; // where the face patch sits in the 512x512 image
const FACE_Y = 144 - 64;

// indices into the 39-float face pose (tha4/poser/modes/pose_parameters.py)
const P = {
  browRaisedL: 6, browRaisedR: 7,
  eyeWinkL: 12, eyeWinkR: 13,
  eyeHappyL: 14, eyeHappyR: 15,
  eyeSurprisedL: 16, eyeSurprisedR: 17,
  mouthAaa: 26, mouthIii: 27, mouthUuu: 28, mouthEee: 29, mouthOoo: 30,
  cornerRaisedL: 34, cornerRaisedR: 35,
  irisX: 37, irisY: 38,
};
const VOWELS = [P.mouthAaa, P.mouthIii, P.mouthUuu, P.mouthEee, P.mouthOoo];

// head-and-shoulders portrait crop of the 512x512 character, shown full-bleed
const CROP = { x: 116, y: 0, w: 280, h: 440 };
// head motion pivots around the base of the neck (512x512 image coords)
const NECK = { x: 256, y: 235 };

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

const lerp = (a, b, k) => a + (b - a) * k;
const clamp01 = (x) => Math.min(1, Math.max(0, x));

export class TalkingHead {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    // full 512x512 character (face patch pasted in), drawn onto `canvas`
    this.full = document.createElement("canvas");
    this.full.width = this.full.height = 512;
    this.fullCtx = this.full.getContext("2d");
    this.faceCanvas = document.createElement("canvas");
    this.pose = new Float32Array(39);
    this.running = false;
    this.analyser = null;
    this.audioCtx = null;
    this.source = null;

    // speech tracking
    this.env = 0; // smoothed loudness envelope
    this.peak = 0.05; // running loudness peak, for auto-gain
    this.mouthOpen = 0;
    this.vowelTarget = new Float32Array(5);
    this.vowel = new Float32Array(5);
    this.vowelTarget[0] = 1;
    this.inSyllable = false;
    this.lastSpeechAt = -1e9;
    this.speaking = false;

    // expression state (0..1, eased toward targets)
    this.smile = 0.3;
    this.happyEyes = 0;
    this.surprise = 0;
    this.warmUntil = 0; // brief smile + happy eyes after a phrase ends

    // eyes and head
    this.nextBlinkAt = performance.now() + 1500;
    this.blinkStart = -1;
    this.gaze = { x: 0, y: 0 };
    this.gazeTarget = { x: 0, y: 0 };
    this.nextGlanceAt = 0;
    this.nod = 0;
    this.nodVel = 0;
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

  // Drive the face from the remote (bot) audio track. audioCtx must have
  // been created/resumed inside a user gesture (the FaceTime tap) - iOS
  // Safari keeps a context made later suspended, and the analyser would
  // then only ever read silence.
  setAudioTrack(track, audioCtx) {
    this.source?.disconnect();
    this.audioCtx = audioCtx;
    this.source = audioCtx.createMediaStreamSource(new MediaStream([track]));
    this.analyser = audioCtx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.2;
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

  // loudness (rms) and spectral brightness (0 = dark/round, 1 = bright/spread)
  readAudio() {
    if (!this.analyser) return { rms: 0, bright: 0.5 };
    this.analyser.getFloatTimeDomainData(this.wave);
    let sum = 0;
    for (const s of this.wave) sum += s * s;
    const rms = Math.sqrt(sum / this.wave.length);

    this.analyser.getByteFrequencyData(this.freq);
    const hz = this.audioCtx.sampleRate / this.analyser.fftSize;
    const band = (lo, hi) => {
      let e = 0;
      for (let i = Math.floor(lo / hz); i < Math.ceil(hi / hz); i++) e += this.freq[i];
      return e;
    };
    const low = band(250, 900), high = band(1800, 3800);
    return { rms, bright: high / (low + high + 1e-6) };
  }

  // Each syllable onset picks a new vowel shape, biased by the spectrum:
  // bright sounds lean ii/ee, dark ones oo/uu, the rest aa. The randomness
  // keeps consecutive syllables visibly different instead of one flapping jaw.
  pickVowel(bright) {
    // open shapes (aa/oo/ee) are favoured: ii mostly reads as a closed,
    // toothy smile, so too much of it makes speech look mumbled
    const w = [
      1.4,                                   // aa
      Math.max(0.05, (bright - 0.4) * 1.5),  // ii
      Math.max(0.05, (0.35 - bright) * 2),   // uu
      Math.max(0.2, (bright - 0.2) * 2),     // ee
      Math.max(0.3, (0.45 - bright) * 3),    // oo
    ];
    let r = Math.random() * w.reduce((a, b) => a + b, 0);
    let pick = 0;
    for (; pick < 4 && r > w[pick]; pick++) r -= w[pick];
    this.vowelTarget.fill(0);
    this.vowelTarget[pick] = 1;
  }

  update(now, dt) {
    const { rms, bright } = this.readAudio();

    // auto-gain: normalise to the voice's own recent peak so a quiet TTS
    // voice still opens the mouth fully on stressed syllables
    this.peak = Math.max(rms, this.peak * Math.pow(0.5, dt / 1.5), 0.015);
    const level = clamp01((rms / this.peak - 0.06) / 0.45);
    this.env = lerp(this.env, level, level > this.env ? 0.8 : 0.4);

    // syllable onsets: the envelope climbing back up after a dip
    if (!this.inSyllable && this.env > 0.35) {
      this.inSyllable = true;
      this.pickVowel(bright);
      if (level > 0.92) this.emphasis(now);
    } else if (this.inSyllable && this.env < 0.2) {
      this.inSyllable = false;
    }

    const talking = this.env > 0.08;
    if (talking) this.lastSpeechAt = now;
    const wasSpeaking = this.speaking;
    this.speaking = now - this.lastSpeechAt < 350;
    if (wasSpeaking && !this.speaking) {
      // phrase ended: warm smile, happy eyes, sometimes a blink
      this.warmUntil = now + 700 + Math.random() * 500;
      if (Math.random() < 0.5) this.nextBlinkAt = now + 120;
    }

    // mouth: full range, with a floor so open syllables never look mumbled
    const open = this.env > 0.05 ? 0.4 + 0.6 * this.env : 0;
    this.mouthOpen = lerp(this.mouthOpen, open, 0.7);
    for (let i = 0; i < 5; i++) this.vowel[i] = lerp(this.vowel[i], this.vowelTarget[i], 0.6);

    const p = this.pose;
    for (let i = 0; i < 5; i++) p[VOWELS[i]] = this.mouthOpen * this.vowel[i];
    // every shape gets some jaw drop, so ee/ii/uu still read as an open,
    // talking mouth rather than lips barely parting
    p[P.mouthAaa] = this.mouthOpen * (this.vowel[0] + 0.4 * (1 - this.vowel[0]));

    // expressions
    const warm = now < this.warmUntil;
    this.smile = lerp(this.smile, warm ? 0.9 : this.speaking ? 0.15 : 0.35, 0.08);
    this.happyEyes = lerp(this.happyEyes, warm ? 0.55 : 0, 0.12);
    this.surprise = lerp(this.surprise, 0, 0.06);
    p[P.cornerRaisedL] = p[P.cornerRaisedR] = this.smile * (1 - this.mouthOpen * 0.6);
    p[P.eyeHappyL] = p[P.eyeHappyR] = this.happyEyes;
    p[P.eyeSurprisedL] = p[P.eyeSurprisedR] = this.surprise * 0.6;
    p[P.browRaisedL] = p[P.browRaisedR] = this.surprise;

    this.updateEyes(now);
    this.updateHead(now, dt);
  }

  emphasis(now) {
    this.surprise = Math.max(this.surprise, 0.6 + Math.random() * 0.4);
    this.nodVel += 5 + Math.random() * 4; // quick downward nod
  }

  updateEyes(now) {
    if (this.blinkStart < 0 && now >= this.nextBlinkAt) this.blinkStart = now;
    let blink = 0;
    if (this.blinkStart >= 0) {
      const t = (now - this.blinkStart) / 150;
      blink = t < 0.5 ? t * 2 : Math.max(0, 2 - t * 2);
      if (t >= 1) {
        this.blinkStart = -1;
        this.nextBlinkAt = now + 1800 + Math.random() * 3500;
      }
    }
    // happy eyes already close the lids; don't stack a full blink on top
    const b = blink * (1 - this.happyEyes);
    this.pose[P.eyeWinkL] = this.pose[P.eyeWinkR] = b;

    // quick glances (saccades) rather than a slow drift; mostly toward the
    // viewer while talking, wandering a little more while listening
    if (now >= this.nextGlanceAt) {
      const spread = this.speaking ? 0.18 : 0.35;
      this.gazeTarget.x = (Math.random() * 2 - 1) * spread;
      this.gazeTarget.y = (Math.random() * 2 - 1) * spread * 0.5;
      if (Math.random() < 0.45) this.gazeTarget.x = this.gazeTarget.y = 0;
      this.nextGlanceAt = now + 700 + Math.random() * 2200;
    }
    this.gaze.x = lerp(this.gaze.x, this.gazeTarget.x, 0.35);
    this.gaze.y = lerp(this.gaze.y, this.gazeTarget.y, 0.35);
    this.pose[P.irisX] = this.gaze.x;
    this.pose[P.irisY] = this.gaze.y;
  }

  // head sway/tilt driven by speech energy plus slow drift, and a springy
  // nod kicked by emphasised syllables
  updateHead(now, dt) {
    const k = 60, damping = 9;
    this.nodVel += (-k * this.nod - damping * this.nodVel) * dt;
    this.nod += this.nodVel * dt;
    const energy = this.speaking ? 0.4 + this.env * 0.6 : 0.25;
    this.headAngle =
      (Math.sin(now / 1700) * 1.6 + Math.sin(now / 610) * 0.9 * energy) * (Math.PI / 180);
    this.headDy = this.nod * 3 + Math.sin(now / 480) * 1.2 * energy * (this.speaking ? 1 : 0);
    this.breath = 1 + Math.sin(now / 700) * 0.006;
  }

  async renderFace() {
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
  }

  draw() {
    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, CROP.w, CROP.h);
    // pivot at the neck, drawing the whole 512x512 image so rotating never
    // reveals an edge inside the crop
    c.translate(NECK.x - CROP.x, NECK.y - CROP.y + this.headDy);
    c.rotate(this.headAngle);
    c.scale(this.breath, this.breath);
    c.drawImage(this.full, -NECK.x, -NECK.y);
  }

  async loop() {
    let last = performance.now();
    while (this.running) {
      const now = performance.now();
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      this.update(now, dt);
      await this.renderFace();
      if (!this.running) break;
      this.draw();
      await new Promise((r) => requestAnimationFrame(r));
    }
  }
}
