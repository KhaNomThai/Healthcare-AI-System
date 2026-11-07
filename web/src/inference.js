// Runs the exported ONNX models fully in the browser (no server).
import * as ort from "onnxruntime-web/wasm";

const sessions = new Map(); // model file -> Promise<InferenceSession>

// Download with progress, then build an inference session (cached per model).
export function loadModel(file, onProgress) {
  if (!sessions.has(file)) {
    const p = (async () => {
      const res = await fetch(file);
      if (!res.ok) throw new Error(`Model download failed (${res.status})`);
      const total = Number(res.headers.get("content-length")) || 0;
      const reader = res.body.getReader();
      const chunks = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        if (total) onProgress?.(loaded / total);
      }
      const bytes = new Uint8Array(loaded);
      let offset = 0;
      for (const c of chunks) {
        bytes.set(c, offset);
        offset += c.length;
      }
      return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    })();
    p.catch(() => sessions.delete(file)); // allow retry after a failed download
    sessions.set(file, p);
  }
  return sessions.get(file);
}

// Per-output-pixel weights of PIL's antialiased bilinear resample (Image.resize BILINEAR).
function pilWeights(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale; // bilinear filter support = 1.0
  const out = [];
  for (let x = 0; x < outSize; x++) {
    const center = (x + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
    const w = [];
    let sum = 0;
    for (let i = xmin; i < xmax; i++) {
      const v = Math.max(0, 1 - Math.abs((i - center + 0.5) / filterScale));
      w.push(v);
      sum += v;
    }
    out.push({ start: xmin, w: w.map((v) => (sum ? v / sum : 0)) });
  }
  return out;
}

const clamp8 = (v) => Math.min(255, Math.max(0, Math.round(v)));

// Must match training preprocessing exactly: torchvision Resize((size, size)) on a PIL image
// (antialiased bilinear, horizontal pass then vertical, uint8 between passes)
// -> ToTensor -> Normalize(mean, std). Browser canvas scaling is NOT equivalent.
export function preprocess(img, size, mean, std) {
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const src = ctx.getImageData(0, 0, W, H).data;

  const wx = pilWeights(W, size);
  const wy = pilWeights(H, size);

  const tmp = new Uint8ClampedArray(size * H * 3); // horizontal pass: H rows x size cols
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < size; x++) {
      const { start, w } = wx[x];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < w.length; k++) {
        const s = (y * W + start + k) * 4;
        r += src[s] * w[k];
        g += src[s + 1] * w[k];
        b += src[s + 2] * w[k];
      }
      const t = (y * size + x) * 3;
      tmp[t] = clamp8(r);
      tmp[t + 1] = clamp8(g);
      tmp[t + 2] = clamp8(b);
    }
  }

  const area = size * size;
  const out = new Float32Array(3 * area);
  for (let y = 0; y < size; y++) {
    const { start, w } = wy[y];
    for (let x = 0; x < size; x++) {
      const acc = [0, 0, 0];
      for (let k = 0; k < w.length; k++) {
        const t = ((start + k) * size + x) * 3;
        acc[0] += tmp[t] * w[k];
        acc[1] += tmp[t + 1] * w[k];
        acc[2] += tmp[t + 2] * w[k];
      }
      const i = y * size + x;
      for (let c = 0; c < 3; c++) out[c * area + i] = (clamp8(acc[c]) / 255 - mean[c]) / std[c];
    }
  }
  return new ort.Tensor("float32", out, [1, 3, size, size]);
}

export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export async function predict(session, tensor) {
  const out = await session.run({ [session.inputNames[0]]: tensor });
  return softmax(out[session.outputNames[0]].data);
}
