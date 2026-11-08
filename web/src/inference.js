// Runs the exported ONNX models fully in the browser (no server).
import * as ort from "onnxruntime-web/wasm";
import { resizeNormalize } from "./resize.js";

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

// Decode pixels with a canvas, then resize + normalize exactly like training (see resize.js).
export function preprocess(img, size, mean, std) {
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const src = ctx.getImageData(0, 0, W, H).data;
  return new ort.Tensor("float32", resizeNormalize(src, W, H, size, mean, std), [1, 3, size, size]);
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
