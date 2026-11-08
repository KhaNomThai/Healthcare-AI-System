// Pure preprocessing math (no DOM / ONNX), so it can be tested in Node.

// Per-output-pixel fixed-point weights of PIL's antialiased bilinear resample (Image.resize BILINEAR).
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
    // PIL rounds normalized weights to fixed-point ints (normalize_coeffs_8bpc)
    out.push({ start: xmin, w: w.map((v) => Math.trunc((sum ? v / sum : 0) * ONE + 0.5)) });
  }
  return out;
}

// PIL's 8-bit resample math: accumulate in fixed point, start at +0.5, then floor (clip8).
const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;
const HALF = 2 ** (PRECISION_BITS - 1);
const clamp8 = (acc) => Math.min(255, Math.max(0, Math.floor(acc / ONE)));

// Must match training preprocessing exactly: torchvision Resize((size, size)) on a PIL image
// (antialiased bilinear, horizontal pass then vertical, uint8 between passes)
// -> ToTensor -> Normalize(mean, std). Browser canvas scaling is NOT equivalent.
// `src` is RGBA bytes (W x H); returns a CHW Float32Array of length 3 * size * size.
export function resizeNormalize(src, W, H, size, mean, std) {

  const wx = pilWeights(W, size);
  const wy = pilWeights(H, size);

  const tmp = new Uint8ClampedArray(size * H * 3); // horizontal pass: H rows x size cols
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < size; x++) {
      const { start, w } = wx[x];
      let r = HALF, g = HALF, b = HALF;
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
      const acc = [HALF, HALF, HALF];
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
  return out;
}

