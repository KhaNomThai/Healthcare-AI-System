// Browser preprocessing must equal torchvision's (fixtures from make_fixtures.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resizeNormalize } from "../src/resize.js";

const fx = JSON.parse(readFileSync(new URL("./resize_fixtures.json", import.meta.url)));

for (const c of fx.cases) {
  test(`resizeNormalize ${c.w}x${c.h} -> ${fx.size} matches torchvision`, () => {
    const got = resizeNormalize(Uint8ClampedArray.from(c.rgba), c.w, c.h, fx.size, fx.mean, fx.std);
    assert.equal(got.length, c.expected.length);
    let maxDiff = 0;
    for (let i = 0; i < got.length; i++) maxDiff = Math.max(maxDiff, Math.abs(got[i] - c.expected[i]));
    assert.ok(maxDiff < 1e-4, `max diff ${maxDiff}`);
  });
}
