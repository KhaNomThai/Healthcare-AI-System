// Small helpers and components shared by the screening and benchmark views.
import { useEffect, useState } from "react";

export const pct = (p) => `${(p * 100).toFixed(1)}%`;

// "Melanoma (MEL)" -> "Melanoma"
export const shortName = (name) => name.replace(/\s*\(.*\)$/, "");

export function CodePill({ code, cancer }) {
  return (
    <span
      className={`inline-block min-w-[44px] rounded border px-1.5 py-0.5 text-center text-[11.5px] font-bold ${
        cancer ? "border-red-100 bg-red-50 text-red-700" : "border-green-100 bg-green-50 text-green-700"
      }`}
    >
      {code}
    </span>
  );
}

// false on the first frame, true right after: lets CSS transitions animate from 0 on mount / new key.
export function useGrow(key) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    setOn(false);
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setOn(true)));
    return () => cancelAnimationFrame(id);
  }, [key]);
  return on;
}

// Count a number up from 0 to `target` (snaps when the user prefers reduced motion).
function useCountUp(target, ms = 800) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return setV(target);
    let id;
    const t0 = performance.now();
    const tick = (now) => {
      const k = Math.min((now - t0) / ms, 1);
      setV(target * (1 - (1 - k) ** 3)); // ease-out cubic
      if (k < 1) id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [target, ms]);
  return v;
}

// A percentage that counts up from 0%.
export function CountUp({ value }) {
  return pct(useCountUp(value));
}
