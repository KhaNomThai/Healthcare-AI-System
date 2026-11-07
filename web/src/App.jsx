import { useEffect, useState } from "react";
import Screening from "./Screening.jsx";
import Benchmark from "./Benchmark.jsx";

const GITHUB_URL = "https://github.com/KhaNomThai/Healthcare-AI-System";

const TABS = [
  { id: "screening", label: "Clinical Screening" },
  { id: "benchmark", label: "Model Performance" },
];

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("screening");

  useEffect(() => {
    fetch("./data.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 pb-10 pt-6 sm:px-6">
      <Header />
      <Notice />

      <nav className="mb-5 inline-flex gap-1 rounded-xl border border-slate-200 bg-slate-200/70 p-1" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-lg px-4 py-2 text-[13px] font-semibold transition sm:px-5 ${
              tab === t.id ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:bg-white/50 hover:text-slate-900"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {error && <div className="card p-6 text-sm text-red-700">Could not load app data: {error}</div>}
      {!data && !error && <div className="card p-6 text-sm text-slate-500">Loading…</div>}
      {data && (tab === "screening" ? <Screening data={data} /> : <Benchmark data={data} />)}

      <Footer />
    </div>
  );
}

function Header() {
  return (
    <header className="mb-4 flex flex-wrap items-center justify-between gap-4 rounded-2xl bg-slate-900 px-5 py-5 text-white shadow-lg sm:px-7">
      <div className="flex items-center gap-4">
        <img src="./logo.svg" alt="" className="h-12 w-12 shrink-0 rounded-xl sm:h-14 sm:w-14" />
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-xl font-extrabold tracking-tight sm:text-2xl">Derma AI</h1>
            <span className="rounded-full border border-blue-400/40 bg-blue-500/20 px-2.5 py-0.5 text-[10.5px] font-semibold uppercase tracking-wider text-blue-200">
              Clinical Decision Support
            </span>
          </div>
          <p className="mt-1 text-[13px] text-slate-300">
            Skin Cancer Multi-Class Classification &amp; Two-Level Screening System
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <Pill label="Dataset" value="ISIC 2019 (25,331)" />
        <Pill label="Taxonomy" value="8 Clinical Classes" />
        <Pill label="Inference" value="100% in-browser" />
      </div>
    </header>
  );
}

function Pill({ label, value }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5">
      <div className="text-[9.5px] font-semibold uppercase tracking-wider text-slate-400">{label}</div>
      <div className="text-[12.5px] font-semibold text-white">{value}</div>
    </div>
  );
}

function Notice() {
  return (
    <div className="mb-5 flex gap-3 rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-[12.5px] leading-relaxed text-slate-700">
      <svg viewBox="0 0 24 24" className="mt-0.5 h-[18px] w-[18px] shrink-0" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
      <p>
        <strong className="text-slate-900">Clinical Notice:</strong> An AI-assisted screening prototype built for
        academic research (CPE310 Healthcare AI System). Outputs are probabilistic estimates and must not replace
        professional dermatological consultation or biopsy. Images are processed on your device and never uploaded.
      </p>
    </div>
  );
}

function Footer() {
  return (
    <footer className="mt-10 border-t border-slate-200 pt-5 text-center text-[11.5px] leading-relaxed text-slate-500">
      <p>
        PyTorch models (ResNet50 · EfficientNet-B3 · DenseNet121) exported to ONNX and run with ONNX Runtime Web.
      </p>
      <p className="mt-1">
        Sample images: ISIC 2019 Challenge dataset (CC BY-NC 4.0). ·{" "}
        <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="font-semibold text-blue-600 hover:underline">
          Source code on GitHub
        </a>
      </p>
    </footer>
  );
}
