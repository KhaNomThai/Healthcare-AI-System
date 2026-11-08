import { useEffect, useState } from "react";
import Screening from "./Screening.jsx";
import Benchmark from "./Benchmark.jsx";

const GITHUB_URL = "https://github.com/KhaNomThai/Healthcare-AI-System";

const TABS = [
  { id: "screening", label: "Clinical Screening", short: "Screening", icon: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM21 21l-4.35-4.35" },
  { id: "benchmark", label: "Model Performance", short: "Performance", icon: "M4 20V10M10 20V4M16 20v-7M22 20H2" },
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

  const tabIdx = TABS.findIndex((t) => t.id === tab);

  // WAI-ARIA tabs: arrow keys move between tabs.
  const onTabKey = (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!step) return;
    const next = TABS[(tabIdx + step + TABS.length) % TABS.length];
    setTab(next.id);
    document.getElementById(`tab-${next.id}`)?.focus();
  };

  return (
    <div className="mx-auto max-w-6xl px-4 pb-10 pt-6 sm:px-6">
      <Header classCount={data?.classes.length} />
      <Notice />
      {data?.models.some((m) => m.metrics && !m.metrics.split) && <LegacyNotice />}

      <nav className="relative mb-5 grid w-full grid-cols-2 rounded-xl border border-slate-200 bg-slate-200/70 p-1 sm:inline-grid sm:w-auto" role="tablist" aria-label="Sections" onKeyDown={onTabKey}>
        {/* Sliding pill behind the active tab */}
        <span
          aria-hidden="true"
          className="absolute inset-y-1 left-1 w-[calc(50%-0.25rem)] rounded-lg bg-white shadow-sm transition-transform duration-300 ease-out"
          style={{ transform: `translateX(${tabIdx * 100}%)` }}
        />
        {TABS.map((t) => (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            role="tab"
            aria-selected={tab === t.id}
            aria-controls="tabpanel"
            tabIndex={tab === t.id ? 0 : -1}
            onClick={() => setTab(t.id)}
            className={`relative flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-[14px] font-semibold transition-colors sm:px-5 ${
              tab === t.id ? "text-blue-700" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d={t.icon} />
            </svg>
            <span className="sm:hidden">{t.short}</span>
            <span className="hidden sm:inline">{t.label}</span>
          </button>
        ))}
      </nav>

      <div id="tabpanel" role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {error && <div className="card p-6 text-sm text-red-700">Could not load app data: {error}</div>}
        {!data && !error && <div className="card animate-pulse p-6 text-sm text-slate-500">Loading…</div>}
        {data && (
          <div key={tab} className="animate-rise">
            {tab === "screening" ? <Screening data={data} /> : <Benchmark data={data} />}
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
}

function Header({ classCount }) {
  return (
    <header className="relative mb-4 flex flex-wrap items-center justify-between gap-4 overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-blue-950 px-5 py-5 text-white shadow-lg sm:px-7">
      {/* Decorative glow blobs + dot grid */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute -left-10 -top-16 h-48 w-48 animate-float rounded-full bg-blue-500/25 blur-3xl" />
        <div className="absolute -bottom-20 right-10 h-52 w-52 animate-float rounded-full bg-teal-400/15 blur-3xl [animation-delay:-3s]" />
        <div className="absolute inset-0 bg-[radial-gradient(rgb(255_255_255/0.07)_1px,transparent_1px)] [background-size:16px_16px] [mask-image:linear-gradient(to_left,black,transparent_70%)]" />
      </div>
      <div className="relative flex items-center gap-4">
        <img src="./logo.svg" alt="" className="h-12 w-12 shrink-0 rounded-xl shadow-lg shadow-blue-500/20 ring-1 ring-white/10 sm:h-14 sm:w-14" />
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="bg-gradient-to-r from-white to-blue-200 bg-clip-text text-xl font-extrabold tracking-tight text-transparent sm:text-2xl">
              Derma AI
            </h1>
            <span className="rounded-full border border-blue-400/40 bg-blue-500/20 px-2.5 py-0.5 text-[11.5px] font-semibold uppercase tracking-wider text-blue-200">
              Research Prototype
            </span>
          </div>
          <p className="mt-1 text-[14px] text-slate-300">
            Skin Cancer Multi-Class Classification &amp; Two-Level Screening System
          </p>
        </div>
      </div>
      <div className="relative flex flex-wrap gap-2">
        <Pill label="Dataset" value="ISIC 2019 (25,331)" />
        <Pill label="Taxonomy" value={classCount ? `${classCount} Clinical Classes` : "…"} />
        <Pill label="Inference" value="100% in-browser" live />
      </div>
    </header>
  );
}

function Pill({ label, value, live }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-sm transition hover:border-white/25 hover:bg-white/10">
      <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-400">{label}</div>
      <div className="flex items-center gap-1.5 text-[13.5px] font-semibold text-white">
        {live && (
          <span className="relative flex h-2 w-2" aria-hidden="true">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
          </span>
        )}
        {value}
      </div>
    </div>
  );
}

function Notice() {
  return (
    <div className="mb-5 flex gap-3 rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-[13.5px] leading-relaxed text-slate-700">
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

// Shown until the models are retrained: metrics without `split` come from the old image-level split.
function LegacyNotice() {
  return (
    <div role="note" className="mb-5 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-[13.5px] leading-relaxed text-amber-900">
      <p>
        <strong>Preliminary models:</strong> these models were trained before a data-split fix, so images of the same
        lesion appeared in both training and test data. The reported metrics are optimistic, and most sample images
        were seen during training. Retrained models are coming.
      </p>
      <p lang="th" className="mt-1 text-amber-800">
        โมเดลชุดนี้เทรนก่อนแก้การแบ่งข้อมูล รอยโรคเดียวกันจึงอยู่ทั้งในชุดเทรนและชุดทดสอบ ตัวเลขประสิทธิภาพจึงสูงเกินจริง
        และภาพตัวอย่างส่วนใหญ่โมเดลเคยเห็นตอนเทรนแล้ว กำลังเทรนโมเดลใหม่
      </p>
    </div>
  );
}

function Footer() {
  return (
    <footer className="mt-10 border-t border-slate-200 pt-5 text-center text-[12.5px] leading-relaxed text-slate-500">
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
