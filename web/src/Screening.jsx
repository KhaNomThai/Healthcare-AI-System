import { useEffect, useRef, useState } from "react";
import { loadModel, predict, preprocess } from "./inference.js";
import Result from "./Result.jsx";

export default function Screening({ data }) {
  const [modelIdx, setModelIdx] = useState(0);
  const [image, setImage] = useState(null); // { url, label? }
  const [status, setStatus] = useState({ phase: "idle" }); // idle | loading | analyzing | done | error
  const [probs, setProbs] = useState(null);
  const [resultModel, setResultModel] = useState(null);
  const model = data.models[modelIdx];

  // Re-run whenever the image or the selected model changes.
  useEffect(() => {
    if (!image) return;
    let cancelled = false;
    (async () => {
      try {
        setStatus({ phase: "loading", progress: 0 });
        const session = await loadModel(model.file, (progress) => {
          if (!cancelled) setStatus({ phase: "loading", progress });
        });
        if (cancelled) return;
        setStatus({ phase: "analyzing" });
        const img = new Image();
        img.src = image.url;
        await img.decode();
        const p = await predict(session, preprocess(img, data.imageSize, data.mean, data.std));
        if (cancelled) return;
        setProbs(p);
        setResultModel(model);
        setStatus({ phase: "done" });
      } catch (e) {
        if (!cancelled) setStatus({ phase: "error", message: e.message || String(e) });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [image, model, data]);

  const busy = status.phase === "loading" || status.phase === "analyzing";
  const resultRef = useRef(null);

  // On narrow screens the result sits below the inputs, so bring it into view.
  const chooseImage = (img) => {
    setImage(img);
    if (window.matchMedia("(max-width: 1023px)").matches) {
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  };

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <div className="space-y-4 lg:col-span-5">
        <ModelPicker models={data.models} value={modelIdx} onChange={setModelIdx} disabled={busy} />
        <ImageInput image={image} onImage={chooseImage} disabled={busy} />
        <Examples examples={data.examples} active={image?.url} onPick={chooseImage} disabled={busy} />
        <Taxonomy classes={data.classes} />
      </div>

      <div ref={resultRef} className="scroll-mt-4 lg:col-span-7">
        {status.phase === "error" ? (
          <div className="card border-red-200 p-6 text-sm text-red-700">Analysis error: {status.message}</div>
        ) : busy ? (
          <Progress status={status} model={model} />
        ) : probs ? (
          <Result probs={probs} classes={data.classes} model={resultModel} groundTruth={image?.label} />
        ) : (
          <Placeholder />
        )}
      </div>
    </div>
  );
}

function ModelPicker({ models, value, onChange, disabled }) {
  return (
    <section className="card p-4">
      <div className="eyebrow mb-1">Neural Network Architecture</div>
      <p className="mb-3 text-[12px] text-slate-500">Select the backbone model for dermoscopic inference.</p>
      <div className="grid grid-cols-3 gap-2">
        {models.map((m, i) => {
          const active = i === value;
          return (
            <button
              key={m.name}
              disabled={disabled}
              onClick={() => onChange(i)}
              aria-pressed={active}
              className={`rounded-xl border p-2.5 text-left transition disabled:cursor-not-allowed disabled:opacity-60 ${
                active
                  ? "border-blue-600 bg-blue-50 ring-2 ring-blue-600/20"
                  : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
              }`}
            >
              <div className={`text-[12.5px] font-bold leading-tight ${active ? "text-blue-700" : "text-slate-900"}`}>
                {m.name}
              </div>
              {m.metrics && (
                <div className="mt-1 text-[11px] text-slate-500">
                  Top-1 <span className="font-semibold text-slate-700">{(m.metrics.overall_accuracy * 100).toFixed(1)}%</span>
                </div>
              )}
              <div className="text-[10.5px] text-slate-400">{m.sizeMB} MB</div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ImageInput({ image, onImage, disabled }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);

  const pickFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    if (image?.url?.startsWith("blob:")) URL.revokeObjectURL(image.url);
    onImage({ url: URL.createObjectURL(file) });
  };

  return (
    <section className="card p-4">
      <div className="eyebrow mb-3">Dermoscopic Lesion Image</div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pickFile(e.dataTransfer.files?.[0]);
        }}
        className={`relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition disabled:cursor-wait ${
          drag ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/40"
        }`}
      >
        {image ? (
          <>
            <img src={image.url} alt="Selected lesion" className="h-full w-full object-contain" />
            <span className="absolute bottom-2 right-2 rounded-md bg-slate-900/75 px-2 py-1 text-[11px] font-medium text-white">
              Click or drop to replace
            </span>
          </>
        ) : (
          <div className="px-6 text-center">
            <svg viewBox="0 0 24 24" className="mx-auto mb-2 h-9 w-9 text-slate-400" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <div className="text-[13px] font-semibold text-slate-700">Drop an image here or click to upload</div>
            <div className="mt-1 text-[11.5px] text-slate-500">JPG / PNG dermoscopy image · processed locally</div>
          </div>
        )}
      </button>
      <p className="mt-2.5 rounded-lg bg-amber-50 px-3 py-2 text-[11.5px] leading-relaxed text-amber-800">
        <strong>Dermoscopy images only.</strong> The models were trained on dermatoscope images; ordinary phone
        photos give unreliable results.
        <span className="block text-amber-700/90">
          รองรับเฉพาะภาพจากกล้องส่องผิวหนัง (dermoscopy) ภาพถ่ายจากมือถือทั่วไปให้ผลที่เชื่อถือไม่ได้
        </span>
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          pickFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </section>
  );
}

function Examples({ examples, active, onPick, disabled }) {
  if (!examples.length) return null;
  return (
    <section className="card p-4">
      <div className="eyebrow mb-1">Try a Sample Image</div>
      <p className="mb-3 text-[11.5px] text-slate-500">Held-out ISIC 2019 test images. Badge = ground-truth label.</p>
      <div className="grid grid-cols-4 gap-2">
        {examples.map((ex) => (
          <button
            key={ex.file}
            disabled={disabled}
            onClick={() => onPick({ url: ex.file, label: ex.label })}
            className={`group relative aspect-square overflow-hidden rounded-lg border-2 transition disabled:cursor-wait ${
              active === ex.file ? "border-blue-600" : "border-transparent hover:border-slate-300"
            }`}
            aria-label={`Sample image, ground truth ${ex.label}`}
          >
            <img src={ex.file} alt="" loading="lazy" className="h-full w-full object-cover transition group-hover:scale-105" />
            <span className="absolute bottom-1 left-1 rounded bg-slate-900/80 px-1.5 py-0.5 text-[10px] font-bold text-white">
              {ex.label}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

function Taxonomy({ classes }) {
  const group = (cancer) => classes.filter((c) => c.cancer === cancer);
  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="eyebrow">Detectable Taxonomy</span>
        <span className="text-[10.5px] text-slate-500">8 Classes</span>
      </div>
      {[true, false].map((cancer) => (
        <div key={String(cancer)} className="mb-2 last:mb-0">
          <div className={`mb-1.5 text-[10.5px] font-bold uppercase tracking-wider ${cancer ? "text-red-700" : "text-green-700"}`}>
            {cancer ? "Cancer / Pre-Cancer (High Risk)" : "Benign (Low Risk)"}
          </div>
          <div className="grid grid-cols-1 gap-x-3 gap-y-1 sm:grid-cols-2">
            {group(cancer).map((c) => (
              <div key={c.code} className="flex items-center gap-2 text-[11.5px] leading-snug text-slate-700">
                <CodePill code={c.code} cancer={cancer} />
                <span>{c.nameEn.replace(/\s*\(.*\)$/, "")}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

export function CodePill({ code, cancer }) {
  return (
    <span
      className={`inline-block min-w-[44px] rounded border px-1.5 py-0.5 text-center text-[10.5px] font-bold ${
        cancer ? "border-red-100 bg-red-50 text-red-700" : "border-green-100 bg-green-50 text-green-700"
      }`}
    >
      {code}
    </span>
  );
}

function Progress({ status, model }) {
  const loading = status.phase === "loading";
  const pct = Math.round((status.progress || 0) * 100);
  return (
    <div className="card flex min-h-[420px] flex-col items-center justify-center p-8 text-center">
      <div className="mb-4 h-10 w-10 animate-spin rounded-full border-4 border-blue-100 border-t-blue-600" />
      <div className="text-[14px] font-semibold text-slate-900">
        {loading ? `Loading ${model.name} model…` : "Analyzing lesion…"}
      </div>
      {loading && (
        <>
          <div className="mt-1 text-[12px] text-slate-500">
            One-time download ({model.sizeMB} MB), cached by your browser afterwards
          </div>
          <div className="mt-4 h-2 w-64 overflow-hidden rounded-full bg-slate-100">
            <div className="h-full rounded-full bg-blue-600 transition-[width]" style={{ width: `${pct}%` }} />
          </div>
          <div className="mt-1.5 text-[11.5px] font-semibold text-slate-600">{pct}%</div>
        </>
      )}
    </div>
  );
}

function Placeholder() {
  return (
    <div className="card flex min-h-[420px] flex-col items-center justify-center p-8 text-center">
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50">
        <svg viewBox="0 0 24 24" className="h-7 w-7 text-blue-600" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="11" cy="11" r="7" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      </div>
      <div className="text-[15px] font-bold text-slate-900">Awaiting Dermoscopic Image</div>
      <p className="mt-1 max-w-sm text-[12.5px] leading-relaxed text-slate-500">
        Upload a lesion image or pick a sample image. The two-level screening result will appear here.
      </p>
      <p className="mt-1 text-[12px] text-slate-400">อัปโหลดภาพรอยโรค หรือเลือกภาพตัวอย่างเพื่อเริ่มการวิเคราะห์</p>
    </div>
  );
}
