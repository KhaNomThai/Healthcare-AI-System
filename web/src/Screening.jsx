import { useEffect, useRef, useState } from "react";
import { loadModel, predict, preprocess } from "./inference.js";
import Result from "./Result.jsx";
import { CodePill, shortName } from "./ui.jsx";

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
        setStatus({ phase: "loading", progress: null }); // null = size unknown (no content-length / cached)
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

  const chooseImage = (img) => {
    // Free the previous upload's blob URL once it is replaced.
    setImage((prev) => {
      if (prev?.url.startsWith("blob:") && prev.url !== img.url) URL.revokeObjectURL(prev.url);
      return img;
    });
    // On narrow screens the result sits below the inputs, so bring it into view.
    if (window.matchMedia("(max-width: 1023px)").matches) {
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  };

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <div className="stagger space-y-4 lg:col-span-5">
        <ModelPicker models={data.models} value={modelIdx} onChange={setModelIdx} disabled={busy} hasImage={!!image} />
        <ImageInput image={image} onImage={chooseImage} disabled={busy} />
        <Examples examples={data.examples} active={image?.url} onPick={chooseImage} disabled={busy} />
        <Taxonomy classes={data.classes} />
      </div>

      <div ref={resultRef} className="scroll-mt-4 lg:col-span-7">
        {status.phase === "error" ? (
          <ErrorCard message={status.message} onRetry={() => setImage((i) => ({ ...i }))} />
        ) : busy ? (
          <Progress status={status} model={model} image={image} />
        ) : probs ? (
          <Result probs={probs} classes={data.classes} model={resultModel} groundTruth={image?.label} />
        ) : (
          <Placeholder />
        )}
      </div>
    </div>
  );
}

function ModelPicker({ models, value, onChange, disabled, hasImage }) {
  return (
    <section className="card p-4">
      <div className="eyebrow mb-1">Neural Network Architecture</div>
      <p className="mb-3 text-[13px] text-slate-500">
        {hasImage
          ? "Switch models to compare results on the same image."
          : "Select the backbone model for dermoscopic inference."}{" "}
        Each model is a one-time download of the size shown.
      </p>
      <div className="grid gap-2 sm:grid-cols-3">
        {models.map((m, i) => {
          const active = i === value;
          return (
            <button
              key={m.name}
              disabled={disabled}
              onClick={() => onChange(i)}
              aria-pressed={active}
              className={`flex items-center justify-between gap-2 rounded-xl border px-3 py-2 text-left transition duration-200 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-60 sm:block sm:p-2.5 ${
                active
                  ? "border-blue-600 bg-gradient-to-br from-blue-50 to-white shadow-sm ring-2 ring-blue-600/20"
                  : "border-slate-200 bg-white hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-md"
              }`}
            >
              <div className={`text-[13.5px] font-bold leading-tight ${active ? "text-blue-700" : "text-slate-900"}`}>
                {m.name}
              </div>
              <div className="flex gap-3 text-[12px] text-slate-500 sm:block">
                {m.metrics && (
                  <div className="sm:mt-1">
                    Top-1 <span className="font-semibold text-slate-700">{(m.metrics.overall_accuracy * 100).toFixed(1)}%</span>
                  </div>
                )}
                <div className="tabular-nums">{m.sizeMB} MB</div>
              </div>
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
    if (file?.type.startsWith("image/")) onImage({ url: URL.createObjectURL(file) });
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
        onDragLeave={(e) => {
          // dragleave also fires when moving onto a child element; ignore those.
          if (!e.currentTarget.contains(e.relatedTarget)) setDrag(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (!disabled) pickFile(e.dataTransfer.files?.[0]);
        }}
        className={`relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition disabled:cursor-wait ${
          drag ? "scale-[1.01] border-blue-500 bg-blue-50 ring-4 ring-blue-500/15" : "border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/40"
        }`}
      >
        {image ? (
          <>
            <img key={image.url} src={image.url} alt="Selected lesion" className="h-full w-full animate-rise object-contain" />
            <span className="absolute bottom-2 right-2 rounded-md bg-slate-900/75 px-2 py-1 text-[12px] font-medium text-white">
              Click or drop to replace
            </span>
          </>
        ) : (
          <div className="px-6 text-center">
            <svg viewBox="0 0 24 24" className={`mx-auto mb-2 h-9 w-9 transition ${drag ? "-translate-y-1 text-blue-500" : "text-slate-400"}`} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <div className="text-[14px] font-semibold text-slate-700">Drop an image here or click to upload</div>
            <div className="mt-1 text-[12.5px] text-slate-500">JPG / PNG dermoscopy image · processed locally</div>
          </div>
        )}
      </button>
      <p className="mt-2.5 rounded-lg bg-amber-50 px-3 py-2 text-[12.5px] leading-relaxed text-amber-800">
        <strong>Dermoscopy images only.</strong> The models were trained on dermatoscope images; ordinary phone
        photos give unreliable results.
        <span lang="th" className="block text-amber-800">
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
      <p className="mb-3 text-[12.5px] text-slate-500">Held-out ISIC 2019 test images. Badge = ground-truth label.</p>
      <div className="grid grid-cols-4 gap-2">
        {examples.map((ex) => (
          <button
            key={ex.file}
            disabled={disabled}
            onClick={() => onPick({ url: ex.file, label: ex.label })}
            className={`group relative aspect-square overflow-hidden rounded-lg border-2 transition disabled:cursor-wait ${
              active === ex.file ? "border-blue-600 ring-2 ring-blue-600/20" : "border-transparent hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md"
            }`}
            aria-label={`Sample image, ground truth ${ex.label}`}
            aria-pressed={active === ex.file}
          >
            <img src={ex.file} alt="" loading="lazy" className="h-full w-full object-cover transition group-hover:scale-105" />
            <span className="absolute bottom-1 left-1 rounded bg-slate-900/80 px-1.5 py-0.5 text-[11px] font-bold text-white">
              {ex.label}
            </span>
            {active === ex.file && (
              <span className="absolute right-1 top-1 flex h-5 w-5 animate-rise items-center justify-center rounded-full bg-blue-600 text-white shadow" aria-hidden="true">
                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </span>
            )}
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
        <span className="text-[11.5px] text-slate-500">{classes.length} Classes</span>
      </div>
      {[true, false].map((cancer) => (
        <div key={String(cancer)} className="mb-2 last:mb-0">
          <div className={`mb-1.5 text-[11.5px] font-bold uppercase tracking-wider ${cancer ? "text-red-700" : "text-green-700"}`}>
            {cancer ? "Cancer / Pre-Cancer (High Risk)" : "Benign (Low Risk)"}
          </div>
          <div className="grid grid-cols-1 gap-x-3 gap-y-1 sm:grid-cols-2">
            {group(cancer).map((c) => (
              <div key={c.code} className="flex items-center gap-2 text-[12.5px] leading-snug text-slate-700">
                <CodePill code={c.code} cancer={cancer} />
                <span>{shortName(c.nameEn)}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

function ErrorCard({ message, onRetry }) {
  return (
    <div className="card flex min-h-[420px] flex-col items-center justify-center border-red-200 p-8 text-center" role="alert">
      <div className="text-[15px] font-bold text-red-700">Analysis failed</div>
      <p className="mt-1 max-w-sm break-words text-[13.5px] text-slate-600">{message}</p>
      <p className="mt-1 text-[13px] text-slate-500">Check your connection and try again, or pick a different image.</p>
      <button
        onClick={onRetry}
        className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-[14px] font-semibold text-white transition hover:bg-slate-700 active:scale-[0.97]"
      >
        Try again
      </button>
    </div>
  );
}

function Progress({ status, model, image }) {
  const loading = status.phase === "loading";
  const known = typeof status.progress === "number";
  const pct = Math.round((status.progress || 0) * 100);
  return (
    <div className="card flex min-h-[420px] flex-col items-center justify-center p-8 text-center" aria-live="polite">
      {image ? (
        <div className="relative mb-5 h-48 w-48 overflow-hidden rounded-2xl bg-slate-900 shadow-lg ring-1 ring-slate-900/10">
          <img src={image.url} alt="" className={`h-full w-full object-cover transition duration-500 ${loading ? "opacity-40 blur-[2px]" : "opacity-90"}`} />
          {/* Viewfinder corners */}
          {["left-2 top-2 border-l-2 border-t-2", "right-2 top-2 border-r-2 border-t-2", "bottom-2 left-2 border-b-2 border-l-2", "bottom-2 right-2 border-b-2 border-r-2"].map((c) => (
            <span key={c} className={`absolute h-5 w-5 rounded-[3px] border-cyan-300 ${c}`} />
          ))}
          {!loading && (
            <div className="absolute inset-x-0 h-0 animate-scan">
              <div className="h-px bg-cyan-300 shadow-[0_0_12px_3px_rgb(103_232_249/0.8)]" />
              <div className="h-10 -translate-y-full bg-gradient-to-t from-cyan-300/30 to-transparent" />
            </div>
          )}
        </div>
      ) : (
        <div className="mb-4 h-10 w-10 animate-spin rounded-full border-4 border-blue-100 border-t-blue-600" />
      )}
      <div className="text-[14px] font-semibold text-slate-900">
        {loading ? `Loading ${model.name} model…` : "Analyzing lesion…"}
      </div>
      {loading ? (
        <>
          <div className="mt-1 text-[13px] text-slate-500">
            One-time download ({model.sizeMB} MB), cached by your browser afterwards
          </div>
          <div className="mt-4 h-2 w-64 overflow-hidden rounded-full bg-slate-100">
            {known ? (
              <div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-[width]" style={{ width: `${pct}%` }} />
            ) : (
              <div className="h-full w-1/3 animate-indeterminate rounded-full bg-gradient-to-r from-blue-500 to-cyan-400" />
            )}
          </div>
          {known && <div className="mt-1.5 text-[12.5px] font-semibold tabular-nums text-slate-600">{pct}%</div>}
        </>
      ) : (
        <div className="mt-1 text-[13px] text-slate-500">Running {model.name} on your device</div>
      )}
    </div>
  );
}

function Placeholder() {
  return (
    <div className="card flex min-h-[420px] flex-col items-center justify-center p-8 text-center">
      <div className="relative mb-5 flex h-16 w-16 items-center justify-center">
        <span className="absolute inset-0 animate-ping rounded-2xl bg-blue-200/50 [animation-duration:2.5s]" aria-hidden="true" />
        <span className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-50 to-blue-100 ring-1 ring-blue-200">
          <svg viewBox="0 0 24 24" className="h-7 w-7 text-blue-600" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </span>
      </div>
      <div className="text-[15px] font-bold text-slate-900">Awaiting Dermoscopic Image</div>
      <p className="mt-1 max-w-sm text-[13.5px] leading-relaxed text-slate-500">
        Upload a lesion image or pick a sample image. The two-level screening result will appear here.
      </p>
      <p lang="th" className="mt-1 text-[13px] text-slate-500">อัปโหลดภาพรอยโรค หรือเลือกภาพตัวอย่างเพื่อเริ่มการวิเคราะห์</p>
    </div>
  );
}
