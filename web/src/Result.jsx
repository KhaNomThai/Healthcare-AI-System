import { CodePill, CountUp, pct, shortName, useGrow } from "./ui.jsx";

export default function Result({ probs, classes, model, groundTruth }) {
  const grow = useGrow(probs);
  const missRate = model.metrics?.false_negative_rate; // share of test-set cancers this model called benign
  const cancerProb = probs.reduce((s, p, i) => s + (classes[i].cancer ? p : 0), 0);
  // Tuned on validation for the target sensitivity (evaluate.py); 0.5 for models exported before that.
  const threshold = model.metrics?.cancer_threshold ?? 0.5;
  const highRisk = cancerProb >= threshold;
  const top = probs.indexOf(Math.max(...probs));
  const topClass = classes[top];
  const rows = (cancer) =>
    classes
      .map((c, i) => ({ ...c, p: probs[i] }))
      .filter((c) => c.cancer === cancer)
      .sort((a, b) => b.p - a.p);

  return (
    <div className="stagger space-y-4">
      {/* Level 1 */}
      <section
        className={`rounded-2xl border p-5 shadow-sm ${
          highRisk ? "border-red-200 bg-gradient-to-br from-red-50 to-white" : "border-green-200 bg-gradient-to-br from-green-50 to-white"
        }`}
      >
        <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="eyebrow">Level 1 — Binary Clinical Triage</div>
            <div className="mt-0.5 text-[12.5px] text-slate-500">
              Evaluated by <strong className="text-slate-900">{model.name}</strong>
            </div>
          </div>
          <span className={`rounded-full px-3.5 py-1 text-[13px] font-bold text-white shadow ${highRisk ? "bg-red-700" : "bg-green-700"}`}>
            {highRisk ? "Higher risk" : "Lower risk"}
          </span>
        </div>
        <div className="flex flex-col items-center gap-5 rounded-xl border border-slate-900/5 bg-white/80 p-4 sm:flex-row">
          <Gauge value={cancerProb} grow={grow} highRisk={highRisk} />
          <div className="w-full flex-1">
            <p className="text-[14px] font-semibold text-slate-900">
              {highRisk ? "Elevated probability of a malignant or pre-cancerous lesion." : "Leans benign — this does not rule out cancer."}
            </p>
            <SplitBar cancer={cancerProb} grow={grow} threshold={threshold} />
          </div>
        </div>
      </section>

      {/* Top-1 */}
      <section className={`card border-l-4 p-5 ${topClass.cancer ? "border-l-red-700" : "border-l-green-700"}`}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="eyebrow mb-1">Primary Suspected Condition (Top-1)</div>
            <div className="text-[17px] font-bold text-slate-900">{topClass.nameEn}</div>
            <div lang="th" className="text-[14px] font-medium text-slate-600">{topClass.nameTh}</div>
          </div>
          <div className="text-right">
            <span
              className={`inline-block rounded-full px-2.5 py-0.5 text-[11.5px] font-bold ${
                topClass.cancer ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
              }`}
            >
              {topClass.cancer ? "Cancer / Pre-Cancer Group" : "Benign Lesion Group"}
            </span>
            <div className="mt-1.5 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-[13px] font-bold text-slate-900">
              Model score: {pct(probs[top])}
            </div>
          </div>
        </div>
        {topClass.cancer !== highRisk && <Mismatch topClass={topClass} cancerProb={cancerProb} threshold={threshold} />}
        <div className="mt-3 border-t border-slate-100 pt-3 text-[13px] leading-relaxed">
          <p className="text-slate-600">{topClass.descEn}</p>
          <p lang="th" className="mt-1 text-slate-500">{topClass.descTh}</p>
        </div>
        {groundTruth && (
          <div
            className={`mt-3 rounded-lg px-3 py-2 text-[12.5px] font-medium ${
              groundTruth === topClass.code ? "bg-blue-50 text-blue-800" : "bg-amber-50 text-amber-800"
            }`}
          >
            Sample ground truth: <strong>{groundTruth}</strong> —{" "}
            {groundTruth === topClass.code ? "model prediction matches" : "model prediction differs"}
          </div>
        )}
      </section>

      {/* Level 2 */}
      <section className="card overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-5 py-3">
          <span className="text-[12px] font-bold uppercase tracking-wider text-slate-900">
            Level 2 — Sub-Type Differential Probability
          </span>
          <span className="text-[11.5px] text-slate-500">{classes.length} Classes</span>
        </div>
        <div className="p-5">
          {[true, false].map((cancer) => (
            <div key={String(cancer)} className={cancer ? "" : "mt-4"}>
              <div
                className={`mb-2 border-b pb-1.5 text-[11.5px] font-bold uppercase tracking-wider ${
                  cancer ? "border-red-100 text-red-700" : "border-green-100 text-green-700"
                }`}
              >
                {cancer ? "Cancer / Pre-Cancer Sub-Types" : "Benign Lesion Sub-Types"}
              </div>
              {rows(cancer).map((c) => (
                <div key={c.code} className="-mx-2 flex items-center gap-3 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-50">
                  <CodePill code={c.code} cancer={cancer} />
                  <div className="min-w-0 flex-1 text-[13px]">
                    <span className="font-medium text-slate-900">{shortName(c.nameEn)}</span>
                    <span lang="th" className="ml-1.5 hidden text-slate-500 sm:inline">({shortName(c.nameTh)})</span>
                  </div>
                  <div className={`h-1.5 w-20 shrink-0 overflow-hidden rounded-full sm:w-32 ${cancer ? "bg-red-100" : "bg-green-100"}`}>
                    <div
                      className={`h-full rounded-full transition-[width] duration-700 ease-out ${cancer ? "bg-red-600" : "bg-green-600"}`}
                      style={{ width: grow ? `${Math.max(c.p * 100, 0.5)}%` : 0 }}
                    />
                  </div>
                  <span className={`w-12 text-right text-[13px] font-bold tabular-nums ${cancer ? "text-red-700" : "text-green-700"}`}>
                    {pct(c.p)}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      {/* Recommendation */}
      <section
        className={`rounded-xl border border-l-4 p-4 text-[13px] leading-relaxed ${
          highRisk ? "border-red-200 border-l-red-700 bg-red-50" : "border-green-200 border-l-green-700 bg-green-50"
        }`}
      >
        <div className={`mb-1.5 text-[11.5px] font-bold uppercase tracking-wider ${highRisk ? "text-red-700" : "text-green-700"}`}>
          Clinical Triage Recommendation
        </div>
        {highRisk ? (
          <>
            <p className="text-slate-800">
              Elevated probability for malignant or pre-cancerous lesion detected. Referral to a dermatologist for
              clinical evaluation and possible biopsy is strongly recommended.
            </p>
            <p lang="th" className="mt-1 text-slate-600">
              ตรวจพบความเสี่ยงสูงต่อรอยโรคมะเร็งหรือก่อนมะเร็ง แนะนำให้ส่งต่อแพทย์ผิวหนังเพื่อประเมินทางคลินิก
              และพิจารณาตรวจชิ้นเนื้อยืนยันผลทางพยาธิวิทยา
            </p>
          </>
        ) : (
          <>
            <p className="text-slate-800">
              The model leans towards a benign lesion, but this does not rule out cancer
              {missRate != null && <> — on the test set it missed about {pct(missRate)} of cancers</>}. Keep
              monitoring the lesion, and see a physician if it changes in size, color or border, or if you are
              concerned.
            </p>
            <p lang="th" className="mt-1 text-slate-600">
              โมเดลประเมินว่าน่าจะเป็นรอยโรคที่ไม่ร้ายแรง แต่ไม่สามารถตัดความเป็นมะเร็งออกได้
              {missRate != null && <> (ในชุดทดสอบ โมเดลพลาดมะเร็งประมาณ {pct(missRate)})</>} ควรสังเกตรอยโรคต่อไป
              และปรึกษาแพทย์หากขนาด สี หรือขอบเปลี่ยนแปลง หรือมีข้อกังวล
            </p>
          </>
        )}
      </section>

      <p className="text-center text-[11.5px] leading-relaxed text-slate-500">
        For educational and research purposes only (CPE310). Model scores are uncalibrated estimates, not
        diagnostic certainty, and do not constitute a medical diagnosis.
      </p>
    </div>
  );
}

// Level 1 sums all cancer sub-types, Top-1 is the single highest class, so they can disagree.
function Mismatch({ topClass, cancerProb, threshold }) {
  return (
    <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[12.5px] leading-relaxed text-amber-900">
      {topClass.cancer ? (
        <>
          <p>
            <strong>Note:</strong> the single most likely class is a cancer type, although all cancer sub-types together
            score only {pct(cancerProb)}, below the {pct(threshold)} threshold. Treat this lesion as suspicious and have it checked.
          </p>
          <p lang="th" className="mt-0.5 text-amber-800">
            คลาสที่มีคะแนนสูงสุดเป็นกลุ่มมะเร็ง แม้คะแนนรวมกลุ่มมะเร็งจะต่ำกว่าเกณฑ์ {pct(threshold)} ควรถือว่าน่าสงสัยและปรึกษาแพทย์
          </p>
        </>
      ) : (
        <>
          <p>
            <strong>Note:</strong> the single most likely class is benign, but the cancer sub-types together add up to{" "}
            {pct(cancerProb)}. The Level 1 risk above takes priority.
          </p>
          <p lang="th" className="mt-0.5 text-amber-800">
            คลาสที่มีคะแนนสูงสุดเป็นกลุ่มไม่ร้ายแรง แต่คะแนนรวมของกลุ่มมะเร็งสูงกว่า ให้ยึดผลความเสี่ยง Level 1 ด้านบนเป็นหลัก
          </p>
        </>
      )}
    </div>
  );
}

function Gauge({ value, grow, highRisk }) {
  const r = 42;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative h-32 w-32 shrink-0">
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90" aria-hidden="true">
        <circle cx="50" cy="50" r={r} fill="none" strokeWidth="9" className="stroke-slate-100" />
        <circle
          cx="50"
          cy="50"
          r={r}
          fill="none"
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={grow ? c * (1 - value) : c}
          className={`transition-[stroke-dashoffset] duration-1000 ease-out ${highRisk ? "stroke-red-600" : "stroke-green-600"}`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-2xl font-extrabold tabular-nums ${highRisk ? "text-red-700" : "text-green-700"}`}>
          <CountUp value={value} />
        </span>
        <span className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Cancer score</span>
      </div>
    </div>
  );
}

// One bar split into cancer (red) vs benign (green); the two shares always sum to 100%.
function SplitBar({ cancer, grow, threshold }) {
  const w = grow ? cancer : 0.5;
  return (
    <div className="mt-3">
      <div className="mb-1.5 flex justify-between text-[13px] font-semibold">
        <span className="text-red-700">
          Cancer / Pre-Cancer <span className="tabular-nums">{pct(cancer)}</span>
        </span>
        <span className="text-green-700">
          Benign <span className="tabular-nums">{pct(1 - cancer)}</span>
        </span>
      </div>
      <div className="relative">
        <div className="flex h-2.5 gap-0.5 overflow-hidden rounded-full">
          <div className="rounded-l-full bg-red-600 transition-[width] duration-700 ease-out" style={{ width: pct(w) }} />
          <div className="flex-1 rounded-r-full bg-green-600" />
        </div>
        {/* Referral threshold marker */}
        <div className="absolute -inset-y-1 w-0.5 rounded bg-slate-900" style={{ left: pct(threshold) }} aria-hidden="true" />
      </div>
      <p className="mt-1.5 text-[12px] text-slate-500">
        Marker = referral threshold: <strong className="text-slate-700">{pct(threshold)}</strong> cancer score or above is
        flagged as higher risk.
      </p>
    </div>
  );
}
