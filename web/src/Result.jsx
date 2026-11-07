import { useEffect, useState } from "react";
import { CodePill } from "./Screening.jsx";

const pct = (p) => `${(p * 100).toFixed(1)}%`;

// Animate bars from 0 on each new result.
function useGrow(key) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    setOn(false);
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setOn(true)));
    return () => cancelAnimationFrame(id);
  }, [key]);
  return on;
}

export default function Result({ probs, classes, modelName, groundTruth }) {
  const grow = useGrow(probs);
  const cancerProb = probs.reduce((s, p, i) => s + (classes[i].cancer ? p : 0), 0);
  const benignProb = 1 - cancerProb;
  const highRisk = cancerProb >= 0.5;
  const top = probs.indexOf(Math.max(...probs));
  const topClass = classes[top];
  const rows = (cancer) =>
    classes
      .map((c, i) => ({ ...c, p: probs[i] }))
      .filter((c) => c.cancer === cancer)
      .sort((a, b) => b.p - a.p);

  return (
    <div className="space-y-4">
      {/* Level 1 */}
      <section
        className={`rounded-2xl border p-5 shadow-sm ${
          highRisk ? "border-red-200 bg-gradient-to-br from-red-50 to-white" : "border-green-200 bg-gradient-to-br from-green-50 to-white"
        }`}
      >
        <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="eyebrow">Level 1 — Binary Clinical Triage</div>
            <div className="mt-0.5 text-[11.5px] text-slate-500">
              Evaluated by <strong className="text-slate-900">{modelName}</strong>
            </div>
          </div>
          <span
            className={`rounded-full px-3.5 py-1 text-[11px] font-bold tracking-wide text-white shadow ${
              highRisk ? "bg-red-700" : "bg-green-700"
            }`}
          >
            {highRisk ? "HIGH RISK — ELEVATED MALIGNANCY PROBABILITY" : "LOW RISK — PREDOMINANTLY BENIGN PROFILE"}
          </span>
        </div>
        <div className="grid gap-4 rounded-xl border border-slate-900/5 bg-white/80 p-4 sm:grid-cols-2">
          <Meter label="Cancer / Pre-Cancer" value={cancerProb} grow={grow} tone="red" />
          <Meter label="Benign (Non-Cancer)" value={benignProb} grow={grow} tone="green" />
        </div>
      </section>

      {/* Top-1 */}
      <section className={`card border-l-4 p-5 ${topClass.cancer ? "border-l-red-700" : "border-l-green-700"}`}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="eyebrow mb-1">Primary Suspected Condition (Top-1)</div>
            <div className="text-[17px] font-bold text-slate-900">{topClass.nameEn}</div>
            <div className="text-[13px] font-medium text-slate-600">{topClass.nameTh}</div>
          </div>
          <div className="text-right">
            <span
              className={`inline-block rounded-full px-2.5 py-0.5 text-[10.5px] font-bold ${
                topClass.cancer ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
              }`}
            >
              {topClass.cancer ? "Cancer / Pre-Cancer Group" : "Benign Lesion Group"}
            </span>
            <div className="mt-1.5 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-[12px] font-bold text-slate-900">
              Confidence: {pct(probs[top])}
            </div>
          </div>
        </div>
        <div className="mt-3 border-t border-slate-100 pt-3 text-[12px] leading-relaxed">
          <p className="text-slate-600">{topClass.descEn}</p>
          <p className="mt-1 text-slate-500">{topClass.descTh}</p>
        </div>
        {groundTruth && (
          <div
            className={`mt-3 rounded-lg px-3 py-2 text-[11.5px] font-medium ${
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
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-900">
            Level 2 — Sub-Type Differential Probability
          </span>
          <span className="text-[10.5px] text-slate-500">8 Classes</span>
        </div>
        <div className="p-5">
          {[true, false].map((cancer) => (
            <div key={String(cancer)} className={cancer ? "" : "mt-4"}>
              <div
                className={`mb-2 border-b pb-1.5 text-[10.5px] font-bold uppercase tracking-wider ${
                  cancer ? "border-red-100 text-red-700" : "border-green-100 text-green-700"
                }`}
              >
                {cancer ? "Cancer / Pre-Cancer Sub-Types" : "Benign Lesion Sub-Types"}
              </div>
              {rows(cancer).map((c) => (
                <div key={c.code} className="flex items-center gap-3 py-1.5">
                  <CodePill code={c.code} cancer={cancer} />
                  <div className="min-w-0 flex-1 text-[12px]">
                    <span className="font-medium text-slate-900">{c.nameEn.replace(/\s*\(.*\)$/, "")}</span>
                    <span className="ml-1.5 hidden text-slate-400 sm:inline">({c.nameTh.replace(/\s*\(.*\)$/, "")})</span>
                  </div>
                  <div className={`h-1.5 w-20 shrink-0 overflow-hidden rounded-full sm:w-32 ${cancer ? "bg-red-100" : "bg-green-100"}`}>
                    <div
                      className={`h-full rounded-full transition-[width] duration-700 ease-out ${cancer ? "bg-red-600" : "bg-green-600"}`}
                      style={{ width: grow ? `${Math.max(c.p * 100, 0.5)}%` : 0 }}
                    />
                  </div>
                  <span className={`w-12 text-right text-[12px] font-bold ${cancer ? "text-red-700" : "text-green-700"}`}>
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
        className={`rounded-xl border border-l-4 p-4 text-[12px] leading-relaxed ${
          highRisk ? "border-red-200 border-l-red-700 bg-red-50" : "border-green-200 border-l-green-700 bg-green-50"
        }`}
      >
        <div className={`mb-1.5 text-[10.5px] font-bold uppercase tracking-wider ${highRisk ? "text-red-700" : "text-green-700"}`}>
          Clinical Triage Recommendation
        </div>
        {highRisk ? (
          <>
            <p className="text-slate-800">
              Elevated probability for malignant or pre-cancerous lesion detected. Referral to a dermatologist for
              clinical evaluation and possible biopsy is strongly recommended.
            </p>
            <p className="mt-1 text-slate-600">
              ตรวจพบความเสี่ยงสูงต่อรอยโรคมะเร็งหรือก่อนมะเร็ง แนะนำให้ส่งต่อแพทย์ผิวหนังเพื่อประเมินทางคลินิก
              และพิจารณาตรวจชิ้นเนื้อยืนยันผลทางพยาธิวิทยา
            </p>
          </>
        ) : (
          <>
            <p className="text-slate-800">
              Analysis suggests a benign lesion with low malignancy probability. Periodic self-monitoring is
              recommended. Consult a physician if changes in size, color, border, or symptoms are observed.
            </p>
            <p className="mt-1 text-slate-600">
              ผลการวิเคราะห์บ่งชี้ว่าเป็นรอยโรคที่ไม่ร้ายแรง มีความเสี่ยงต่ำต่อมะเร็ง แนะนำให้ติดตามสังเกตอาการเป็นระยะ
              หากพบการเปลี่ยนแปลงของขนาด สี ขอบ หรือมีอาการผิดปกติ ควรปรึกษาแพทย์
            </p>
          </>
        )}
      </section>

      <p className="text-center text-[10.5px] leading-relaxed text-slate-400">
        For educational and research purposes only (CPE310). Results are probabilistic estimates and do not
        constitute a medical diagnosis.
      </p>
    </div>
  );
}

function Meter({ label, value, grow, tone }) {
  const red = tone === "red";
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[12px] font-semibold text-slate-700">{label}</span>
        <span className={`text-lg font-extrabold ${red ? "text-red-700" : "text-green-700"}`}>{pct(value)}</span>
      </div>
      <div className={`h-2 overflow-hidden rounded-full ${red ? "bg-red-100" : "bg-green-100"}`}>
        <div
          className={`h-full rounded-full bg-gradient-to-r transition-[width] duration-700 ease-out ${
            red ? "from-red-500 to-red-700" : "from-green-500 to-green-700"
          }`}
          style={{ width: grow ? pct(value) : 0 }}
        />
      </div>
    </div>
  );
}
