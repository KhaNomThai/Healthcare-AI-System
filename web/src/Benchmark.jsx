const MODEL_COLORS = ["#2563eb", "#0d9488", "#d97706"]; // blue, teal, amber

// higherIsBetter=false for error-type metrics (best = lowest)
const METRICS = [
  { key: "overall_accuracy", label: "Top-1 Accuracy", note: "Exact 8-class match" },
  { key: "top2_accuracy", label: "Top-2 Accuracy", note: "True class in top 2" },
  { key: "top3_accuracy", label: "Top-3 Accuracy", note: "True class in top 3" },
  { key: "binary_accuracy", label: "Binary Triage Accuracy", note: "Cancer vs. benign" },
  { key: "cancer_auc", label: "Cancer AUC-ROC", note: "Ranking quality", raw: true },
  { key: "cancer_sensitivity", label: "Cancer Sensitivity", note: "Cancers caught" },
  { key: "cancer_specificity", label: "Cancer Specificity", note: "Benign correctly cleared" },
  { key: "false_negative_rate", label: "False Negative Rate", note: "Cancers missed", lowerIsBetter: true },
  { key: "macro_f1", label: "Macro F1", note: "Class-balanced", raw: true },
  { key: "weighted_f1", label: "Weighted F1", note: "Support-weighted", raw: true },
];

const fmt = (m, v) => (m.raw ? v.toFixed(3) : `${(v * 100).toFixed(1)}%`);

export default function Benchmark({ data }) {
  const models = data.models.filter((m) => m.metrics);
  if (!models.length) return <div className="card p-6 text-sm text-slate-500">No benchmark metrics available.</div>;

  const best = (m) => {
    const vals = models.map((x) => x.metrics[m.key]);
    return m.lowerIsBetter ? Math.min(...vals) : Math.max(...vals);
  };

  return (
    <div className="space-y-5">
      {/* Headline cards */}
      <div className="grid gap-4 md:grid-cols-3">
        {models.map((m, i) => (
          <section key={m.name} className="card relative overflow-hidden p-5">
            <div className="absolute inset-x-0 top-0 h-1" style={{ background: MODEL_COLORS[i] }} />
            <div className="flex items-center justify-between">
              <h3 className="text-[15px] font-bold text-slate-900">{m.name}</h3>
              {m.metrics.overall_accuracy === best(METRICS[0]) && (
                <span className="rounded-full bg-blue-600 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
                  Best
                </span>
              )}
            </div>
            <div className="mt-3 text-3xl font-extrabold tracking-tight text-slate-900">
              {(m.metrics.overall_accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-[11.5px] text-slate-500">Top-1 accuracy (held-out test set)</div>
            <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-2 border-t border-slate-100 pt-3 text-[11.5px]">
              <Stat label="AUC-ROC" value={m.metrics.cancer_auc.toFixed(3)} />
              <Stat label="Sensitivity" value={`${(m.metrics.cancer_sensitivity * 100).toFixed(1)}%`} />
              <Stat label="Parameters" value={`${(m.metrics.total_params / 1e6).toFixed(1)}M`} />
              <Stat label="Train time" value={`${(m.metrics.training_time_seconds / 60).toFixed(0)} min`} />
            </dl>
          </section>
        ))}
      </div>

      {/* Full comparison table */}
      <section className="card overflow-hidden">
        <div className="border-b border-slate-200 bg-slate-50 px-5 py-3">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-900">Clinical Metric Comparison</span>
          <span className="ml-2 text-[10.5px] text-slate-500">Best value per row highlighted</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-[12px]">
            <thead>
              <tr className="border-b border-slate-200 text-left text-[10.5px] uppercase tracking-wider text-slate-500">
                <th className="px-5 py-2.5 font-semibold">Metric</th>
                {models.map((m, i) => (
                  <th key={m.name} className="px-3 py-2.5 text-right font-semibold">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: MODEL_COLORS[i] }} />
                    {m.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRICS.map((metric) => {
                const b = best(metric);
                return (
                  <tr key={metric.key} className="border-b border-slate-100 last:border-0">
                    <td className="px-5 py-2.5">
                      <div className="font-semibold text-slate-800">{metric.label}</div>
                      <div className="text-[10.5px] text-slate-400">{metric.note}</div>
                    </td>
                    {models.map((m) => {
                      const v = m.metrics[metric.key];
                      const isBest = v === b;
                      return (
                        <td key={m.name} className="px-3 py-2.5 text-right tabular-nums">
                          <span className={isBest ? "rounded-md bg-blue-50 px-2 py-0.5 font-bold text-blue-700" : "text-slate-700"}>
                            {fmt(metric, v)}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Per-class F1 */}
      <section className="card p-5">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-900">Per-Class F1 Score</span>
          <div className="flex flex-wrap gap-3 text-[11px] text-slate-600">
            {models.map((m, i) => (
              <span key={m.name} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: MODEL_COLORS[i] }} />
                {m.name}
              </span>
            ))}
          </div>
        </div>
        <p className="mb-4 text-[11.5px] text-slate-500">Rare classes (SCC, AK, DF) remain the hardest to separate.</p>
        <div className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
          {data.classes.map((c) => (
            <div key={c.code}>
              <div className="mb-1 flex items-center gap-2 text-[11.5px]">
                <span className={`font-bold ${c.cancer ? "text-red-700" : "text-green-700"}`}>{c.code}</span>
                <span className="truncate text-slate-500">{c.nameEn.replace(/\s*\(.*\)$/, "")}</span>
              </div>
              <div className="space-y-1">
                {models.map((m, i) => {
                  const v = m.metrics.per_class_f1?.[c.code] ?? 0;
                  return (
                    <div key={m.name} className="flex items-center gap-2">
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                        <div className="h-full rounded-full" style={{ width: `${v * 100}%`, background: MODEL_COLORS[i] }} />
                      </div>
                      <span className="w-9 text-right text-[11px] tabular-nums text-slate-600">{v.toFixed(2)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Confusion matrices */}
      <section className="card p-5">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-900">Confusion Matrices (Test Set)</span>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          {models.filter((m) => m.cm).map((m) => (
            <figure key={m.name} className="rounded-xl border border-slate-200 p-2">
              <img src={m.cm} alt={`${m.name} confusion matrix`} loading="lazy" className="w-full rounded-lg" />
              <figcaption className="mt-1.5 text-center text-[11.5px] font-semibold text-slate-700">{m.name}</figcaption>
            </figure>
          ))}
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div>
      <dt className="text-slate-400">{label}</dt>
      <dd className="font-semibold text-slate-800">{value}</dd>
    </div>
  );
}
