import { useState } from "react";
import { api, fmt, METHOD } from "../lib/api";
import { useApi } from "../lib/useApi";
import { C, Chart } from "../components/Chart";
import { Note, Page, Pending, Seg } from "../components/ui";

type Task = "cn_ad" | "cn_mci";
const METHODS = ["msp", "energy", "mc_dropout", "feature_maha", "adrs"];
const COMP: Record<string, string> = { label_definition: "Label definition", age_sex: "Age / sex", field_strength: "Field strength", residual: "Residual shift" };

export default function Reliability() {
  const [task, setTask] = useState<Task>("cn_ad");
  const [coh, setCoh] = useState<"adni" | "oasis3">("oasis3");
  const run = `proto_resnet3d_${task}_s0`;
  const rel = useApi(() => api.table("reliability"));
  const sd = useApi(() => api.table("shift_decomposition"));
  const rc = useApi(() => Promise.all(METHODS.map((m) => api.file("risk_coverage", `${run}_${coh}_${m}`))), [task, coh]);
  const rows = (rel.data?.rows ?? []).filter((r) => r.run_id === run && r.cohort === coh);
  const g = (m: string, k: string) => rows.find((r) => r.method === m && r.metric === k)?.value as number | undefined;
  const dec = (sd.data?.rows ?? []).filter((r) => r.run_id === `proto_ensemble_resnet3d_${task}` && r.cohort === "oasis3");
  const comps = ["label_definition", "age_sex", "field_strength", "residual"].map((k) => dec.find((r) => r.component === k));
  const total = dec.find((r) => r.component === "total_drop");
  return (
    <Page eyebrow="Robustness & reliability (C1, C3)" title="Why does performance drop, and can we tell when the model is wrong?"
      lead="Left: where the ADNI → OASIS-3 AUROC drop comes from. Right: how well different uncertainty scores flag the scans the model misclassifies, including our label-free ADRS score.">
      <div className="flex flex-wrap items-center gap-4">
        <Seg value={task} onChange={setTask} options={[["cn_ad", "CN vs AD"], ["cn_mci", "CN vs MCI"]]} />
        <Seg value={coh} onChange={setCoh} options={[["oasis3", "OASIS-3 (external)"], ["adni", "ADNI (internal)"]]} />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        {total ? (
          <Chart title="Where the cross-cohort drop comes from" subtitle={`Total AUROC drop ADNI → OASIS-3: ${fmt(total.value, 3)} (95% CI ${fmt(total.ci_low, 2)} to ${fmt(total.ci_high, 2)})`}
            data={[{ type: "waterfall", orientation: "v", measure: [...comps.map(() => "relative"), "total"], x: [...comps.map((_, i) => COMP[Object.keys(COMP)[i]]), "Total drop"],
              y: [...comps.map((r) => Number(r?.value ?? 0)), 0], text: [...comps.map((r) => fmt(r?.value, 3)), fmt(total.value, 3)], textposition: "outside",
              increasing: { marker: { color: C.int } }, decreasing: { marker: { color: C.green } }, totals: { marker: { color: C.ext } }, connector: { line: { color: "#cbd5e1" } } }] as Plotly.Data[]}
            layout={{ yaxis: { title: { text: "Δ AUROC" } }, showlegend: false }} source="results/shift_decomposition.csv" />
        ) : <Pending what="Shift decomposition" />}
        {rows.length ? (
          <Chart title="Detecting the model's own mistakes" subtitle="AUROC for separating wrong from correct predictions (0.5 = no better than chance)"
            data={[{ type: "bar", x: METHODS.map((m) => METHOD[m]), y: METHODS.map((m) => g(m, "err_auroc") ?? null), text: METHODS.map((m) => fmt(g(m, "err_auroc"), 2)), textposition: "outside",
              marker: { color: METHODS.map((m) => (m === "adrs" ? C.int : "#c7d2fe")) } }] as Plotly.Data[]}
            layout={{ yaxis: { range: [0.3, 1], title: { text: "error-detection AUROC" } }, showlegend: false }} source="results/reliability.csv" />
        ) : <Pending what="Reliability scores" />}
      </div>
      {rc.data?.some(Boolean) && (
        <Chart title="Risk–coverage: refer uncertain scans to a clinician" subtitle="Error rate on the scans the model keeps, as it refers more of them (lower curve is better)"
          data={rc.data.map((d, i) => d && { type: "scatter", mode: "lines", name: METHOD[METHODS[i]], x: d.rows.map((r) => r.coverage), y: d.rows.map((r) => r.risk),
            line: { width: METHODS[i] === "adrs" ? 4 : 2, color: METHODS[i] === "adrs" ? C.int : undefined } }).filter(Boolean) as Plotly.Data[]}
          layout={{ xaxis: { title: { text: "coverage (fraction of scans the model decides)" }, range: [0.1, 1] }, yaxis: { title: { text: "error rate" } } }} source="results/risk_coverage" />
      )}
      {rows.length > 0 && (
        <div className="card overflow-x-auto">
          <h3 className="h2 mb-3">Selective prediction summary</h3>
          <table className="data">
            <thead><tr><th>Method</th><th>Error-detection AUROC</th><th>AURC ↓</th><th>Accuracy @ 80% coverage</th><th>Accuracy @ 90% coverage</th></tr></thead>
            <tbody>{METHODS.map((m) => (
              <tr key={m} className={m === "adrs" ? "bg-brand-50/60 font-semibold" : ""}><td>{METHOD[m]}</td><td>{fmt(g(m, "err_auroc"))}</td><td>{fmt(g(m, "aurc"))}</td><td>{fmt(g(m, "acc@80"))}</td><td>{fmt(g(m, "acc@90"))}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      <Note>ADRS needs no labels on the new cohort: it measures how unusual a scan's explanation pattern is compared with ADNI validation scans. In the full study it is also compared with a deep ensemble and with GAIA-style attribution-abnormality scores.</Note>
    </Page>
  );
}
