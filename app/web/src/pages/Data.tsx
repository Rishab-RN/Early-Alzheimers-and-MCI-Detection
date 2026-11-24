import { useState } from "react";
import { api, type Row } from "../lib/api";
import { useApi } from "../lib/useApi";
import { C, Chart } from "../components/Chart";
import { Note, Page, Pending, Seg } from "../components/ui";

const CLS = ["CN", "MCI", "AD"];
const COH: [string, string, string][] = [["adni", "ADNI", C.int], ["oasis3", "OASIS-3", C.ext]];

export default function Data() {
  const [def, setDef] = useState<"dx_clinical" | "dx_cdr" | "dx_amyloid">("dx_clinical");
  const cs = useApi(() => api.table("cohort_summary"));
  const ov = useApi(() => api.table("label_overlap"));
  const lk = useApi(() => api.table("leakage_checks"));
  if (!cs.data?.rows.length) return <Page title="Data & cohorts"><Pending what="Cohort summary" /></Page>;
  const rows = cs.data.rows.filter((r) => r.label_def === def);
  const g = (c: string, v: string, k: string, f: keyof Row = "count") => rows.find((r) => r.cohort === c && r.variable === v && r.group === k)?.[f] as number | undefined;
  const ovRows = (ov.data?.rows ?? []).filter((r) => r.def_a === "dx_clinical" && r.def_b === "dx_cdr");
  const heat = (c: string) => {
    const sub = ovRows.filter((r) => r.cohort === c);
    const cols = ["CN", "MCI", "AD"];
    return { type: "heatmap", x: cols.map((x) => `CDR → ${x}`), y: CLS.map((x) => `Clinical ${x}`),
      z: CLS.map((a) => cols.map((b) => Number(sub.find((r) => r.class_a === a && r.class_b === b)?.count ?? 0))),
      text: CLS.map((a) => cols.map((b) => String(sub.find((r) => r.class_a === a && r.class_b === b)?.count ?? 0))),
      texttemplate: "%{text}", colorscale: [[0, "#f8fafc"], [1, c === "adni" ? C.int : C.ext]], showscale: false } as Plotly.Data;
  };
  return (
    <Page eyebrow="Data" title="Two independent cohorts, three ways to label them"
      lead={<>One baseline T1-weighted MRI per subject. Models are trained on <b>ADNI</b> only; <b>OASIS-3</b> is a locked external test cohort. Every subject is labelled three ways so we can measure <i>label-definition shift</i>.</>}>
      <div className="grid gap-4 md:grid-cols-2">
        {COH.map(([c, name, col]) => {
          const n = CLS.reduce((s, k) => s + (Number(rows.find((r) => r.cohort === c && r.variable === "n" && r.group === k)?.count) || 0), 0);
          return (
            <div key={c} className="card">
              <div className="flex items-center gap-2"><span className="h-3 w-3 rounded-full" style={{ background: col }} /><span className="font-semibold">{name}</span>
                <span className="pill ml-auto bg-ink-100 text-ink-700">{c === "adni" ? "training + internal test" : "external test only"}</span></div>
              <div className="mt-3 text-3xl font-semibold tabular-nums">{n.toLocaleString()} <span className="text-base font-normal muted">labelled subjects</span></div>
              <div className="mt-3 flex gap-6 text-sm">
                {CLS.map((k) => <div key={k}><div className="muted">{k}</div><div className="font-semibold tabular-nums">{g(c, "n", k) ?? 0}</div></div>)}
                <div><div className="muted">Female</div><div className="font-semibold">{Math.round(100 * (g(c, "female", "CN", "mean") ?? 0))}% (CN)</div></div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-3"><span className="text-sm muted">Label definition</span>
        <Seg value={def} onChange={setDef} options={[["dx_clinical", "Clinical"], ["dx_cdr", "CDR"], ["dx_amyloid", "Amyloid-confirmed"]]} /></div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Chart title="Subjects per class" subtitle="Changes with the label definition"
          data={COH.map(([c, name, col]) => ({ type: "bar", name, x: CLS, y: CLS.map((k) => g(c, "n", k) ?? 0), marker: { color: col } })) as Plotly.Data[]}
          layout={{ barmode: "group", yaxis: { title: { text: "subjects" } } }} source="results/cohort_summary.csv" />
        <Chart title="Age by class" subtitle="Mean ± SD. AD groups are older, so age is a confound we control for"
          data={COH.map(([c, name, col]) => ({ type: "bar", name, x: CLS, y: CLS.map((k) => g(c, "age", k, "mean") ?? null),
            error_y: { type: "data", array: CLS.map((k) => g(c, "age", k, "sd") ?? 0), visible: true, color: "#94a3b8" }, marker: { color: col } })) as Plotly.Data[]}
          layout={{ barmode: "group", yaxis: { title: { text: "years" }, range: [50, 90] } }} source="results/cohort_summary.csv" />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        {COH.map(([c, name]) => (
          <Chart key={c} title={`${name}: clinical diagnosis vs CDR label`} height={300}
            subtitle={c === "oasis3" ? "Most OASIS-3 clinical AD cases are CDR 0.5, i.e. 'MCI-like' by CDR" : "Same subjects, two label definitions"}
            data={[heat(c)]} layout={{ margin: { l: 110, r: 10, t: 10, b: 50 } }} source="results/label_overlap.csv" />
        ))}
      </div>
      <Note>The off-diagonal cells are <b>label-definition shift</b>: the same person counts as a different class depending on the cohort's diagnostic practice. Contribution C1 measures how much of the cross-cohort performance drop this explains.</Note>
      {lk.data?.rows.length ? (
        <div className="card">
          <h3 className="h2 mb-3">Leakage checks</h3>
          <div className="grid gap-3 md:grid-cols-3">
            {lk.data.rows.slice(0, 3).map((r) => (
              <div key={String(r.check)} className="rounded-xl bg-emerald-50 px-4 py-3 text-sm">
                <div className="font-semibold text-emerald-700">✓ {String(r.check).replace(/_/g, " ")}</div>
                <div className="text-emerald-800/80">{String(r.detail)}</div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </Page>
  );
}
