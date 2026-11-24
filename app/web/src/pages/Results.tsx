import { useState } from "react";
import { api, COHORT, fmt, LABELDEF, MODEL_NAME, modelKey, type Row, type TableResp } from "../lib/api";
import { useApi } from "../lib/useApi";
import { C, Chart } from "../components/Chart";
import { Note, Page, Pending, Seg } from "../components/ui";

type Task = "cn_ad" | "cn_mci";
type Def = "clinical" | "cdr" | "amyloid";
const ORDER = ["age_only", "hv_age", "mlp", "ensemble_resnet3d", "ensemble_medicalnet"];
const METRICS: [string, string][] = [["auroc", "AUROC"], ["bal_acc", "Bal. acc."], ["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["ece", "ECE ↓"]];

function runFor(rows: Row[], model: string, task: Task) {
  const ids = [...new Set(rows.map((r) => String(r.run_id)))].filter((id) => modelKey(id) === model && id.includes(task) && !/_(bnadapt|tent)$/.test(id));
  return ids.find((id) => id.startsWith("proto_")) ?? ids[0];
}

export default function Results() {
  const [task, setTask] = useState<Task>("cn_ad");
  const [def, setDef] = useState<Def>("clinical");
  const m = useApi(() => api.table("metrics"));
  const cnn = `proto_ensemble_resnet3d_${task}`;
  const curves = useApi(() => Promise.all(["adni", "oasis3"].map((c) => api.file("curves", `${cnn}_${c}_${def}_roc`))), [task, def]);
  const conf = useApi(() => Promise.all(["adni", "oasis3"].map((c) => api.file("confusion", `${cnn}_${c}_${def}`))), [task, def]);
  const tc = useApi(async () => (await api.file("training_curves", `proto_resnet3d_${task}_s1`)) ?? api.file("training_curves", `proto_resnet3d_${task}_s0`), [task]);
  if (!m.data) return <Page title="Results"><Pending what="Metrics" /></Page>;
  const rows = m.data.rows;
  const models = ORDER.filter((k) => runFor(rows, k, task));
  const get = (run: string | undefined, cohort: string, metric = "auroc", subset = "all", d: Def = def) =>
    rows.find((r) => r.run_id === run && r.cohort === cohort && r.label_def === d && r.subset === subset && r.metric === metric);
  const bars = (["adni", "oasis3"] as const).map((c) => {
    const rr = models.map((k) => get(runFor(rows, k, task), c));
    return { type: "bar", name: COHORT[c], x: models.map((k) => MODEL_NAME[k]), y: rr.map((r) => r?.value ?? null),
      marker: { color: c === "adni" ? C.int : C.ext }, text: rr.map((r) => (r ? fmt(r.value, 2) : "")), textposition: "outside",
      error_y: { type: "data", symmetric: false, color: "#94a3b8", array: rr.map((r) => (r ? Number(r.ci_high) - Number(r.value) : 0)), arrayminus: rr.map((r) => (r ? Number(r.value) - Number(r.ci_low) : 0)) } };
  });
  const defBars = (["adni", "oasis3"] as const).map((c) => ({
    type: "bar", name: COHORT[c], x: Object.values(LABELDEF), marker: { color: c === "adni" ? C.int : C.ext },
    y: (["clinical", "cdr", "amyloid"] as Def[]).map((d) => get(cnn, c, "auroc", "all", d)?.value ?? null) }));
  const tta = ["", "_bnadapt"].map((t) => get(`${cnn}${t}`, "oasis3"));
  const cnnExists = rows.some((r) => r.run_id === cnn);
  const roc = (d: (TableResp | null)[] | null) => (d ?? []).map((x, i) => x && ({ type: "scatter", mode: "lines", name: COHORT[["adni", "oasis3"][i]],
    x: x.rows.map((r) => r.x), y: x.rows.map((r) => r.y), line: { width: 3, color: i ? C.ext : C.int } })).filter(Boolean);
  return (
    <Page eyebrow="Results" title="How well does it detect AD and MCI, inside and outside ADNI?"
      lead="Every model is trained on ADNI clinical diagnosis only, then evaluated on held-out ADNI subjects and, without any retraining, on OASIS-3. Bars show AUROC with 95% bootstrap confidence intervals.">
      <div className="flex flex-wrap items-center gap-4">
        <Seg value={task} onChange={setTask} options={[["cn_ad", "CN vs AD"], ["cn_mci", "CN vs MCI (early)"]]} />
        <Seg value={def} onChange={setDef} options={[["clinical", "Clinical labels"], ["cdr", "CDR labels"], ["amyloid", "Amyloid-confirmed"]]} />
      </div>
      <Chart title="Model comparison" subtitle={`${LABELDEF[def]} labels · ${task === "cn_ad" ? "CN vs AD" : "CN vs MCI"}`} data={bars as Plotly.Data[]} height={380}
        layout={{ barmode: "group", yaxis: { title: { text: "AUROC" }, range: [0.4, 1.05] } }} source="results/metrics.csv" />
      {!cnnExists && <Pending what="3D CNN results for this task" />}
      {cnnExists && (
        <>
          <div className="card overflow-x-auto">
            <h3 className="h2 mb-3">All metrics: {LABELDEF[def]} labels</h3>
            <table className="data">
              <thead><tr><th>Model</th><th>Cohort</th>{METRICS.map(([, l]) => <th key={l}>{l}</th>)}<th>n</th></tr></thead>
              <tbody>{models.flatMap((k) => (["adni", "oasis3"] as const).map((c) => {
                const run = runFor(rows, k, task);
                return (<tr key={k + c}><td className="font-medium">{MODEL_NAME[k]}</td><td>{COHORT[c]}</td>
                  {METRICS.map(([mk]) => { const r = get(run, c, mk); return <td key={mk}>{fmt(r?.value)}<div className="text-[11px] muted">{r ? `${fmt(r.ci_low, 2)}–${fmt(r.ci_high, 2)}` : ""}</div></td>; })}
                  <td>{get(run, c)?.n ?? "—"}</td></tr>);
              }))}</tbody>
            </table>
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <Chart title="ROC curve: 3D CNN" subtitle="Internal vs external" data={[...roc(curves.data), { type: "scatter", mode: "lines", name: "chance", x: [0, 1], y: [0, 1], line: { dash: "dot", color: C.grey } }] as Plotly.Data[]}
              layout={{ xaxis: { title: { text: "False positive rate" } }, yaxis: { title: { text: "True positive rate" } } }} source="results/curves" />
            <Chart title="Same model, three label definitions" subtitle="How much AUROC depends on how the cohort labels patients" data={defBars as Plotly.Data[]}
              layout={{ barmode: "group", yaxis: { title: { text: "AUROC" }, range: [0.4, 1] } }} source="results/metrics.csv" />
          </div>
          <div className="grid gap-6 lg:grid-cols-3">
            {(conf.data ?? []).map((d, i) => d && (
              <div key={i} className="card">
                <h3 className="h2 mb-3">Confusion matrix: {COHORT[["adni", "oasis3"][i]]}</h3>
                <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center text-sm">
                  <div />{["Pred. negative", "Pred. positive"].map((h) => <div key={h} className="text-xs font-semibold muted">{h}</div>)}
                  {["Neg", "Pos"].map((t) => [<div key={t} className="self-center pr-2 text-right text-xs font-semibold muted">{t === "Neg" ? "True CN" : `True ${task === "cn_ad" ? "AD" : "MCI"}`}</div>,
                    ...["Neg", "Pos"].map((p) => { const v = Number(d.rows.find((r) => r.true === t && r.pred === p)?.count ?? 0);
                      return <div key={t + p} className={`rounded-lg py-4 text-xl font-semibold tabular-nums ${t === p ? "bg-brand-50 text-brand-700" : "bg-ink-100 text-ink-700"}`}>{v}</div>; })])}
                </div>
              </div>
            ))}
            <div className="card">
              <h3 className="h2">Test-time adaptation</h3>
              <p className="mt-1 text-sm muted">OASIS-3 AUROC before and after re-estimating batch-norm statistics on the unlabelled external scans.</p>
              <div className="mt-5 space-y-3">
                {[["No adaptation", tta[0]], ["BN-adapt", tta[1]]].map(([l, r]) => (
                  <div key={String(l)} className="flex items-center justify-between rounded-xl bg-ink-100 px-4 py-3">
                    <span className="text-sm font-medium">{String(l)}</span><span className="text-xl font-semibold tabular-nums text-ext-500">{fmt((r as Row | undefined)?.value)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          {tc.data && (
            <div className="grid gap-6 lg:grid-cols-2">
              <Chart title="Training curves: loss" subtitle="3D CNN, early stopping on ADNI validation AUROC" data={[
                { type: "scatter", mode: "lines", name: "training loss", x: tc.data.rows.map((r) => r.epoch), y: tc.data.rows.map((r) => r.train_loss), line: { color: C.int, width: 3 } },
                { type: "scatter", mode: "lines", name: "validation loss", x: tc.data.rows.map((r) => r.epoch), y: tc.data.rows.map((r) => r.val_loss), line: { color: C.ext, width: 3 } },
              ] as Plotly.Data[]} layout={{ xaxis: { title: { text: "epoch" } }, yaxis: { title: { text: "loss" }, range: [0, 1.5] } }} source="results/training_curves" />
              <Chart title="Training curves: accuracy" subtitle={tc.data.rows[0]?.train_acc == null ? "Accuracy is logged from the second training seed onwards" : "Training vs validation accuracy, plus validation AUROC"} data={[
                { type: "scatter", mode: "lines", name: "training accuracy", x: tc.data.rows.map((r) => r.epoch), y: tc.data.rows.map((r) => r.train_acc ?? null), line: { color: C.int, width: 3 } },
                { type: "scatter", mode: "lines", name: "validation accuracy", x: tc.data.rows.map((r) => r.epoch), y: tc.data.rows.map((r) => r.val_acc ?? null), line: { color: C.ext, width: 3 } },
                { type: "scatter", mode: "lines", name: "validation AUROC", x: tc.data.rows.map((r) => r.epoch), y: tc.data.rows.map((r) => r.val_auc), line: { color: C.green, dash: "dash" } },
              ] as Plotly.Data[]} layout={{ xaxis: { title: { text: "epoch" } }, yaxis: { title: { text: "accuracy / AUROC" }, range: [0.4, 1] } }} source="results/training_curves" />
            </div>
          )}
        </>
      )}
      <Note>Current models are trained on 2,019 ADNI subjects and evaluated on a stratified OASIS-3 set; the full study adds the remaining scans, a 5-seed ensemble and a transfer-learning architecture.</Note>
    </Page>
  );
}
