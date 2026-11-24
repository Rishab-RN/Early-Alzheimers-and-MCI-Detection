import { ArrowRight, Brain, Layers, ShieldCheck, Target } from "lucide-react";
import { Link } from "react-router-dom";
import { api, fmt, type Row } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Kpi } from "../components/ui";

const STEPS = [
  ["T1-weighted MRI", "ADNI · OASIS-3"], ["N4 + HD-BET", "bias correction, skull strip"], ["Affine → MNI152", "atrophy preserved"],
  ["3D CNN", "ResNet, ensemble-ready"], ["Calibrated risk", "AD / MCI probability"], ["Atlas attributions", "112 brain regions"], ["ADRS flag", "reliable / defer"],
];
const CONTRIB = [
  { icon: Layers, k: "C1", t: "Label-aware shift decomposition", d: "Splits the internal → external performance drop into label definition, demographics, field strength and residual scanner/population shift." },
  { icon: Brain, k: "C2", t: "Do explanations survive shift?", d: "Integrated Gradients aggregated to atlas regions; cross-cohort agreement, AD-region focus, faithfulness and sanity checks." },
  { icon: ShieldCheck, k: "C3", t: "ADRS failure detection", d: "A label-free Attribution-Drift Reliability Score that flags scans the model is likely to get wrong." },
  { icon: Target, k: "C4", t: "Honest early detection", d: "CN vs AD and CN vs MCI against age-only and hippocampal-volume baselines, with confound controls." },
];

const ROADMAP: [string, string, string[]][] = [
  ["Done", "bg-emerald-50 text-emerald-700", ["ADNI + OASIS-3 acquisition, QC and 3-way harmonised labels", "Standardised preprocessing (N4, HD-BET, affine MNI) with orientation fix",
    "Leakage-free splits; age-only and hippocampal-volume baselines", "3D CNN for CN vs AD and CN vs MCI; external OASIS-3 evaluation", "Atlas-level explanations, ADRS reliability score, scan-analysis app"]],
  ["In progress", "bg-amber-50 text-amber-700", ["Preprocessing the remaining ADNI and full OASIS-3 cohort", "Full training: 3D ResNet (5 seeds) and MedicalNet transfer model (3 seeds)",
    "Deep-ensemble uncertainty and test-time adaptation (BN-adapt, TENT)"]],
  ["Next", "bg-brand-50 text-brand-700", ["Shift decomposition with confidence intervals on the full cohorts", "Explanation-drift study and ADRS vs GAIA-style baseline",
    "AIBL as a second external cohort (access pending)", "Final report and paper write-up"]],
];

const pick = (rows: Row[], run: string, cohort: string, task: string) =>
  rows.find((r) => r.run_id === run && r.cohort === cohort && r.task === task && r.label_def === "clinical" && r.subset === "all" && r.metric === "auroc");

export default function About() {
  const m = useApi(() => api.table("metrics"));
  const st = useApi(api.status);
  const rows = m.data?.rows ?? [];
  const cnnInt = pick(rows, "proto_ensemble_resnet3d_cn_ad", "adni", "cn_ad");
  const cnnExt = pick(rows, "proto_ensemble_resnet3d_cn_ad", "oasis3", "cn_ad");
  const hv = pick(rows, "hv_age_cn_ad", "adni", "cn_ad");
  const s = st.data?.status ?? {};
  const adni = (s.adni ?? {}) as Record<string, number>;
  const oasis = (s.oasis3 ?? {}) as Record<string, number>;
  return (
    <div>
      <section className="border-b border-ink-300/50 bg-white">
        <div className="mx-auto grid max-w-7xl gap-10 px-6 py-12 lg:grid-cols-[1.4fr_1fr] lg:px-10">
          <div>
            <div className="eyebrow mb-3">About the project</div>
            <h1 className="text-4xl font-semibold leading-tight tracking-tight text-ink-900 md:text-5xl">
              Cross-cohort robust &amp; explainable deep learning for early Alzheimer's detection
            </h1>
            <p className="lead mt-5 max-w-2xl">
              Models trained on one research cohort often fail on another. We train 3D CNNs on <b>ADNI</b> and test them,
              without retraining, on the independent <b>OASIS-3</b> cohort, asking how much of the drop comes from
              <b> how each cohort labels patients</b>, whether the model's <b>explanations stay consistent</b>, and whether
              explanations can <b>flag unreliable predictions</b>.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link to="/" className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-brand-700">Analyse a scan <ArrowRight size={16} /></Link>
              <Link to="/explain" className="inline-flex items-center gap-2 rounded-xl border border-ink-300 px-4 py-2.5 text-sm font-semibold text-ink-700 hover:bg-ink-100">Explore the brain maps</Link>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 self-center">
            <Kpi label="ADNI subjects" value={(adni.selected ?? 3153).toLocaleString()} sub={`${(adni.preprocessed_ok ?? 0).toLocaleString()} preprocessed`} tone="ink" />
            <Kpi label="OASIS-3 subjects" value={(oasis.target ?? 1376).toLocaleString()} sub="independent external cohort" tone="ink" />
            <Kpi label="3D CNN · ADNI" value={fmt(cnnInt?.value)} sub={hv ? `AUROC CN vs AD · baseline ${fmt(hv.value)}` : "AUROC CN vs AD"} />
            <Kpi label="3D CNN · OASIS-3" value={fmt(cnnExt?.value)} sub="AUROC, zero-shot external" tone="ext" />
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-7xl space-y-10 px-6 py-10 lg:px-10">
        <section>
          <h2 className="h2 mb-4">Pipeline</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-7">
            {STEPS.map(([a, b], i) => (
              <div key={a} className="card relative p-4">
                <div className="text-[11px] font-semibold text-brand-600">STEP {i + 1}</div>
                <div className="mt-1 text-sm font-semibold">{a}</div>
                <div className="mt-0.5 text-xs muted">{b}</div>
              </div>
            ))}
          </div>
        </section>
        <section>
          <h2 className="h2 mb-4">What is new</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {CONTRIB.map(({ icon: Icon, k, t, d }) => (
              <div key={k} className="card flex gap-4">
                <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-brand-50 text-brand-600"><Icon size={21} /></div>
                <div><div className="text-xs font-semibold text-brand-600">{k}</div><div className="font-semibold">{t}</div><p className="mt-1 text-sm muted">{d}</p></div>
              </div>
            ))}
          </div>
        </section>
        <section>
          <h2 className="h2 mb-4">Roadmap</h2>
          <div className="grid gap-4 md:grid-cols-3">
            {ROADMAP.map(([title, tone, items]) => (
              <div key={title as string} className="card">
                <div className={`pill ${tone}`}>{title as string}</div>
                <ul className="mt-3 space-y-2 text-sm text-ink-700">{(items as string[]).map((i) => <li key={i} className="flex gap-2"><span className="muted">•</span>{i}</li>)}</ul>
              </div>
            ))}
          </div>
        </section>
        <section className="grid gap-4 md:grid-cols-3">
          {[["Leakage-free", "One scan per subject; subject-level splits; external cohort never used for training, tuning or thresholds."],
            ["Pre-registered", "Analysis plan, label definitions and model configurations fixed before any external evaluation."],
            ["Confound-controlled", "Age/sex-matched and 1.5 T vs 3 T subsets; age-only and hippocampal-volume baselines."]].map(([t, d]) => (
            <div key={t} className="card"><div className="font-semibold">{t}</div><p className="mt-1 text-sm muted">{d}</p></div>
          ))}
        </section>
      </div>
    </div>
  );
}
