import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import { AlertTriangle, CheckCircle2, FileUp, Loader2, RotateCcw, ShieldQuestion, UploadCloud } from "lucide-react";
import { Seg } from "../components/ui";

interface Region { roi: string; share: number; ad_region: boolean }
interface TaskRes { negative: string; positive: string; probability: number; threshold: number; prediction: string;
  reliability: { available: boolean; reliable?: boolean; adrs?: number; cutoff?: number }; top_regions: Region[] }
interface Job { step: string; pct: number; done?: boolean; error?: string; result?: { tasks: Record<string, TaskRes>; seconds: number } }
interface Example { id: string; label: string; truth: string; age: number; sex: string }

const STEPS = ["Upload", "Preprocess", "Predict", "Explain", "Report"];

function Viewer({ job, task }: { job: string; task: string }) {
  const [method, setMethod] = useState<"attr" | "gradcam">("attr");
  const canvas = useRef<HTMLCanvasElement>(null);
  const nv = useRef<Niivue | null>(null);
  const [mode, setMode] = useState<"multi" | "axial" | "render">("multi");
  useEffect(() => {
    if (!canvas.current) return;
    if (!nv.current) { nv.current = new Niivue({ backColor: [0.04, 0.06, 0.12, 1], isColorbar: false, crosshairColor: [0.4, 0.45, 1, 0.5] }); nv.current.attachToCanvas(canvas.current); }
    nv.current.loadVolumes([
      { url: `/api/job/${job}/brain_disp`, name: "brain.nii.gz", colormap: "gray" },
      { url: `/api/job/${job}/${method}_${task}`, name: `${method}.nii.gz`, colormap: "warm", opacity: 0.7, cal_min: 0.3, cal_max: 1 },
    ]).then(() => nv.current?.setSliceType(mode === "multi" ? 3 : mode === "axial" ? 0 : 4));
  }, [job, task, method]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { nv.current?.setSliceType(mode === "multi" ? 3 : mode === "axial" ? 0 : 4); }, [mode]);
  return (
    <div className="card overflow-hidden p-0">
      <div className="flex items-center justify-between border-b border-ink-100 px-5 py-3">
        <div><div className="text-sm font-semibold">Where the model looked</div><div className="text-xs muted">Warm colours = regions that most influenced the decision</div></div>
        <div className="flex gap-2">
          <Seg value={method} onChange={setMethod} options={[["attr", "Integrated Gradients"], ["gradcam", "Grad-CAM"]]} />
          <Seg value={mode} onChange={setMode} options={[["multi", "3 planes"], ["axial", "Axial"], ["render", "3D"]]} />
        </div>
      </div>
      <div style={{ position: "relative", height: 440 }}><canvas ref={canvas} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} /></div>
    </div>
  );
}

function RiskCard({ t, active, onClick }: { t: TaskRes; active: boolean; onClick: () => void }) {
  const pct = Math.round(t.probability * 100);
  const pos = t.prediction === t.positive;
  const rel = t.reliability;
  return (
    <button onClick={onClick} className={`card w-full text-left transition ${active ? "ring-2 ring-brand-500" : "hover:border-brand-100"}`}>
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold muted">{t.negative} vs {t.positive}</div>
        {rel.available ? (rel.reliable
          ? <span className="pill bg-emerald-50 text-emerald-700"><CheckCircle2 size={13} /> Reliable</span>
          : <span className="pill bg-amber-50 text-amber-700"><AlertTriangle size={13} /> Refer for review</span>)
          : <span className="pill bg-ink-100 text-ink-500"><ShieldQuestion size={13} /> Reliability pending</span>}
      </div>
      <div className="mt-3 flex items-end gap-3">
        <div className={`text-4xl font-semibold tabular-nums ${pos ? "text-rose-600" : "text-emerald-600"}`}>{pct}%</div>
        <div className="pb-1 text-sm muted">probability of <b className="text-ink-700">{t.positive}</b></div>
      </div>
      <div className="relative mt-3 h-2.5 rounded-full bg-gradient-to-r from-emerald-200 via-amber-200 to-rose-300">
        <div className="absolute -top-1 h-4.5 w-1.5 rounded-full bg-ink-900" style={{ left: `calc(${pct}% - 3px)` }} />
        <div className="absolute -top-2 h-6 w-px bg-ink-500/60" style={{ left: `${Math.round(t.threshold * 100)}%` }} title="decision threshold" />
      </div>
      <div className="mt-3 text-sm">Model output: <b className={pos ? "text-rose-600" : "text-emerald-600"}>{pos ? `consistent with ${t.positive}` : `consistent with ${t.negative}`}</b></div>
    </button>
  );
}

export default function Analyze() {
  const [job, setJob] = useState<string | null>(null);
  const [st, setSt] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [task, setTask] = useState("cn_ad");
  const [examples, setExamples] = useState<Example[]>([]);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => { fetch("/api/examples").then((r) => r.json()).then(setExamples).catch(() => {}); }, []);
  useEffect(() => {
    if (!job) return;
    const t = setInterval(async () => {
      const r = await fetch(`/api/job/${job}`); const j: Job = await r.json(); setSt(j);
      if (j.done) clearInterval(t);
    }, 1200);
    return () => clearInterval(t);
  }, [job]);
  const start = async (p: Promise<Response>) => {
    setErr(null); setSt({ step: "Uploading", pct: 1 });
    const r = await p; const j = await r.json();
    if (!r.ok) { setErr(j.detail ?? "Upload failed"); setSt(null); return; }
    setJob(j.job);
  };
  const upload = (f: File) => { const fd = new FormData(); fd.append("file", f); start(fetch("/api/analyze", { method: "POST", body: fd })); };
  const reset = () => { setJob(null); setSt(null); setErr(null); };
  const res = st?.result;
  const stepIdx = !st ? 0 : st.done ? 5 : st.pct < 10 ? 1 : st.pct < 45 ? 2 : st.pct < 70 ? 3 : 4;
  const ex = examples.find((e) => job && st && e);

  return (
    <div className="mx-auto max-w-7xl px-6 py-8 lg:px-10">
      {!st && (
        <div className="grid gap-8 lg:grid-cols-[1.15fr_1fr]">
          <div>
            <div className="eyebrow mb-2">Scan analysis</div>
            <h1 className="h1">Analyse a brain MRI for signs of Alzheimer's disease</h1>
            <p className="lead mt-3 max-w-xl">Upload a T1-weighted MRI. The app standardises it, estimates the probability of Alzheimer's disease and of mild cognitive impairment, shows which brain regions drove the result, and tells you when the result should not be trusted.</p>
            <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) upload(f); }}
              onClick={() => fileRef.current?.click()}
              className={`mt-7 flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition ${drag ? "border-brand-500 bg-brand-50" : "border-ink-300 bg-white hover:border-brand-500"}`}>
              <UploadCloud size={36} className="text-brand-600" />
              <div className="mt-3 font-semibold">Drop a NIfTI file here or click to browse</div>
              <div className="mt-1 text-sm muted">.nii or .nii.gz · T1-weighted structural MRI · processed locally on this computer</div>
              <input ref={fileRef} type="file" accept=".nii,.gz" className="hidden" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
            </div>
            {err && <div className="mt-3 rounded-xl bg-rose-50 px-4 py-3 text-sm text-rose-700">{err}</div>}
          </div>
          <div className="lg:pt-16">
            <div className="text-sm font-semibold">Or try an example scan</div>
            <p className="mt-1 text-sm muted">Held-out ADNI test subjects (never seen during training), diagnosis shown for comparison.</p>
            <div className="mt-4 space-y-3">
              {examples.map((e) => (
                <button key={e.id} onClick={() => start(fetch(`/api/analyze/example/${e.id}`, { method: "POST" }))}
                  className="card flex w-full items-center gap-4 p-4 text-left hover:border-brand-500">
                  <div className="grid h-11 w-11 place-items-center rounded-xl bg-brand-50 text-brand-600"><FileUp size={19} /></div>
                  <div className="flex-1"><div className="font-semibold">{e.label}</div><div className="text-sm muted">{e.sex === "F" ? "Female" : "Male"}, {e.age} y</div></div>
                  <span className="pill bg-ink-100 text-ink-700">Clinical diagnosis: {e.truth}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {st && (
        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {!st.done ? <Loader2 className="animate-spin text-brand-600" size={22} /> : st.error ? <AlertTriangle className="text-rose-600" size={22} /> : <CheckCircle2 className="text-emerald-600" size={22} />}
                <div><div className="font-semibold">{st.error ? "Analysis failed" : st.done ? "Analysis complete" : st.step}</div>
                  <div className="text-sm muted">{st.error ?? (st.done && res ? `Finished in ${res.seconds}s` : "This usually takes under a minute")}</div></div>
              </div>
              <button onClick={reset} className="inline-flex items-center gap-2 rounded-xl border border-ink-300 px-3 py-2 text-sm font-medium hover:bg-ink-100"><RotateCcw size={15} /> New scan</button>
            </div>
            <div className="mt-5 grid grid-cols-5 gap-2">
              {STEPS.map((s, i) => (
                <div key={s}><div className={`h-1.5 rounded-full ${i < stepIdx ? "bg-brand-600" : i === stepIdx ? "bg-brand-100" : "bg-ink-100"}`} />
                  <div className={`mt-1.5 text-xs font-medium ${i <= stepIdx ? "text-ink-700" : "text-ink-300"}`}>{s}</div></div>
              ))}
            </div>
          </div>
          {res && job && (
            <>
              <div className="grid gap-4 md:grid-cols-2">
                {Object.entries(res.tasks).map(([k, t]) => <RiskCard key={k} t={t} active={k === task} onClick={() => setTask(k)} />)}
              </div>
              <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
                <Viewer job={job} task={res.tasks[task] ? task : Object.keys(res.tasks)[0]} />
                <div className="card">
                  <div className="text-sm font-semibold">Most influential regions</div>
                  <div className="text-xs muted">{res.tasks[task]?.negative} vs {res.tasks[task]?.positive} model</div>
                  <div className="mt-4 space-y-3">
                    {(res.tasks[task] ?? Object.values(res.tasks)[0]).top_regions.map((r) => {
                      const max = (res.tasks[task] ?? Object.values(res.tasks)[0]).top_regions[0].share;
                      return (
                        <div key={r.roi}>
                          <div className="flex items-center justify-between text-sm"><span className="truncate pr-2">{r.roi}</span>
                            {r.ad_region && <span className="pill shrink-0 bg-brand-50 text-brand-700">AD region</span>}</div>
                          <div className="mt-1 h-2 rounded-full bg-ink-100"><div className="h-2 rounded-full bg-brand-500" style={{ width: `${(100 * r.share) / max}%` }} /></div>
                        </div>
                      );
                    })}
                  </div>
                  <p className="mt-5 text-xs muted">"AD region": hippocampus, amygdala, parahippocampal, entorhinal and temporal cortex, regions known to shrink early in Alzheimer's disease.</p>
                </div>
              </div>
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                Research software, not a medical device. Results are model estimates from research cohorts and must not be used for diagnosis or treatment decisions.
              </div>
            </>
          )}
          {ex === undefined && null}
        </div>
      )}
    </div>
  );
}
