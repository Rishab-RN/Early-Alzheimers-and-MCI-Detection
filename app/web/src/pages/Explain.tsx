import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import { api, fmt, type Row } from "../lib/api";
import { useApi } from "../lib/useApi";
import { C, Chart } from "../components/Chart";
import { Kpi, Note, Page, Pending, Seg } from "../components/ui";

type Task = "cn_ad" | "cn_mci";
type Coh = "adni" | "oasis3";

function BrainViewer({ mapName }: { mapName: string | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const nv = useRef<Niivue | null>(null);
  const [view, setView] = useState<"multi" | "render">("multi");
  useEffect(() => {
    if (!canvas.current) return;
    if (!nv.current) {
      nv.current = new Niivue({ backColor: [1, 1, 1, 1], show3Dcrosshair: false, isColorbar: true, crosshairColor: [0.31, 0.27, 0.9, 0.6] });
      nv.current.attachToCanvas(canvas.current);
    }
    const vols: { url: string; name: string; colormap: string; opacity?: number; cal_min?: number; cal_max?: number }[] =
      [{ url: "/api/template", name: "mni.nii.gz", colormap: "gray" }];
    if (mapName) vols.push({ url: `/api/map/${mapName}`, name: `${mapName}.nii.gz`, colormap: "warm", opacity: 0.75, cal_min: 0.25, cal_max: 1 });
    nv.current.loadVolumes(vols).then(() => nv.current?.setSliceType(view === "multi" ? 3 : 4));
  }, [mapName]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { nv.current?.setSliceType(view === "multi" ? 3 : 4); }, [view]);
  return (
    <div className="card p-3">
      <div className="mb-2 flex items-center justify-between px-2">
        <div className="text-sm font-semibold">Group-average attribution on MNI template</div>
        <Seg value={view} onChange={setView} options={[["multi", "Slices"], ["render", "3D"]]} />
      </div>
      <div style={{ position: "relative", height: 460 }}><canvas ref={canvas} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", borderRadius: 12 }} /></div>
      {!mapName && <div className="px-2 pt-2 text-sm muted">Attribution map pending; showing the template only.</div>}
    </div>
  );
}

export default function Explain() {
  const [task, setTask] = useState<Task>("cn_ad");
  const [coh, setCoh] = useState<Coh>("adni");
  const [meth, setMeth] = useState<"ig" | "gradcam">("ig");
  const run = `proto_resnet3d_${task}_s0`;
  const maps = useApi(api.maps);
  const roi = useApi(() => Promise.all((["adni", "oasis3"] as Coh[]).map((c) => api.file("roi_attribution", `${run}_${c}${meth === "gradcam" ? "_gradcam" : ""}`))), [task, meth]);
  const xs = useApi(() => api.table("xai_stats"));
  const mapName = maps.data?.names.find((n) => n === `${run}_${coh}_mean_${meth}`) ?? null;
  const stats = (xs.data?.rows ?? []).filter((r) => r.run_id === run);
  const st = (cmp: string, metric: string) => stats.find((r) => r.comparison === cmp && r.metric === metric);
  const [a, o] = roi.data ?? [null, null];
  const top = (a?.rows ?? []).slice(0, 15);
  const oByRoi = new Map((o?.rows ?? []).map((r) => [r.roi, r]));
  const bars = [
    { type: "bar", orientation: "h", name: "ADNI", y: top.map((r) => r.roi), x: top.map((r) => r.mean), marker: { color: top.map((r) => (String(r.ad_prior) === "true" || r.ad_prior === true ? C.int : "#a5b4fc")) } },
    ...(o ? [{ type: "bar", orientation: "h", name: "OASIS-3", y: top.map((r) => r.roi), x: top.map((r) => (oByRoi.get(r.roi) as Row | undefined)?.mean ?? null), marker: { color: C.ext } }] : []),
  ];
  return (
    <Page eyebrow="Explainability (C2)" title="What does the model look at, and does it stay the same on a new cohort?"
      lead="Integrated Gradients and Grad-CAM attributions are averaged over subjects and aggregated to 112 Harvard-Oxford brain regions, so explanations can be compared with numbers across cohorts instead of eyeballing individual heat-maps.">
      <div className="flex flex-wrap items-center gap-4">
        <Seg value={task} onChange={setTask} options={[["cn_ad", "CN vs AD"], ["cn_mci", "CN vs MCI"]]} />
        <Seg value={coh} onChange={setCoh} options={[["adni", "ADNI"], ["oasis3", "OASIS-3"]]} />
        <Seg value={meth} onChange={setMeth} options={[["ig", "Integrated Gradients"], ["gradcam", "Grad-CAM"]]} />
      </div>
      <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
        <BrainViewer mapName={mapName} />
        <div className="grid grid-cols-2 content-start gap-4">
          <Kpi label="AD-region focus (ADNI)" value={fmt(st("adni", "ad_prior_mass")?.value, 2)} sub="share of attribution in a-priori AD regions (chance ≈ 0.23)" />
          <Kpi label="AD-region focus (OASIS-3)" value={fmt(st("oasis3", "ad_prior_mass")?.value, 2)} sub="same measure on the external cohort" tone="ext" />
          <Kpi label="Cross-cohort agreement" value={fmt(st("adni_vs_oasis3", "spearman")?.value, 2)} sub={`Spearman of region profiles (internal split-half ${fmt(st("adni_split_half", "spearman")?.value, 2)})`} tone="ink" />
          <Kpi label="Sanity check" value={fmt(st("adni", "randomization_sim")?.value, 2)} sub="similarity to a randomised model (lower = explanations depend on learned weights)" tone="green" />
        </div>
      </div>
      {a ? (
        <Chart title="Top 15 regions by attribution" subtitle="Dark indigo = a-priori Alzheimer's regions (hippocampus, amygdala, parahippocampal, temporal)"
          data={bars as Plotly.Data[]} height={520} layout={{ barmode: "group", margin: { l: 280, r: 20, t: 10, b: 50 }, yaxis: { autorange: "reversed" }, xaxis: { title: { text: "share of attribution (density-normalised)" } } }}
          source="results/roi_attribution" />
      ) : <Pending what="Region attributions for this model" />}
      {st("adni", "deletion_auc_attr") && (
        <Note>Faithfulness: removing the regions the model highlights lowers its confidence faster (deletion AUC {fmt(st("adni", "deletion_auc_attr")?.value, 2)}) than removing random regions ({fmt(st("adni", "deletion_auc_random")?.value, 2)}), so the highlighted regions genuinely drive the prediction.</Note>
      )}
    </Page>
  );
}
