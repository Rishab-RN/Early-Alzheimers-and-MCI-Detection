export type Row = Record<string, string | number | boolean | null>;
export interface TableResp { source: string; synthetic: boolean; rows: Row[] }

const cache = new Map<string, Promise<unknown>>();
function get<T>(url: string): Promise<T> {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); }));
  return cache.get(url) as Promise<T>;
}

export const api = {
  table: (name: string) => get<TableResp>(`/api/table/${name}`),
  file: (folder: string, name: string) => get<TableResp>(`/api/file/${folder}/${name}`).catch(() => null),
  status: () => get<{ status: Record<string, Record<string, unknown>> }>("/api/status"),
  maps: () => get<{ names: string[] }>("/api/maps"),
};

/** Display names. Current runs are tagged proto_; the 1.5 mm pilot (proto15_) is hidden. */
export const MODEL_NAME: Record<string, string> = {
  age_only: "Age only", hv_age: "Hippocampal volume + age", mlp: "MLP (simple ANN)", ensemble_resnet3d: "3D CNN (ResNet)",
  ensemble_medicalnet: "3D CNN (MedicalNet)",
};
export const COHORT = { adni: "ADNI (internal test)", oasis3: "OASIS-3 (external)" } as Record<string, string>;
export const TASK = { cn_ad: "CN vs AD", cn_mci: "CN vs MCI" } as Record<string, string>;
export const LABELDEF = { clinical: "Clinical diagnosis", cdr: "CDR", amyloid: "Amyloid-confirmed" } as Record<string, string>;
export const METHOD = {
  msp: "Softmax confidence", energy: "Energy score", mc_dropout: "MC dropout", ensemble: "Ensemble spread",
  feature_maha: "Feature Mahalanobis", adrs: "ADRS (ours)",
} as Record<string, string>;

export function modelKey(runId: string): string | null {
  if (runId.startsWith("proto15_")) return null;
  const r = runId.replace(/^proto_/, "").replace(/_(cn_ad|cn_mci)(_bnadapt|_tent)?$/, "");
  if (/_s\d+$/.test(r)) return null;                     // single seeds hidden; ensemble rows are shown
  return r in MODEL_NAME ? r : null;
}
export const isTTA = (runId: string) => /_(bnadapt|tent)$/.test(runId);
export const fmt = (v: unknown, d = 3) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—");
