import type { ReactNode } from "react";
import { Clock3, Info } from "lucide-react";

export function Page({ eyebrow, title, lead, children }: { eyebrow?: string; title: string; lead?: ReactNode; children: ReactNode }) {
  return (
    <div className="mx-auto max-w-7xl space-y-6 px-6 py-8 lg:px-10">
      <header className="max-w-4xl">
        {eyebrow && <div className="eyebrow mb-2">{eyebrow}</div>}
        <h1 className="h1">{title}</h1>
        {lead && <p className="lead mt-3">{lead}</p>}
      </header>
      {children}
    </div>
  );
}

export function Kpi({ label, value, sub, tone = "brand" }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "brand" | "ext" | "ink" | "green" }) {
  const c = { brand: "text-brand-600", ext: "text-ext-500", ink: "text-ink-900", green: "text-emerald-600" }[tone];
  return (
    <div className="card">
      <div className="text-sm font-medium muted">{label}</div>
      <div className={`mt-2 text-4xl font-semibold tabular-nums tracking-tight ${c}`}>{value}</div>
      {sub && <div className="mt-2 text-sm muted">{sub}</div>}
    </div>
  );
}

export function Pending({ what }: { what: string }) {
  return (
    <div className="card flex items-center gap-3 border-dashed text-sm muted">
      <Clock3 size={18} className="shrink-0 text-ink-300" />
      <span><b className="text-ink-700">{what}</b>: being computed by the pipeline; this panel fills in automatically when the result file appears.</span>
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-xl bg-brand-50 px-4 py-3 text-sm text-brand-700">
      <Info size={17} className="mt-0.5 shrink-0" /><div>{children}</div>
    </div>
  );
}

export function Seg<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: [T, string][] }) {
  return (
    <div className="seg">
      {options.map(([v, l]) => <button key={v} data-on={v === value} onClick={() => onChange(v)}>{l}</button>)}
    </div>
  );
}
