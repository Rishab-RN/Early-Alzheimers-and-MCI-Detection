import PlotlyLib from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { Download } from "lucide-react";
import { useRef, type ReactNode } from "react";

const Plot = createPlotlyComponent(PlotlyLib);
export const C = { int: "#4f46e5", ext: "#f59e0b", grey: "#94a3b8", green: "#10b981", red: "#ef4444", sky: "#0ea5e9" };

interface Props {
  title: string;
  subtitle?: ReactNode;
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  height?: number;
  source?: string;
}

export function Chart({ title, subtitle, data, layout = {}, height = 340, source }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const exportFig = () => {
    const gd = ref.current?.querySelector(".js-plotly-plot") as HTMLElement | null;
    if (gd) PlotlyLib.downloadImage(gd, { format: "png", filename: title.toLowerCase().replace(/[^a-z0-9]+/g, "_"), width: 1400, height: Math.round(height * 1.6) });
  };
  return (
    <div className="card" ref={ref}>
      <div className="mb-1 flex items-start justify-between gap-3">
        <div>
          <h3 className="h2">{title}</h3>
          {subtitle && <p className="mt-0.5 text-sm muted">{subtitle}</p>}
        </div>
        <button onClick={exportFig} title="Download PNG" className="rounded-lg p-1.5 text-ink-300 hover:bg-ink-100 hover:text-ink-700">
          <Download size={17} />
        </button>
      </div>
      <Plot data={data} useResizeHandler style={{ width: "100%", height }}
        layout={{
          autosize: true, height, margin: { l: 56, r: 16, t: 12, b: 80 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "Segoe UI, system-ui, sans-serif", size: 13, color: "#334155" },
          legend: { orientation: "h", y: -0.32, font: { size: 12 } }, hoverlabel: { font: { size: 13 } },
          ...layout,
          xaxis: { gridcolor: "#eef2f7", zerolinecolor: "#e2e8f0", linecolor: "#cbd5e1", ...(layout.xaxis ?? {}) },
          yaxis: { gridcolor: "#eef2f7", zerolinecolor: "#e2e8f0", linecolor: "#cbd5e1", ...(layout.yaxis ?? {}) },
        } as Partial<Plotly.Layout>}
        config={{ displaylogo: false, responsive: true, displayModeBar: false }} />
      {source && <div className="mt-2 text-[11px] text-ink-300">Source: {source}</div>}
    </div>
  );
}
