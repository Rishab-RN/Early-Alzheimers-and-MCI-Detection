"""Generate paper tables/figures/numbers from results/ and data/labels (never hand-typed).

Outputs: paper/generated/numbers.tex, paper/generated/results.tex, paper/figures/*.pdf
Only aggregate statistics are written.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, GEN, FIG = ROOT / "results", ROOT / "paper/generated", ROOT / "paper/figures"
NAMES = {"age_only": "Age only", "hv_age": "HV + age", "ensemble_resnet3d": "3D ResNet (ens.)",
         "ensemble_medicalnet": "MedicalNet (ens.)", "clinical": "Clinical", "cdr": "CDR", "amyloid": "Amyloid",
         "adni": "ADNI (int.)", "oasis3": "OASIS-3 (ext.)", "cn_ad": "CN vs AD", "cn_mci": "CN vs MCI"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False})


def qc_ok(cohort: str) -> set:
    keys = set()
    for q in (ROOT / f"data/preprocessed/{cohort}").glob("qc_*.csv"):
        d = pd.read_csv(q)
        keys |= set(d[(d.status == "ok") & (d.reg_corr >= 0.3)].key)
    return keys


def cohort_table() -> tuple[str, dict]:
    rows, nums = [], {}
    for c, lab in (("adni", "ADNI"), ("oasis3", "OASIS-3")):
        d = pd.read_csv(ROOT / f"data/labels/{c}_labels.csv")
        d = d[d.dx_clinical.isin(["CN", "MCI", "AD"])]
        nums[c] = len(d)
        for g in ("CN", "MCI", "AD"):
            s = d[d.dx_clinical == g]
            rows.append(f"{lab} & {g} & {len(s)} & {s.age.mean():.1f} $\\pm$ {s.age.std():.1f} & "
                        f"{100 * (s.sex == 'F').mean():.0f} & {100 * (s.field_strength >= 2.9).mean():.0f} & "
                        f"{s.mmse.mean():.1f} & {100 * s.amyloid_pos.mean():.0f} ({s.amyloid_pos.notna().sum()}) \\\\")
        rows.append("\\midrule")
    tex = ("\\begin{table}[t]\\centering\\caption{Cohort characteristics (clinical diagnosis, one scan per subject).}"
           "\\label{tab:cohorts}\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llrrrrrr}\\toprule "
           "Cohort & Dx & n & Age & F\\% & 3T\\% & MMSE & A+\\% (n) \\\\ \\midrule " + " ".join(rows[:-1]) +
           "\\bottomrule\\end{tabular}}\\end{table}")
    return tex, nums


def auroc_table(m: pd.DataFrame, models: list[str]) -> str:
    lines = []
    for task in ("cn_ad", "cn_mci"):
        for model in models:
            rid = f"{model}_{task}" if not model.startswith("ensemble") else f"{model}_{task}"
            cells = []
            for c in ("adni", "oasis3"):
                for ld in ("clinical", "cdr", "amyloid"):
                    r = m[(m.run_id == rid) & (m.cohort == c) & (m.label_def == ld) & (m.subset == "all") & (m.metric == "auroc")]
                    cells.append(f"{r.value.iloc[0]:.3f}" if len(r) else "--")
            lines.append(f"{NAMES[task]} & {NAMES.get(model, model)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    return ("\\begin{table*}[t]\\centering\\caption{AUROC on ADNI (internal test) and OASIS-3 (external) under the three "
            "label definitions. Models are trained on ADNI clinical diagnosis only.}\\label{tab:auroc}"
            "\\begin{tabular}{llcccccc}\\toprule & & \\multicolumn{3}{c}{ADNI} & \\multicolumn{3}{c}{OASIS-3} \\\\ "
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8} Task & Model & Clin. & CDR & Amyl. & Clin. & CDR & Amyl. \\\\ \\midrule "
            + " ".join(lines[:-1]) + "\\bottomrule\\end{tabular}\\end{table*}")


def fig_label_shift(m: pd.DataFrame, model: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8), sharey=True)
    for ax, task in zip(axes, ("cn_ad", "cn_mci")):
        x = np.arange(3)
        for k, (c, col) in enumerate((("adni", "#4f46e5"), ("oasis3", "#f59e0b"))):
            v, lo, hi = [], [], []
            for ld in ("clinical", "cdr", "amyloid"):
                r = m[(m.run_id == f"{model}_{task}") & (m.cohort == c) & (m.label_def == ld) & (m.subset == "all") & (m.metric == "auroc")]
                v.append(r.value.iloc[0] if len(r) else np.nan); lo.append(r.ci_low.iloc[0] if len(r) else np.nan); hi.append(r.ci_high.iloc[0] if len(r) else np.nan)
            v, lo, hi = map(np.array, (v, lo, hi))
            ax.bar(x + (k - .5) * .38, v, .36, yerr=[v - lo, hi - v], color=col, label=NAMES[c], capsize=1.5, error_kw={"lw": .6})
        ax.set_xticks(x, ["Clin.", "CDR", "Amyl."]); ax.set_title(NAMES[task]); ax.set_ylim(.5, 1)
    axes[0].set_ylabel("AUROC"); axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / f"label_shift_{model}.pdf"); plt.close(fig)


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    cohort_tex, nums = cohort_table()
    (GEN / "numbers.tex").write_text(
        f"\\newcommand{{\\ADNIn}}{{{nums['adni']:,}}}\n\\newcommand{{\\OASISn}}{{{nums['oasis3']:,}}}\n", encoding="utf-8")
    parts = [cohort_tex]
    m = pd.read_csv(RES / "metrics.csv") if (RES / "metrics.csv").exists() else pd.DataFrame()
    if len(m):
        models = [x for x in ("age_only", "hv_age", "ensemble_resnet3d", "ensemble_medicalnet")
                  if (m.run_id.str.startswith(x)).any()]
        parts.append(auroc_table(m, models))
        for model in models:
            fig_label_shift(m, model)
        parts.append("\\begin{figure}[t]\\centering\\includegraphics[width=\\columnwidth]{label_shift_" + models[-1] +
                     ".pdf}\\caption{Same frozen model, three label definitions: AUROC with 95\\% bootstrap CIs.}"
                     "\\label{fig:labelshift}\\end{figure}")
    (GEN / "results.tex").write_text("\n\n".join(parts), encoding="utf-8")
    print("wrote", GEN / "results.tex", "and figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
