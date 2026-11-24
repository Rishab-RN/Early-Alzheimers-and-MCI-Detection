"""Harmonised labels + covariates for every cohort (PROTOCOL.md "Labels").

Writes data/labels/<cohort>_labels.csv (subject-level, git-ignored) with columns:
  key, subject, cohort, age, sex, field_strength, manufacturer,
  dx_clinical, dx_cdr, dx_amyloid, cdr, mmse, centiloid, amyloid_pos,
  hv_l, hv_r, icv, conv36
and an aggregate-only summary results/cohort_summary.csv + results/label_overlap.csv.

Label definitions (identical across cohorts):
  dx_clinical : CN / MCI / AD (ADNI DXSUM; OASIS-3 UDSd1 with AD-etiology dementia only)
  dx_cdr      : CDR 0 -> CN, 0.5 -> MCI, >=1 -> AD
  dx_amyloid  : dx_clinical restricted to biomarker-concordant cases (CN&A-, MCI/AD&A+)
  amyloid_pos : Centiloid >= 20 (PET, nearest within +/-3 y); ADNI falls back to CSF A+ class
  conv36      : for clinical MCI: 'pMCI' if AD within 36 months, 'sMCI' if followed >=36 months
                without AD, else blank
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "labels"
RES = ROOT / "results"
CL_CUT = 20.0
DX_WIN, PET_WIN, CONV = 365, 1095, 1095


def nearest(left: pd.DataFrame, right: pd.DataFrame, on_subj: str, l_t: str, r_t: str,
            cols: list[str], window: int, suffix: str = "") -> pd.DataFrame:
    """For each left row, attach right row of the same subject with the closest time."""
    r = right[[on_subj, r_t] + cols].dropna(subset=[r_t]).rename(columns={r_t: "_rt"})
    m = left[[on_subj, l_t]].rename(columns={l_t: "_lt"}).reset_index().merge(r, on=on_subj, how="left")
    m["_gap"] = (m["_rt"] - m["_lt"]).abs()
    m = m[m["_gap"] <= window].sort_values(["index", "_gap"]).groupby("index").head(1)
    m = m.set_index("index")[cols]
    return left.join(m.add_suffix(suffix))


def cdr_to_dx(c):
    if pd.isna(c):
        return np.nan
    return "CN" if c == 0 else ("MCI" if c == 0.5 else "AD")


def amyloid_dx(row):
    dx, a = row["dx_clinical"], row["amyloid_pos"]
    if pd.isna(dx) or pd.isna(a):
        return np.nan
    if dx == "CN" and a == 0:
        return "CN"
    if dx in ("MCI", "AD") and a == 1:
        return dx
    return np.nan


# ----------------------------------------------------------------------------- OASIS-3
def oasis() -> pd.DataFrame:
    base = ROOT / "oasis3_clinical" / "OASIS3_data_files"
    f = lambda p: glob.glob(str(base / p / "**" / "*.csv"), recursive=True)[0]
    lab = pd.read_csv(ROOT / "labels.csv")
    lab = lab[lab["nifti_path"].notna()].copy()
    lab["day"] = lab["MR ID"].str.extract(r"_d(\d+)$")[0].astype(int)
    sc = pd.read_csv(ROOT / "scanner_info.csv")[["MR ID", "field_strength", "manufacturer"]]
    df = lab.merge(sc, on="MR ID", how="left")
    df = df.rename(columns={"MR ID": "key", "Subject": "subject", "CDR": "cdr", "MMSE": "mmse"})
    df["cohort"] = "oasis3"
    demo = pd.read_csv(f("demo")).rename(columns={"OASISID": "subject"})
    df = df.merge(demo[["subject", "AgeatEntry", "GENDER"]], on="subject", how="left")
    df["age"] = (df["AgeatEntry"] + df["day"] / 365.25).round(1).fillna(df["age"])   # exact age at scan
    df["sex"] = df["sex"].fillna(df["GENDER"].map({1: "M", 2: "F"}))

    d1 = pd.read_csv(f("UDSd1")).rename(columns={"OASISID": "subject"})
    mci = d1[["MCIAMEM", "MCIAPLUS", "MCINON1", "MCINON2"]].fillna(0).max(axis=1) == 1
    adet = (d1["PROBAD"] == 1) | (d1["POSSAD"] == 1)
    d1["dxc"] = np.select([d1["NORMCOG"] == 1, (d1["DEMENTED"] == 1) & adet,
                           (d1["DEMENTED"] == 1), (d1["NORMCOG"] == 0) & mci],
                          ["CN", "AD", "DEM_other", "MCI"], default="other")
    df = nearest(df, d1, "subject", "day", "days_to_visit", ["dxc"], DX_WIN)
    df["dx_clinical"] = df["dxc"].where(df["dxc"].isin(["CN", "MCI", "AD"]))

    cl = pd.read_csv(f("Centiloid")).rename(columns={"subject_id": "subject"})
    cl["pday"] = cl["oasis_session_id"].str.extract(r"_d(\d+)$")[0].astype(float)
    cl["centiloid"] = cl["Centiloid_fSUVR_TOT_CORTMEAN"]
    df = nearest(df, cl, "subject", "day", "pday", ["centiloid"], PET_WIN)
    df["amyloid_pos"] = np.where(df["centiloid"].isna(), np.nan, (df["centiloid"] >= CL_CUT).astype(float))

    fs = pd.read_csv(f("FS"))
    fs["subject"] = fs["Subject"]
    fs["fday"] = fs["MR_session"].str.extract(r"_d(\d+)$")[0].astype(float)
    fs = fs.rename(columns={"Left-Hippocampus_volume": "hv_l", "Right-Hippocampus_volume": "hv_r",
                            "IntraCranialVol": "icv"})
    df = nearest(df, fs, "subject", "day", "fday", ["hv_l", "hv_r", "icv"], 365)

    # conversion (clinical MCI at scan -> AD within 36 months)
    d1s = d1[["subject", "days_to_visit", "dxc"]]
    conv = []
    for _, r in df.iterrows():
        if r["dx_clinical"] != "MCI":
            conv.append(np.nan); continue
        fu = d1s[(d1s.subject == r.subject) & (d1s.days_to_visit > r.day)]
        if ((fu.dxc == "AD") & (fu.days_to_visit - r.day <= CONV)).any():
            conv.append("pMCI")
        elif (fu.days_to_visit - r.day >= CONV).any() and not (fu.dxc.isin(["AD", "DEM_other"]) & (fu.days_to_visit - r.day <= CONV)).any():
            conv.append("sMCI")
        else:
            conv.append(np.nan)
    df["conv36"] = conv
    return df


# ----------------------------------------------------------------------------- ADNI
def adni() -> pd.DataFrame:
    import pyreadr
    R = lambda n: list(pyreadr.read_r(str(ROOT / "adni/tables/rda/ADNIMERGE2/data" / f"{n}.rda")).values())[0]
    sel = pd.read_csv(ROOT / "adni/selection/selected_images.csv", parse_dates=["image_date"])
    df = sel.rename(columns={"subject_id": "subject", "dx": "dx_clinical"})
    df["key"] = "I" + df["image_id"].astype(str)
    df["cohort"] = "adni"
    df["t"] = df["image_date"].map(pd.Timestamp.toordinal)

    demo = R("ADSP_PHC_DEMODX").rename(columns={"PTID": "subject"})
    demo["t"] = pd.to_datetime(demo["EXAMDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    df = nearest(df, demo, "subject", "t", "t", ["PHC_Age_Diagnosis", "PHC_Sex"], DX_WIN, "_d")
    df["age"] = df["PHC_Age_Diagnosis_d"]
    df["sex"] = df["PHC_Sex_d"].map({1.0: "M", 2.0: "F"})

    fs = R("ADSP_PHC_T1_FS").rename(columns={"PTID": "subject"})
    fs = fs.rename(columns={"Left_Hippocampus": "hv_l", "Right_Hippocampus": "hv_r",
                            "EstimatedTotalIntraCranialVol": "icv", "FieldSgth": "field_strength",
                            "ScannerManufacturer": "manufacturer"})
    fs["t"] = pd.to_datetime(fs["PHC_SCANDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    df = nearest(df, fs, "subject", "t", "t", ["hv_l", "hv_r", "icv", "field_strength", "manufacturer"], 90)
    # field-strength fallback from MRI metadata at the same visit
    meta = pd.concat([R("MRIMETA"), R("MRI3META")])[["PTID", "EXAMDATE", "FIELD_STRENGTH"]].rename(columns={"PTID": "subject"})
    meta["t"] = pd.to_datetime(meta["EXAMDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    meta["fs_meta"] = pd.to_numeric(meta["FIELD_STRENGTH"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    df = nearest(df, meta, "subject", "t", "t", ["fs_meta"], 60)
    df["field_strength"] = df["field_strength"].fillna(df["fs_meta"])
    df["manufacturer"] = df["manufacturer"].str.upper().str.extract(r"(SIEMENS|GE|PHILIPS)")[0]

    cdr = R("CDR").rename(columns={"PTID": "subject"})
    cdr["t"] = pd.to_datetime(cdr["VISDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    df = nearest(df, cdr.dropna(subset=["CDGLOBAL"]), "subject", "t", "t", ["CDGLOBAL"], 180)
    df["cdr"] = df["CDGLOBAL"].where(df["CDGLOBAL"] >= 0)
    mm = R("MMSE").rename(columns={"PTID": "subject"})
    mm["t"] = pd.to_datetime(mm["VISDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    df = nearest(df, mm.dropna(subset=["MMSCORE"]), "subject", "t", "t", ["MMSCORE"], 180)
    df["mmse"] = df["MMSCORE"].where(df["MMSCORE"] >= 0)

    am = R("ADSP_PHC_PET_Amyloid_Simple").rename(columns={"PTID": "subject"})
    am["t"] = pd.to_datetime(am["PHC_SCANDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    df = nearest(df, am.dropna(subset=["PHC_CENTILOIDS"]), "subject", "t", "t", ["PHC_CENTILOIDS"], PET_WIN)
    df["centiloid"] = df["PHC_CENTILOIDS"]
    csf = R("ADSP_PHC_CSF").rename(columns={"PTID": "subject"})
    csf["t"] = pd.to_datetime(csf["DRAWDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    csf["csf_apos"] = csf["AB42pTau181_Class"].astype(str).str.startswith("A+").astype(float).where(csf["AB42pTau181_Class"].notna())
    df = nearest(df, csf, "subject", "t", "t", ["csf_apos"], PET_WIN)
    df["amyloid_pos"] = np.where(df["centiloid"].notna(), (df["centiloid"] >= CL_CUT).astype(float), df["csf_apos"])

    dx = R("DXSUM").rename(columns={"PTID": "subject"})
    dx["t"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    dx = dx[dx["DIAGNOSIS"].isin(["CN", "MCI", "Dementia"])][["subject", "t", "DIAGNOSIS"]].dropna()
    conv = []
    for _, r in df.iterrows():
        if r["dx_clinical"] != "MCI":
            conv.append(np.nan); continue
        fu = dx[(dx.subject == r.subject) & (dx.t > r.t)]
        if ((fu.DIAGNOSIS == "Dementia") & (fu.t - r.t <= CONV)).any():
            conv.append("pMCI")
        elif (fu.t - r.t >= CONV).any():
            conv.append("sMCI")
        else:
            conv.append(np.nan)
    df["conv36"] = conv
    return df


COLS = ["key", "subject", "cohort", "age", "sex", "field_strength", "manufacturer",
        "dx_clinical", "dx_cdr", "dx_amyloid", "cdr", "mmse", "centiloid", "amyloid_pos",
        "hv_l", "hv_r", "icv", "conv36"]


def finish(df: pd.DataFrame) -> pd.DataFrame:
    df["dx_cdr"] = df["cdr"].map(cdr_to_dx)
    df["dx_amyloid"] = df.apply(amyloid_dx, axis=1)
    return df[COLS]


def summarise(frames: list[pd.DataFrame]) -> None:
    RES.mkdir(exist_ok=True)
    rows, ov = [], []
    for df in frames:
        c = df["cohort"].iloc[0]
        for ld in ("dx_clinical", "dx_cdr", "dx_amyloid"):
            for g, sub in df.groupby(ld):
                rows.append([c, ld, "n", g, len(sub), np.nan, np.nan])
                rows.append([c, ld, "age", g, sub.age.notna().sum(), sub.age.mean(), sub.age.std()])
                rows.append([c, ld, "female", g, (sub.sex == "F").sum(), (sub.sex == "F").mean(), np.nan])
                rows.append([c, ld, "3T", g, (sub.field_strength >= 2.9).sum(), (sub.field_strength >= 2.9).mean(), np.nan])
        for a, b in (("dx_clinical", "dx_cdr"), ("dx_clinical", "dx_amyloid")):
            t = pd.crosstab(df[a], df[b])
            for ca in t.index:
                for cb in t.columns:
                    ov.append([c, a, b, ca, cb, int(t.loc[ca, cb])])
    pd.DataFrame(rows, columns=["cohort", "label_def", "variable", "group", "count", "mean", "sd"]).round(3).to_csv(RES / "cohort_summary.csv", index=False)
    pd.DataFrame(ov, columns=["cohort", "def_a", "def_b", "class_a", "class_b", "count"]).to_csv(RES / "label_overlap.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, fn in (("oasis3", oasis), ("adni", adni)):
        df = finish(fn())
        df.to_csv(OUT / f"{name}_labels.csv", index=False)
        frames.append(df)
        print(f"== {name}: {len(df)} scans")
        for ld in ("dx_clinical", "dx_cdr", "dx_amyloid", "conv36"):
            print(f"  {ld}: {df[ld].value_counts().to_dict()}")
        print(f"  missing: age {df.age.isna().sum()}, field {df.field_strength.isna().sum()}, hv {df.hv_l.isna().sum()}, amyloid {df.amyloid_pos.isna().sum()}")
    summarise(frames)


if __name__ == "__main__":
    main()
