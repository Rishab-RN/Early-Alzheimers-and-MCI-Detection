"""C2 (explanation drift) + C3 (ADRS failure detection) for one trained seed.

  python src/xai_reliability.py --model resnet3d --task cn_ad --seed 0 [--external oasis3] [--limit N]

Steps
 1. Integrated Gradients (32 steps, zero baseline) for every ADNI val/test scan (+ external if allowed),
    aggregated to Harvard-Oxford ROIs -> per-scan ROI profile (share of |attribution|).
 2. C2 stats: mean ROI profiles per cohort, cross-cohort Spearman / JSD, internal split-half Spearman,
    AD-prior attribution mass, ROI-deletion faithfulness (vs random order), model-randomisation sanity.
 3. C3: uncertainty scores (MSP, energy, MC-dropout, ensemble SD, feature Mahalanobis, ADRS) ->
    misclassification-detection AUROC, AURC, accuracy at 80/90% coverage, risk-coverage curves.
External cohorts require the config to be frozen (same lock as evaluate.py).
"""
from __future__ import annotations

import argparse
import copy

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from cache_tensors import crop_box, target_shape
from evaluate import DEV, frozen_ids, load_cohort
from metrics import ROOT, RES
from train import TASKS, build_model

PFX = __import__("os").environ.get("ANN_RUN_PREFIX", "")   # e.g. "proto_" for prototype runs
EPS = 1e-6


def atlas_on_grid(res: float):
    box = crop_box(); shape = target_shape(box, res)
    a = nib.load(str(ROOT / "references/atlas_ho_mni2009c_1mm.nii.gz")).get_fdata()[box]
    t = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
    a = F.interpolate(t, size=shape, mode="nearest")[0, 0].long().numpy()
    labels = pd.read_csv(ROOT / "references/atlas_ho_labels.csv")
    return a, labels


def load_x(path: str) -> torch.Tensor:
    return torch.from_numpy(np.load(path).astype(np.float32))[None, None].to(DEV)


def ig(model, x, steps: int = int(__import__("os").environ.get("ANN_IG_STEPS", 32)), sign: float = 1.0) -> np.ndarray:
    """Integrated gradients w.r.t. the logit of the predicted class (sign=+1 pos, -1 neg)."""
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (x * a).requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            z = model(xi).float().sum() * sign
        g, = torch.autograd.grad(z, xi)
        total += g.float()
    return (x * total / steps)[0, 0].detach().cpu().numpy()


def roi_profile(attr: np.ndarray, atlas: np.ndarray, n_roi: int) -> np.ndarray:
    # mean |attribution| per voxel in each ROI (density; avoids large-ROI bias), normalised to sum 1
    w = np.bincount(atlas.ravel(), weights=np.abs(attr).ravel(), minlength=n_roi + 1)[1:]
    n = np.bincount(atlas.ravel(), minlength=n_roi + 1)[1:]
    w = w / np.maximum(n, 1)
    return w / max(w.sum(), EPS)


@torch.no_grad()
def forward_feats(model, x):
    feats = {}
    last = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    h = last.register_forward_hook(lambda m, i, o: feats.__setitem__("f", i[0].detach()))
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
        z = model(x).float().item()
    h.remove()
    return z, feats["f"].float().cpu().numpy().ravel()


def mc_dropout(model, x, n=int(__import__("os").environ.get("ANN_MC_PASSES", 20))) -> float:
    m = model.eval()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
        ps = [torch.sigmoid(m(x).float()).item() for _ in range(n)]
    m.eval()
    return float(np.std(ps))


def deletion_auc(model, x, prof, atlas, order, steps=(0, 1, 2, 4, 8, 16, 32)) -> float:
    """Mean predicted-class probability while occluding ROIs in the given order (lower = more faithful)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
        p0 = torch.sigmoid(model(x).float()).item()
        sign = 1 if p0 >= 0.5 else -1
        vals = []
        at = torch.from_numpy(atlas).to(DEV)
        for k in steps:
            xm = x.clone()
            for r in order[:k]:
                xm[0, 0][at == (r + 1)] = 0.0
            p = torch.sigmoid(model(xm).float()).item()
            vals.append(p if sign == 1 else 1 - p)
    return float(np.trapezoid(vals, dx=1.0 / (len(vals) - 1)))


def gradcam(model, x, sign: float = 1.0) -> np.ndarray:
    """3D Grad-CAM on the last convolutional block (layer4), upsampled to the input grid."""
    layer = model.layer4 if hasattr(model, "layer4") else [m for m in model.modules() if hasattr(m, "layer4")][0].layer4
    store = {}
    def fwd(m, i, o):
        store["a"] = o.detach()
        o.register_hook(lambda g: store.__setitem__("g", g.detach()))   # tensor hook: no module-hook leak
    h1 = layer.register_forward_hook(fwd)
    xi = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
        z = model(xi).float().sum() * sign
    model.zero_grad(); z.backward()
    h1.remove()
    a, g = store["a"].float(), store["g"].float()
    cam = torch.relu((g.mean(dim=(2, 3, 4), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
    return cam[0, 0].detach().cpu().numpy()


def save_mean_maps(mean_maps: dict, run_id: str, res: float, suffix: str = "ig") -> None:
    """Group-average |IG| maps (aggregate, no subject data) on the MNI template grid."""
    box = crop_box(); shape = target_shape(box, res)
    tpl = nib.load(str(ROOT / "references/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"))
    ext = np.array([s.stop - s.start for s in box], float)
    vs = ext / np.array(shape)
    start = np.array([s.start for s in box], float) + vs / 2 - 0.5
    aff = tpl.affine.copy(); aff[:3, :3] = tpl.affine[:3, :3] @ np.diag(vs); aff[:3, 3] = tpl.affine[:3, :3] @ start + tpl.affine[:3, 3]
    out = RES / "maps"; out.mkdir(parents=True, exist_ok=True)
    for name, (sumarr, n) in mean_maps.items():
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(sumarr / max(n, 1), sigma=1.0).astype(np.float32)
        m = m / (np.percentile(m[m > 0], 99.5) + 1e-8)
        img = nib.Nifti1Image(np.clip(m, 0, 1), aff)
        if suffix == "gradcam":
            from map_display import display_map
            img = display_map(img)
        nib.save(img, str(out / f"{run_id}_{name}_mean_{suffix}.nii.gz"))


def maha_fit(X):
    lw = LedoitWolf().fit(X)
    return lw.location_, lw.precision_


def maha(X, mu, prec):
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))


def aurc(conf, err):
    o = np.argsort(-conf)
    e = err[o]
    risks = np.cumsum(e) / np.arange(1, len(e) + 1)
    cov = np.arange(1, len(e) + 1) / len(e)
    return float(np.trapezoid(risks, cov)), cov, risks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet3d")
    ap.add_argument("--task", default="cn_ad")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=float, default=1.5)
    ap.add_argument("--external", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    neg, pos = TASKS[args.task]
    cfg = f"{args.model}_{args.task}"
    run_id = f"{PFX}{cfg}_s{args.seed}"
    if args.external and cfg not in frozen_ids():
        raise SystemExit(f"{cfg} not frozen; external XAI/reliability refused")
    torch.manual_seed(0); np.random.seed(0)

    model = build_model(args.model).to(DEV)
    model.load_state_dict(torch.load(ROOT / f"models/runs/{run_id}.pt", map_location=DEV)["model"]); model.eval()
    atlas, labels = atlas_on_grid(args.res)
    n_roi = len(labels)
    ad_prior = labels.ad_prior.values.astype(bool)

    adni = load_cohort("adni", args.task, args.res)
    sets = {"adni_val": adni[adni.split == "val"], "adni": adni[adni.split == "test"]}
    for c in filter(None, args.external.split(",")):
        sets[c] = load_cohort(c, args.task, args.res)
    ens_dir = RES / "predictions"

    recs = []
    mean_maps, cam_maps = {}, {}
    for name, df in sets.items():
        df = df[df.dx_clinical.isin([neg, pos]) | (name not in ("adni_val",))]
        if args.limit:
            df = df.head(args.limit)
        ens = ens_dir / f"{PFX}ensemble_{cfg}_{'adni' if name.startswith('adni') else name}.csv"
        ens_sd = pd.read_csv(ens).set_index("key")["ens_sd"] if ens.exists() else pd.Series(dtype=float)
        for i, r in enumerate(df.itertuples()):
            x = load_x(r.path)
            z, f = forward_feats(model, x)
            attr = ig(model, x, sign=1.0 if z >= 0 else -1.0)
            prof = roi_profile(attr, atlas, n_roi)
            acc = mean_maps.setdefault(name, [np.zeros(attr.shape, np.float64), 0])
            acc[0] += np.abs(attr); acc[1] += 1
            cam = gradcam(model, x, sign=1.0 if z >= 0 else -1.0)
            camp = roi_profile(cam, atlas, n_roi)
            accg = cam_maps.setdefault(name, [np.zeros(cam.shape, np.float64), 0])
            accg[0] += cam; accg[1] += 1
            recs.append({"set": name, "key": r.key, "dx_clinical": r.dx_clinical, "dx_cdr": r.dx_cdr,
                         "dx_amyloid": r.dx_amyloid, "z": z, "mc_sd": mc_dropout(model, x),
                         "ens_sd": ens_sd.get(r.key, np.nan), "feat": f, "prof": prof, "camp": camp})
            model.zero_grad(set_to_none=True)
            if DEV == "cuda":
                torch.cuda.empty_cache()                 # avoid creeping into shared memory on 4 GB GPUs
            if i % 50 == 0:
                print(f"{name}: {i}/{len(df)}", flush=True)
    R = pd.DataFrame(recs)
    save_mean_maps(mean_maps, run_id, args.res)
    save_mean_maps(cam_maps, run_id, args.res, suffix="gradcam")
    P = np.stack(R.prof.values)
    Fm = np.stack(R.feat.values)

    # ---------------- C2: explanation drift ----------------
    out_roi = RES / "roi_attribution"; out_roi.mkdir(parents=True, exist_ok=True)
    mean_prof = {}
    for s in R.set.unique():
        Ps = P[R.set.values == s]
        mean_prof[s] = Ps.mean(0)
        pd.DataFrame({"roi": labels.roi, "mean": Ps.mean(0), "sd": Ps.std(0), "n": len(Ps), "ad_prior": ad_prior}) \
            .sort_values("mean", ascending=False).to_csv(out_roi / f"{run_id}_{s}.csv", index=False)
        Cs = np.stack(R.camp.values)[R.set.values == s]
        pd.DataFrame({"roi": labels.roi, "mean": Cs.mean(0), "sd": Cs.std(0), "n": len(Cs), "ad_prior": ad_prior})             .sort_values("mean", ascending=False).to_csv(out_roi / f"{run_id}_{s}_gradcam.csv", index=False)
    stats = []
    Pt = P[R.set.values == "adni"]
    rng = np.random.default_rng(0)
    halves = [spearmanr(Pt[i[: len(i) // 2]].mean(0), Pt[i[len(i) // 2:]].mean(0))[0]
              for i in (rng.permutation(len(Pt)) for _ in range(200))]
    stats.append([run_id, "adni_split_half", "spearman", np.mean(halves), np.percentile(halves, 2.5), np.percentile(halves, 97.5)])
    CP = np.stack(R.camp.values)
    for s in mean_prof:
        stats.append([run_id, s, "ig_vs_gradcam_spearman", spearmanr(mean_prof[s], CP[R.set.values == s].mean(0))[0], np.nan, np.nan])
        mass = P[R.set.values == s][:, ad_prior].sum(1)
        stats.append([run_id, s, "ad_prior_mass", mass.mean(), np.percentile(mass, 2.5), np.percentile(mass, 97.5)])
        if s not in ("adni", "adni_val"):
            Pe = P[R.set.values == s]
            boots = []
            for _ in range(500):
                a = Pt[rng.integers(0, len(Pt), len(Pt))].mean(0); b = Pe[rng.integers(0, len(Pe), len(Pe))].mean(0)
                boots.append((spearmanr(a, b)[0], jensenshannon(a, b) ** 2))
            boots = np.array(boots)
            stats.append([run_id, f"adni_vs_{s}", "spearman", spearmanr(mean_prof["adni"], mean_prof[s])[0], *np.percentile(boots[:, 0], [2.5, 97.5])])
            stats.append([run_id, f"adni_vs_{s}", "jsd", jensenshannon(mean_prof["adni"], mean_prof[s]) ** 2, *np.percentile(boots[:, 1], [2.5, 97.5])])
    # faithfulness (ROI deletion) and randomisation sanity on a subsample of ADNI test
    sub = R[R.set == "adni"].head(40)
    dl_attr, dl_rand = [], []
    for r in sub.itertuples():
        x = load_x(sets["adni"].set_index("key").loc[r.key, "path"])
        order = np.argsort(-r.prof)
        dl_attr.append(deletion_auc(model, x, r.prof, atlas, order))
        dl_rand.append(deletion_auc(model, x, r.prof, atlas, rng.permutation(n_roi)))
    stats.append([run_id, "adni", "deletion_auc_attr", np.mean(dl_attr), *np.percentile(dl_attr, [2.5, 97.5])])
    stats.append([run_id, "adni", "deletion_auc_random", np.mean(dl_rand), *np.percentile(dl_rand, [2.5, 97.5])])
    rand_model = copy.deepcopy(model)
    for mod in list(rand_model.modules())[-12:]:
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    sims = []
    for r in sub.head(15).itertuples():
        x = load_x(sets["adni"].set_index("key").loc[r.key, "path"])
        sims.append(spearmanr(r.prof, roi_profile(ig(rand_model, x), atlas, n_roi))[0])
    stats.append([run_id, "adni", "randomization_sim", np.mean(sims), *np.percentile(sims, [2.5, 97.5])])
    xs = RES / "xai_stats.csv"
    new = pd.DataFrame(stats, columns=["run_id", "comparison", "metric", "value", "ci_low", "ci_high"])
    if xs.exists():
        old = pd.read_csv(xs); new = pd.concat([old[old.run_id != run_id], new])
    new.round(5).to_csv(xs, index=False)

    # ---------------- C3: reliability / failure detection ----------------
    val = R.set.values == "adni_val"
    yval = (R.dx_clinical[val] == pos).astype(int).values
    p = 1 / (1 + np.exp(-R.z.values))
    pv = p[val]
    thr = max(np.unique(np.quantile(pv, np.linspace(.01, .99, 99))),
              key=lambda t: ((pv >= t) == yval).mean() if False else
              (np.mean((pv >= t)[yval == 1]) + np.mean((pv < t)[yval == 0])) / 2)
    pred = (p >= thr).astype(int)
    # class-conditional feature Mahalanobis
    fm = {c: maha_fit(Fm[val][pred[val] == c]) for c in (0, 1)}
    feat_score = np.where(pred == 1, maha(Fm, *fm[1]), maha(Fm, *fm[0]))
    # ADRS: CLR-transformed ROI profile -> PCA (fit on ADNI val) -> class-conditional Mahalanobis
    clr = np.log(P + EPS); clr -= clr.mean(1, keepdims=True)
    pca = PCA(n_components=min(10, val.sum() - 1)).fit(clr[val])
    Z = pca.transform(clr)
    am = {c: maha_fit(Z[val][pred[val] == c]) for c in (0, 1)}
    adrs = np.where(pred == 1, maha(Z, *am[1]), maha(Z, *am[0]))
    scores = {"msp": -np.maximum(p, 1 - p), "energy": -np.logaddexp(0, np.abs(R.z.values)),
              "mc_dropout": R.mc_sd.values, "ensemble": R.ens_sd.values, "feature_maha": feat_score, "adrs": adrs}
    rel_rows = []
    rc_dir = RES / "risk_coverage"; rc_dir.mkdir(parents=True, exist_ok=True)
    for s in [x for x in R.set.unique() if x != "adni_val"]:
        m = (R.set.values == s) & R.dx_clinical.isin([neg, pos]).values
        y = (R.dx_clinical.values[m] == pos).astype(int)
        err = (pred[m] != y).astype(int)
        if err.sum() < 3 or err.sum() == len(err):
            continue
        for meth, sc in scores.items():
            u = sc[m]
            if np.isnan(u).all():
                continue
            u = np.nan_to_num(u, nan=np.nanmedian(u))
            au, cov, risk = aurc(-u, err)
            o = np.argsort(u)
            acc = lambda c: 1 - err[o][: int(len(o) * c)].mean()
            for met, v in [("err_auroc", roc_auc_score(err, u)), ("aurc", au), ("acc@80", acc(.8)), ("acc@90", acc(.9))]:
                rel_rows.append([run_id, meth, s, met, v, np.nan, np.nan])
            pd.DataFrame({"coverage": cov, "risk": risk}).iloc[:: max(1, len(cov) // 100)] \
                .to_csv(rc_dir / f"{run_id}_{s}_{meth}.csv", index=False)
    rp = RES / "reliability.csv"
    new = pd.DataFrame(rel_rows, columns=["run_id", "method", "cohort", "metric", "value", "ci_low", "ci_high"])
    if rp.exists():
        old = pd.read_csv(rp); new = pd.concat([old[old.run_id != run_id], new])
    new.round(5).to_csv(rp, index=False)
    R.drop(columns=["feat", "prof", "camp"]).assign(adrs=adrs, feat_maha=feat_score) \
        .to_csv(RES / "predictions" / f"{run_id}_xai_scores.csv", index=False)
    np.save(RES / "predictions" / f"{run_id}_roi_profiles.npy", P)
    print(pd.DataFrame(stats, columns=["run", "cmp", "metric", "v", "lo", "hi"]).round(3).to_string())
    print(new[new.metric == "err_auroc"].pivot_table(index="method", columns="cohort", values="value").round(3))


if __name__ == "__main__":
    main()
