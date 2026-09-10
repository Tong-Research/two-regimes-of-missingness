"""Figures for paper-two-regimes, generated from the committed pre-registered results.
Run: ../paper-plco-hypergraph/.venv/bin/python figures/make_figures.py"""
import glob, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SRC = pathlib.Path(__file__).resolve().parents[1]; OUT = pathlib.Path(__file__).resolve().parent
NAME = {"cand_eicu_regime": "eICU", "cand_mimic4_regime": "MIMIC-IV", "cand_nhanes": "NHANES", "sweep_42739": "road safety",
        "cand_acs_income": "ACS income", "cand_airbnb": "Airbnb", "cand_higgs": "Higgs", "cand_porto": "Porto"}
REG = ["cand_eicu_regime", "cand_mimic4_regime", "cand_nhanes", "sweep_42739"]
STR = ["cand_acs_income", "cand_airbnb", "cand_higgs", "sweep_42737", "sweep_46654", "sweep_46703"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
def nm(d): return NAME.get(d, d.replace("sweep_", ""))
def panels(sub):
    x = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(SRC / f"results/{sub}/*.csv")))], ignore_index=True)
    return x[x.kind == "auroc"].groupby(["dataset", "model"]).value.mean().unstack("model")

def fig_dissociation():
    dv, mi = panels("rrstruct"), panels("rrmirror")
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2))
    for a, m, ttl in ((ax[0], dv, "fit on development half"), (ax[1], mi, "fit on holdout half (mirror)")):
        a.plot([0.45, 1.02], [0.45, 1.02], color="0.75", lw=0.8, zorder=0)
        seen = set()
        for ds in m.index:
            g = "protocol-driven" if ds in REG else "structural" if ds in STR else "unclassified"
            c = {"protocol-driven": "#c0392b", "structural": "#2c6fbb", "unclassified": "0.6"}[g]
            mk = {"protocol-driven": "o", "structural": "s", "unclassified": "^"}[g]
            a.scatter(m.loc[ds, "values"], m.loc[ds, "panels_tree"], s=26, c=c, marker=mk, zorder=3,
                      label=None if g in seen else g); seen.add(g)
            xy = (m.loc[ds, "values"], m.loc[ds, "panels_tree"])
            near = sum(1 for e in m.index if e != ds and abs(m.loc[e, "values"] - xy[0]) < 0.03 and abs(m.loc[e, "panels_tree"] - xy[1]) < 0.03)
            off = (4, 3) if not near else ((4, 3) if hash(ds) % 2 else (4, -9))
            a.annotate(nm(ds), xy, fontsize=6, xytext=off, textcoords="offset points", color=c)
        a.set_xlim(0.45, 1.03); a.set_ylim(0.45, 1.03); a.set_xlabel("predicted from the observed values")
        a.set_title(ttl, fontsize=8)
    ax[0].set_ylabel("predicted from the other panels' presences")
    h, l = ax[0].get_legend_handles_labels(); d = dict(zip(l, h)); ax[0].legend(d.values(), d.keys(), frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "dissociation.pdf"); plt.close(fig)

def fig_maskinc():
    x = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(SRC / "results/maskinc/*.csv")))], ignore_index=True)
    w = x[x.arm != "mask_full"].groupby(["dataset", "q", "arm"]).auprc.median().unstack("arm")
    gf = (w["values+mask_full"] - w["values"]).unstack("q"); gk = (w["values+mask_kept"] - w["values"]).unstack("q")
    qs = [0.1, 0.25, 0.5, 0.75, 1.0]
    fig, a = plt.subplots(figsize=(3.9, 2.9))
    for ds in gf.index:
        c = "#c0392b" if ds in REG else "#2c6fbb" if ds in STR else "0.75"
        a.plot(qs, [gf.loc[ds, q] for q in qs], color=c, lw=0.9, marker="o", ms=2.5, alpha=0.85, zorder=3 if ds in REG else 2)
    a.plot(qs, [gk[q].median() for q in qs], color="k", lw=1.4, ls="--", marker="s", ms=3, label="retained columns' indicators (median)")
    a.plot(qs, [gf[q].median() for q in qs], color="k", lw=1.4, marker="o", ms=3, label="all indicators (median)")
    a.axhline(0, color="0.8", lw=0.7, zorder=0); a.set_xlabel("fraction of value columns retained")
    a.set_ylabel("gain in average precision"); a.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.6)
    a.set_xticks(qs); a.set_xticklabels(["0.1", "0.25", "0.5", "0.75", "1"])
    fig.tight_layout(); fig.savefig(OUT / "maskinc.pdf"); plt.close(fig)

if __name__ == "__main__":
    fig_dissociation(); fig_maskinc(); print("wrote figures/dissociation.pdf and figures/maskinc.pdf")
