#!/usr/bin/env python3
"""Build index.html from the repository's result files.

Every number shown on the page is read from runs.csv, summary.csv, correlations.csv,
ablation*.csv, theorem_validation_*.csv/json, wedge_failure_frequency.csv and the
README-quoted counts are asserted against the data before the page is written.

Usage (from the repository root):
    KATEX=/path/to/node_modules/katex python3 tools/build_site.py

KaTeX is only needed at build time (equations are pre-rendered); the built page
loads no external scripts. Requires: pandas, numpy, pillow, node.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

MODELS = {"rf": "RF", "mlp_data": "MLP (data)", "mlp_physics": "MLP (physics-aware)"}
SAMPLING = {"uniform": "Uniform", "boundary": "Boundary-aware"}
BUDGETS = [50, 100, 250, 500, 1000, 2500]

# ----------------------------------------------------------------- data
runs = pd.read_csv("runs.csv")
abl_raw = pd.read_csv("ablation_raw.csv")
summ = pd.read_csv("summary.csv")
corr = pd.read_csv("correlations.csv")
abl = pd.read_csv("ablation.csv").set_index("variant")
bh = pd.read_csv("ablation_bh_correction.csv").set_index("test")
th = pd.read_csv("theorem_validation_multiseed.csv")
th_cfg = pd.read_csv("theorem_validation_by_config.csv")
th_sum = json.load(open("theorem_validation_summary.json"))
wf = pd.read_csv("wedge_failure_frequency.csv")
rfw = pd.read_csv("rf_wedge_seedbyseed.csv")
main2500 = pd.read_csv("main_table_N2500.csv").set_index(["model", "sampling_strategy"])
fail = pd.read_csv("failures.csv")
npz = np.load("figure1_data.npz")

runs["wedge_detected"] = runs["wedge_detected"].astype(bool)

# ------------------------------------------------- consistency assertions
assert len(runs) == 360 and (runs.status == "COMPLETED").all() and runs.seed.nunique() == 10
assert len(abl_raw) == 80 and (abl_raw.status == "COMPLETED").all() and abl_raw.seed.nunique() == 20
assert len(fail) == 0
assert int(th["n_testable"].sum()) == th_sum["total_testable_checks"] == 7950
assert int(th["n_satisfying_bound"].sum()) == th_sum["total_satisfying"] == 7934
assert len(th) == th_sum["n_config_seed_evaluations"] == 50
assert th_sum["n_boundary_points"] == 210 and th_sum["n_testable"] == 159
pooled = corr.iloc[0]
assert pooled["n"] == 360
from scipy.stats import spearmanr  # recompute pooled correlations independently
r1 = spearmanr(runs.rmse, runs.hausdorff)
r2 = spearmanr(runs.rmse, runs.region_disagreement)
assert abs(r1[0] - pooled.rho_hausdorff) < 1e-9 and abs(r2[0] - pooled.rho_region) < 1e-9
assert abs(r1[1] - pooled.p_hausdorff) < 1e-9

wedge_missed = (~runs.wedge_detected).groupby([runs.model, runs.budget]).sum().unstack()  # model x budget (of 20)
for _, row in wf[wf.sampling == "both"].iterrows():
    assert int(wedge_missed.loc[row.model, row.budget]) == int(row.n_wedge_missed)
assert len(rfw) == 20 and (~rfw.wedge_detected.astype(bool)).all()
assert wedge_missed.loc["rf", 1000] == 20 and wedge_missed.loc["rf", 2500] == 20
assert wedge_missed.loc[["mlp_data", "mlp_physics"], [1000, 2500]].values.sum() == 0

m_rb = runs.groupby(["budget", "model"])[["rmse", "hausdorff"]].mean().unstack()
assert (m_rb["hausdorff"].idxmax(axis=1) == "rf").all()                 # RF worst d_H at every budget
assert (m_rb["rmse"].loc[[250, 500, 1000, 2500]].idxmin(axis=1) == "rf").all()  # RF best RMSE from 250
assert m_rb["rmse"].loc[[50, 100]].idxmin(axis=1).ne("rf").all()          # ...but not at 50/100
assert m_rb["rmse"]["rf"].is_monotonic_decreasing
assert wedge_missed.loc["mlp_data", [50, 100, 250]].tolist() == [1, 1, 1]

# ------------------------------------------------------------- formatters
def fp(p):
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.2e}".replace("e-0", "e-").replace("e-", "×10<sup>−") + "</sup>"

def fp_plain(p):
    return f"{p:.2f}" if p >= 0.01 else f"{p:.1e}".replace("e-0", "e-")

def fp_html(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    m, e = f"{p:.2e}".split("e")
    return f"{float(m):.2f}×10<sup>{int(e)}</sup>".replace("-", "−")

def pm(m, s, d=3):
    return f'{m:.{d}f} <span class="pm">± {s:.{d}f}</span>'

def sgn(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")

# ------------------------------------------------------------------ tables
def table_rows():
    out = []
    s = summ.copy()
    s["m_order"] = s.model.map({k: i for i, k in enumerate(MODELS)})
    s["s_order"] = s.sampling_strategy.map({"uniform": 0, "boundary": 1})
    s = s.sort_values(["m_order", "budget", "s_order"])
    wm = (~runs.wedge_detected).groupby([runs.model, runs.budget, runs.sampling_strategy]).sum()
    prev = None
    for _, r in s.iterrows():
        assert r.n_seeds == 10
        w = int(wm.loc[(r.model, r.budget, r.sampling_strategy)])
        cls = ' class="grp"' if prev is not None and prev != r.model else ""
        prev = r.model
        wcls = "bad" if w == 10 else ("good" if w == 0 else "")
        out.append(
            f'          <tr{cls} data-model="{r.model}" data-budget="{int(r.budget)}" data-sampling="{r.sampling_strategy}" '
            f'data-rmse="{r.rmse_mean:.6f}" data-dh="{r.hausdorff_mean:.6f}" data-reg="{r.region_disagreement_mean:.6f}" '
            f'data-miss="{r.r_miss_mean:.6f}" data-wedge="{w}">'
            f'<td class="l">{MODELS[r.model]}</td><td>{int(r.budget)}</td><td class="l">{SAMPLING[r.sampling_strategy]}</td>'
            f'<td>{pm(r.rmse_mean, r.rmse_std)}</td><td>{pm(r.hausdorff_mean, r.hausdorff_std)}</td>'
            f'<td>{pm(r.region_disagreement_mean, r.region_disagreement_std, 4)}</td>'
            f'<td>{pm(r.r_miss_mean, r.r_miss_std, 4)}</td><td class="{wcls}">{w}/10</td></tr>'
        )
    return "\n".join(out)

def sweep_rows():
    out = []
    for b in BUDGETS:
        rm = [m_rb["rmse"].loc[b, k] for k in MODELS]
        dh = [m_rb["hausdorff"].loc[b, k] for k in MODELS]
        cells = []
        for vals in (rm, dh):
            lo = min(vals)
            cells += [f'<td class="{"good" if v == lo else ""}">{v:.3f}</td>' for v in vals]
        out.append(f"          <tr><td>{b}</td>{''.join(cells)}</tr>")
    return "\n".join(out)

def wedge_rows():
    out = []
    for k, name in MODELS.items():
        cells = []
        for b in BUDGETS:
            n = int(wedge_missed.loc[k, b])
            cls = "bad" if n == 20 else ("good" if n == 0 else "")
            cells.append(f'<td class="{cls}">{n}/20</td>')
        out.append(f'          <tr><td class="l">{name}</td>{"".join(cells)}</tr>')
    return "\n".join(out)

def corr_rows():
    def lab(s):
        if s.startswith("POOLED"):
            return "Pooled (all 360 runs)"
        if s.startswith("SEED-AGG"):
            return "Seed-aggregated (36 configs, 1 point each)"
        k, v = s.split("=")
        if k == "model":
            return f"Model: {MODELS[v]}"
        if k == "sampling":
            return f"Sampling: {SAMPLING[v]}"
        return f"Budget: N = {v[1:]}"
    out = []
    for _, r in corr.iterrows():
        out.append(
            f'          <tr><td class="l">{lab(r.label)}</td><td>{int(r.n)}</td><td>{sgn(r.rho_hausdorff)}</td>'
            f'<td>{fp_html(r.p_hausdorff)}</td><td>{sgn(r.rho_region)}</td><td>{fp_html(r.p_region)}</td></tr>'
        )
    return "\n".join(out)

def th_rows():
    out = []
    for _, r in th_cfg.iterrows():
        out.append(
            f'          <tr><td class="l">{MODELS[r.model]}</td><td>{int(r.N)}</td><td>{int(r.n_seed_evals)}</td>'
            f'<td>{100 * r.mean_fraction_satisfying:.2f}%</td><td>{100 * r.min_fraction_satisfying:.2f}%</td>'
            f'<td>{sgn(r.mean_spearman_m_vs_disp, 2)}</td></tr>'
        )
    return "\n".join(out)

def abl_rows():
    names = {"data": "Data loss only (baseline)", "tsirelson": "+ Tsirelson penalty",
             "boundary": "+ boundary weighting", "both": "+ both"}
    out = []
    for v in ["data", "tsirelson", "boundary", "both"]:
        r = abl.loc[v]
        base = v == "data"
        rm = f"{r.rmse_mean:.3f} ({r.rmse_ci95_lo:.3f}–{r.rmse_ci95_hi:.3f})"
        rm += "" if base else f' <span class="pm">p = {fp_html(r.rmse_mannwhitney_p)}</span>'
        dh = f"{r.hausdorff_mean:.3f} ({r.hausdorff_ci95_lo:.3f}–{r.hausdorff_ci95_hi:.3f})"
        ts = f"{r.tsirelson_violation_rate_mean:.3f} ({r.tsirelson_violation_rate_ci95_lo:.3f}–{r.tsirelson_violation_rate_ci95_hi:.3f})"
        if base:
            p1 = p2 = "—"
            c1 = c2 = ""
        else:
            a, b = bh.loc[f"{v}_hausdorff", "p_BH_adj"], bh.loc[f"{v}_tsirelson_rate", "p_BH_adj"]
            p1, p2 = fp_html(a), fp_html(b)
            c1, c2 = ("good" if a < 0.05 else ""), ("good" if b < 0.05 else "")
        out.append(f'          <tr><td class="l">{names[v]}</td><td>{rm}</td><td>{dh}</td><td class="{c1}">{p1}</td>'
                   f'<td>{ts}</td><td class="{c2}">{p2}</td></tr>')
    return "\n".join(out)

def seed_matrix():
    seeds = sorted(runs.seed.unique())
    pitch, r = 9, 3.3
    cw, ch, gap = 106, 46, 6
    x0, y0 = 92, 30
    W = x0 + len(BUDGETS) * (cw + gap)
    H = y0 + len(MODELS) * (ch + 10) + 4
    parts = [f'<svg class="seedmatrix" style="min-width:640px" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Wedge outcome for every run: one mark per seed, per sampling strategy, model and budget. '
             f'Random Forest misses the wedge in 20 of 20 evaluations at N = 1000 and 2500; both MLPs recover it in all of them.">'
             f'<title>Wedge outcome of every main run</title>']
    for j, b in enumerate(BUDGETS):
        parts.append(f'<text x="{x0 + j * (cw + gap) + cw / 2}" y="18" text-anchor="middle" class="lab">N = {b}</text>')
    for i, k in enumerate(MODELS):
        y = y0 + i * (ch + 10)
        parts.append(f'<text x="0" y="{y + 22}" class="lab">{MODELS[k]}</text>')
        for j, b in enumerate(BUDGETS):
            x = x0 + j * (cw + gap)
            parts.append(f'<rect class="cell" x="{x}" y="{y}" width="{cw}" height="{ch}" rx="6"/>')
            for sr, samp in enumerate(["uniform", "boundary"]):
                sub = runs[(runs.model == k) & (runs.budget == b) & (runs.sampling_strategy == samp)].set_index("seed")
                for si, sd in enumerate(seeds):
                    det = bool(sub.loc[sd, "wedge_detected"])
                    cx, cy = x + 10 + pitch * si + 4.5, y + 10 + pitch * sr + 2
                    tip = f"{MODELS[k]}, N={b}, {samp}, seed {sd}: wedge {'recovered' if det else 'missed'}"
                    parts.append(f'<circle class="{"found" if det else "missed"}" cx="{cx}" cy="{cy}" r="{r}"><title>{tip}</title></circle>')
            parts.append(f'<text x="{x + cw / 2}" y="{y + ch - 8}" text-anchor="middle">{int(wedge_missed.loc[k, b])}/20 missed</text>')
    parts.append("</svg>")
    return "".join(parts)

# -------------------------------------------------------------- figures
FIGS = [
    ("figW_ground_truth_landscape", "W", "Ground-truth landscape and boundary",
     "Exact S_max(T, h) with the extracted S_max = 2 boundary; red points are the high-field wedge (h/J ≥ 2.5)."),
    ("figA_rmse_vs_budget", "A", "RMSE versus training budget", "Mean RMSE by model with ±1 s.d. bands (10 seeds × 2 sampling strategies)."),
    ("figB_hausdorff_vs_budget", "B", "Hausdorff error versus training budget", "Mean Hausdorff boundary error by model."),
    ("figC_rmse_vs_hausdorff", "C", "RMSE versus Hausdorff error", "All 360 runs; color = model, marker = sampling."),
    ("figD_rmse_vs_region", "D", "RMSE versus region disagreement", "All 360 runs; color = model, marker = sampling."),
    ("figE_dH_distribution_N2500", "E", "Hausdorff error distribution at N = 2500", "Box plots across 10 seeds."),
    ("figF_wedge_recovery", "F", "Wedge recovery by seed", "Maximum h/J reached by each run's predicted boundary at N = 1000 and 2500."),
    ("figH_ablation_20seed", "H", "Tsirelson-aware loss ablation", "Boundary error and Tsirelson violation rate, 20 seeds, 95% CI."),
]
FIGMAP = {f[0]: f for f in FIGS}

def size(stem):
    im = Image.open(f"assets/figures/{stem}.png")
    return im.size

def has_pdf(stem):
    return Path(f"assets/figures/{stem}.pdf").exists()

def fig_html(stem, title, cap, lazy=True, links=True):
    w, h = size(stem)
    png = f"assets/figures/{stem}.png"
    alt = f"{title}. {cap}"
    l = ""
    if links:
        l = f'<div class="fig-links"><a href="{png}">Open full resolution</a>'
        if has_pdf(stem):
            l += f'<a href="assets/figures/{stem}.pdf">PDF</a>'
        l += "</div>"
    return (f'<figure class="fig"><a class="zoom" href="{png}" data-title="{title}"><div class="imgwrap">'
            f'<img {"loading=\"lazy\" " if lazy else ""}src="{png}" width="{w}" height="{h}" alt="{alt}"></div></a>'
            f'<figcaption><b>{title}.</b> {cap}{l}</figcaption></figure>')

def gallery():
    return "\n".join("      " + fig_html(s, f"Figure {k}: {t}", c) for s, k, t, c in FIGS)

# ---------------------------------------------------------------- numbers
bd = []
for j, hh in enumerate(npz["h_vals"]):
    sign = npz["S_max"][:, j] - 2.0
    for i0 in np.where(np.diff(np.sign(sign)) != 0)[0]:
        y0, y1 = sign[i0], sign[i0 + 1]
        if y1 == y0:
            continue
        t0, t1 = np.log(npz["T_vals"][i0]), np.log(npz["T_vals"][i0 + 1])
        bd.append((np.exp(t0 + (0 - y0) * (t1 - t0) / (y1 - y0)), hh))
bd = np.array(bd)
assert len(bd) == th_sum["n_boundary_points"]
wedge_pts = bd[bd[:, 1] >= 2.5]

w_png = size("figW_ground_truth_landscape")[0]
def tok_frac(k, b):
    return f"{int(wedge_missed.loc[k, b])}/20"

rf_dh_sd = main2500.loc[("rf", "boundary"), "hausdorff_std"]
gap = main2500.loc[("rf", "boundary"), "hausdorff_mean"] - main2500.loc[("mlp_data", "boundary"), "hausdorff_mean"]
small = wedge_missed.loc["rf", [50, 100, 250, 500]]

VALUES = {
    "W_SIZE": str(w_png),
    "N_BD": str(len(bd)), "N_WEDGE": str(len(wedge_pts)), "WEDGE_TMAX": f"{wedge_pts[:, 0].max():.3f}",
    "RF_RMSE": f"{main2500.loc[('rf','boundary'),'rmse_mean']:.3f}", "MD_RMSE": f"{main2500.loc[('mlp_data','boundary'),'rmse_mean']:.3f}",
    "RF_DH": f"{main2500.loc[('rf','boundary'),'hausdorff_mean']:.3f}", "MD_DH": f"{main2500.loc[('mlp_data','boundary'),'hausdorff_mean']:.3f}",
    "RF_MISS_1000": tok_frac("rf", 1000), "RF_MISS_2500": tok_frac("rf", 2500),
    "RF_MISS_SMALL": f"{int(small.min())} to {int(small.max())}",
    "RHO_H": sgn(pooled.rho_hausdorff), "P_H": f"{pooled.p_hausdorff:.2f}", "P_H_BH": f"{pooled.p_hausdorff_BH_adj_primary:.2f}",
    "RHO_R": f"{pooled.rho_region:.3f}", "P_R": fp_html(pooled.p_region),
    "TH_SAT": f"{th_sum['total_satisfying']:,}", "TH_TOT": f"{th_sum['total_testable_checks']:,}",
    "TH_PCT": f"{100 * th_sum['overall_fraction_satisfying']:.2f}%",
    "TH_VIOL": str(th_sum["total_testable_checks"] - th_sum["total_satisfying"]),
    "TH_EVALS": str(th_sum["n_config_seed_evaluations"]), "TH_TESTABLE": str(th_sum["n_testable"]), "TH_BD": str(th_sum["n_boundary_points"]),
    "P_TS_BH": fp_html(bh.loc["tsirelson_tsirelson_rate", "p_BH_adj"]),
    "P_DH_BH_MIN": f"{bh.loc[[i for i in bh.index if i.endswith('hausdorff')], 'p_BH_adj'].min():.2f}",
    "N_MAIN": str(len(runs)), "N_SEEDS": str(runs.seed.nunique()), "N_ABL": str(len(abl_raw)),
    "N_TOTAL": str(len(runs) + len(abl_raw)), "N_FAIL": str(len(fail) + int((runs.status != "COMPLETED").sum()) + int((abl_raw.status != "COMPLETED").sum())),
    "RF_DH_50": f"{m_rb['hausdorff'].loc[50, 'rf']:.3f}", "RF_DH_100": f"{m_rb['hausdorff'].loc[100, 'rf']:.3f}",
    "RF_DH_2500": f"{m_rb['hausdorff'].loc[2500, 'rf']:.3f}",
    "RF_DH_SD": f"{rf_dh_sd:.3f}", "GAP_DH": f"{gap:.1f}",
    "ABL_DH_DATA": f"{abl.loc['data', 'hausdorff_mean']:.3f}", "ABL_DH_TS": f"{abl.loc['tsirelson', 'hausdorff_mean']:.3f}",
    "ABL_DH_BOTH": f"{abl.loc['both', 'hausdorff_mean']:.3f}",
    "BUDGET_OPTIONS": "".join(f'<option value="{b}">{b}</option>' for b in BUDGETS),
    "WEDGE_HEAD": "".join(f"<th>N = {b}</th>" for b in BUDGETS),
    "TABLE_ROWS": table_rows(), "SWEEP_ROWS": sweep_rows(), "WEDGE_ROWS": wedge_rows(), "CORR_ROWS": corr_rows(),
    "TH_ROWS": th_rows(), "ABL_ROWS": abl_rows(), "SEED_MATRIX": seed_matrix(), "GALLERY": gallery(),
}
# sanity: values quoted in the repository README
assert VALUES["TH_PCT"] == "99.80%" and VALUES["RHO_H"] == "−0.075" and VALUES["RHO_R"] == "0.559"
assert VALUES["N_TOTAL"] == "440" and VALUES["N_FAIL"] == "0" and len(wedge_pts) == 40

# ------------------------------------------------------------------- math
MATH = {
    "M:Smax": r"S_{\max}", "M:Sgt2": r"S_{\max}>2", "M:tsir": r"2\sqrt{2}", "M:TJ": r"T/J",
    "M:TT": r"\mathcal{T}^{\mathsf{T}}\mathcal{T}", "M:u": r"u_1\ge u_2\ge u_3",
    "M:hgeq25": r"h/J\ge 2.5", "M:h2": r"h/J=2", "M:h3": r"h/J=3", "M:h24": r"h/J\approx 2.4",
    "M:TJrange": r"T/J\in[0.02,3]", "M:hJrange": r"h/J\in[0,3]",
    "M:fhat": r"\hat f", "M:f": r"f", "M:mdef": r"|\partial_T f(x_0)|\ge m>0",
    "D:ham": r"H(h)=J\left(\sigma^x\!\otimes\!\sigma^x+\sigma^y\!\otimes\!\sigma^y+\sigma^z\!\otimes\!\sigma^z\right)+h\left(\sigma^z\!\otimes\!\mathbb{1}+\mathbb{1}\!\otimes\!\sigma^z\right)",
    "D:rho": r"\rho(T,h)=\frac{e^{-H(h)/k_BT}}{\mathrm{Tr}\,e^{-H(h)/k_BT}}",
    "D:smax": r"\mathcal{T}_{ij}=\mathrm{Tr}\!\left[\rho\,(\sigma_i\otimes\sigma_j)\right],\qquad S_{\max}(T,h)=2\sqrt{u_1+u_2}",
    "D:region": r"\Omega=\{(T,h):\,S_{\max}(T,h)>2\},\qquad \partial\Omega=\{(T,h):\,S_{\max}(T,h)=2\}",
    "D:thm": r"d\!\left(x_0,\partial\hat\Omega\right)\;\lesssim\;\frac{\varepsilon}{m},\qquad \varepsilon\ \text{bounding}\ |\hat f-f|\ \text{near}\ x_0",
}

def render_math(tpl):
    used = sorted(set(re.findall(r"\{\{([MD]:[A-Za-z0-9_]+)\}\}", tpl)))
    missing = [u for u in used if u not in MATH]
    assert not missing, f"undefined math tokens: {missing}"
    env = dict(os.environ)
    res = subprocess.run(["node", "tools/render_math.js"], input=json.dumps({u: MATH[u] for u in used}),
                         capture_output=True, text=True, env=env, check=True)
    return json.loads(res.stdout)

def main():
    tpl = Path("tools/template.html").read_text()
    html = tpl
    for k, m in render_math(tpl).items():
        html = html.replace("{{" + k + "}}", m)
    def fig(m):
        stem, title, cap = m.group(1).split("|")
        k = FIGMAP[stem][1]
        return fig_html(stem, f"Figure {k}: {title}", cap)
    html = re.sub(r"\{\{FIG:([^}]+)\}\}", fig, html)
    for k, v in VALUES.items():
        html = html.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[^}]+\}\}", html)
    assert not left, f"unfilled tokens: {left[:5]}"
    Path("index.html").write_text(html)
    print("wrote index.html", len(html), "bytes")

if __name__ == "__main__":
    main()
