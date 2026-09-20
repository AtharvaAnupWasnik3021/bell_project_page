"""Website-only figure: ground-truth S_max landscape from figure1_data.npz.
Uses only archived data (no retraining). Run from the repository root."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10, "mathtext.fontset": "cm"})
d = np.load("figure1_data.npz")
h, T, S = d["h_vals"], d["T_vals"], d["S_max"]      # S rows = T (log-spaced), cols = h
def boundary_points(h, T, S, level=2.0):   # same logic as experiment_core.extract_boundary_points
    pts = []
    for j, hh in enumerate(h):
        sign = S[:, j] - level
        for i0 in np.where(np.diff(np.sign(sign)) != 0)[0]:
            y0, y1 = sign[i0], sign[i0 + 1]
            if y1 == y0:
                continue
            t0, t1 = np.log(T[i0]), np.log(T[i0 + 1])
            pts.append((np.exp(t0 + (0 - y0) * (t1 - t0) / (y1 - y0)), hh))
    return np.array(pts)

P = boundary_points(h, T, S)
wedge = P[:, 1] >= 2.5
fig, ax = plt.subplots(figsize=(5.8, 4.3))
im = ax.pcolormesh(h, T, S, cmap="cividis", shading="auto", rasterized=True)
ax.scatter(P[~wedge, 1], P[~wedge, 0], s=5, color="white", label=f"true boundary ({(~wedge).sum()} pts)")
ax.scatter(P[wedge, 1], P[wedge, 0], s=9, color="tab:red", label=f"high-field wedge, $h/J\\geq2.5$ ({wedge.sum()} pts)")
ax.set_yscale("log")
ax.set_xlabel(r"$h/J$"); ax.set_ylabel(r"$T/J$")
ax.set_title(r"Ground-truth $S_{\max}(T,h)$ and extracted $S_{\max}=2$ boundary")
ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
fig.colorbar(im, ax=ax, label=r"$S_{\max}$")
fig.tight_layout()
fig.savefig("assets/figures/figW_ground_truth_landscape.png", dpi=200, bbox_inches="tight")
print("boundary pts", len(P), "wedge", wedge.sum())
