"""Create a publication-style figure using measured per-trajectory errors only."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [
    ("C6", "Six substeps", "#555555", "s", "-"),
    ("C12", "Twelve substeps", "#157B77", "^", "-"),
    ("legacy_fixed", "Fixed field filter", "#C45A20", "D", "--"),
    ("field_dt2", "Step-tapered field filter", "#8966B5", "o", ":"),
    ("increment_dt", "Step-conditioned increment", "#1969AD", "o", "-"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text())["summary"]
    fig, ax = plt.subplots(figsize=(7.1, 4.25), dpi=180)
    for key, label, color, marker, ls in METHODS:
        data = sorted((r for r in rows if r["method"] == key),
                      key=lambda r: -r["dt"])
        x = [r["dt"] / 1e-4 for r in data]
        y = [r["error_mean"] for r in data]
        sd = [r["error_sd"] for r in data]
        ax.errorbar(x, y, yerr=sd, label=label, color=color,
                    linestyle=ls, marker=marker, markersize=5.2,
                    linewidth=1.7, capsize=2, elinewidth=.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xticks([1, .5, .25, .125],
                  [r"$h_0$", r"$h_0/2$", r"$h_0/4$", r"$h_0/8$"])
    ax.set_xlabel(r"Macro step $h$ ($h_0=10^{-4}$)")
    ax.set_ylabel("Mean relative field error")
    ax.grid(True, which="major", alpha=.22)
    ax.grid(True, which="minor", alpha=.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015),
              ncol=2, fontsize=7.8, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print(args.out)


if __name__ == "__main__":
    main()
