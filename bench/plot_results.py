"""Draws the README charts from the benchmark CSVs.

Usage: python bench/plot_results.py
Reads  results/bench_cpp_vs_python.csv, results/bench_batch_sweep.csv
Writes results/cpp_vs_python.png, results/batch_sweep.png

Each point/bar is the mean over repeats; whiskers show the min..max range, so the
run-to-run spread is visible instead of hidden.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Validated categorical palette (slots 1 and 2) and neutral inks.
CPP_COLOR = "#2a78d6"     # blue: C++ server, in every chart
PY_COLOR = "#eb6834"      # orange: Python baseline, in every chart
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def summarize(df: pd.DataFrame, by: str, col: str, sort: bool = False) -> pd.DataFrame:
    g = df.groupby(by, sort=sort)[col]
    return pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max()})


def whiskers(s) -> list:
    """[below, above] distances from the mean, clipped at 0: when every repeat is
    identical, mean - min can come out as -1e-15 from float rounding."""
    return [np.maximum(s["mean"] - s["min"], 0), np.maximum(s["max"] - s["mean"], 0)]


def bar_with_range(ax, x, s, color, width):
    """Bar at the mean plus a min..max whisker over the repeats."""
    lo, hi = whiskers(s)
    err = [[lo], [hi]]
    ax.bar(x, s["mean"], width=width, color=color, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, s["mean"], yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)


def plot_cpp_vs_python() -> None:
    df = pd.read_csv(RESULTS / "bench_cpp_vs_python.csv")
    servers = list(dict.fromkeys(df["server"]))  # keep run order: cpp first
    colors = {s: CPP_COLOR if s == "cpp" else PY_COLOR for s in servers}
    names = {s: "C++ server" if s == "cpp" else f"Python baseline ({s.split('_t')[-1]} threads)"
             for s in servers}
    n_rep = df["repeat"].nunique()
    users = int(df["users"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})

    # Panel 1: throughput
    for i, s in enumerate(servers):
        st = summarize(df[df["server"] == s], "server", "images_per_s").iloc[0]
        bar_with_range(ax1, i, st, colors[s], 0.6)
        ax1.text(i, st["max"], f"{st['mean']:.1f}", ha="center", va="bottom", color=INK,
                 fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(servers)), [names[s].split(" (")[0] for s in servers])
    ax1.set_ylabel("images / second (higher is better)")
    ax1.set_title("Throughput", loc="left", fontweight="bold")
    ax1.margins(y=0.15)

    # Panel 2: latency percentiles, grouped by percentile
    width = 0.36
    for i, s in enumerate(servers):
        sub = df[df["server"] == s]
        for j, col in enumerate(["p50_ms", "p99_ms"]):
            st = summarize(sub, "server", col).iloc[0]
            x = j + (i - (len(servers) - 1) / 2) * width
            bar_with_range(ax2, x, st, colors[s], width)
            ax2.text(x, st["max"], f"{st['mean']:.0f}", ha="center", va="bottom", color=INK,
                     fontsize=9)
    ax2.set_xticks([0, 1], ["p50", "p99"])
    ax2.set_ylabel("latency, ms (lower is better)")
    ax2.set_title("Latency", loc="left", fontweight="bold")
    ax2.margins(y=0.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in servers]
    ax2.legend(handles, [names[s] for s in servers], frameon=False, loc="upper left")

    fig.suptitle(f"resnet50 FP32, {users} concurrent users (mean of {n_rep} runs, "
                 f"whiskers = min..max)", x=0.01, ha="left", color=INK_2, fontsize=10)
    fig.tight_layout()
    out = RESULTS / "cpp_vs_python.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_batch_sweep() -> None:
    df = pd.read_csv(RESULTS / "bench_batch_sweep.csv")
    tp = summarize(df, "max_batch_size", "images_per_s", sort=True)
    p99 = summarize(df, "max_batch_size", "p99_ms", sort=True)
    n_rep = df["repeat"].nunique()
    users = int(df["users"].iloc[0])

    # Two panels sharing the x axis (batch size): throughput and p99 latency are
    # different units, so they get separate y axes rather than one dual-axis chart.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [str(bs) for bs in tp.index]
    for ax, s, unit, fmt in [(ax1, tp, "images / second (higher is better)", "{:.1f}"),
                             (ax2, p99, "p99 latency, ms (lower is better)", "{:.0f}")]:
        for i, bs in enumerate(s.index):
            row = s.loc[bs]
            bar_with_range(ax, i, row, CPP_COLOR, 0.6)
            ax.text(i, row["max"], fmt.format(row["mean"]), ha="center", va="bottom",
                    color=INK, fontsize=10)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("max_batch_size")
        ax.set_ylabel(unit)
        # Magnitudes start at 0: a zoomed axis would make a 5% difference look dramatic.
        ax.set_ylim(0, s["max"].max() * 1.18)
    ax1.set_title("Throughput", loc="left", fontweight="bold")
    ax2.set_title("Tail latency", loc="left", fontweight="bold")
    fig.suptitle(f"C++ server, resnet50 FP32, {users} users, max_wait_ms 5 "
                 f"(mean of {n_rep} runs, whiskers = min..max)",
                 x=0.01, ha="left", color=INK_2, fontsize=10)
    fig.tight_layout()
    out = RESULTS / "batch_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_cpp_vs_python()
    plot_batch_sweep()
