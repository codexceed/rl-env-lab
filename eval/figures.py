"""Report and blog figures: v1 results, one message per chart, light and dark themes.

Usage::

    python eval/figures.py                  # evals/ + runs/ -> reports/figures/ (`make figures`)
    python eval/figures.py --out some/dir

`eval/compare.py` (`make report`) calls `make_figures` too, so the report and these files stay in step.
Per figure it writes ``<name>-light.{png,svg}`` and ``<name>-dark.{png,svg}``; PNGs are 300 dpi.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # run as a script: make `eval` and `train` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.compare import (  # figures.py is only imported lazily by compare.py
    REWARDS,
    Analysis,
    Dataset,
    RunInfo,
    _eval_curve,
    _rolling,
    _seed_infos,
    _sizes,
    analyze,
    load_dataset,
)
from eval.metrics import Metric

FONTS = ["Inter", "Avenir Next", "Helvetica Neue", "Helvetica", "DejaVu Sans"]
SOURCE = "Qwen3.5 · GRPO, 1,000 steps · 3 seeds per condition · OOD test: 250 tasks \N{MULTIPLICATION SIGN} 4 samples"


@dataclass(frozen=True)
class Theme:
    """Colours for one figure theme (dataviz reference palette, validated for each surface)."""

    name: str
    bg: str
    ink: str
    muted: str
    faint: str
    grid: str
    robust: str
    leaky: str

    def reward(self, reward: str) -> str:
        """Series colour of a reward condition."""
        return self.robust if reward == "robust" else self.leaky


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#2a78d6", "#eb6834")
DARK = Theme("dark", "#1a1a19", "#f0efec", "#c3c2b7", "#8a8984", "#2e2e2b", "#3987e5", "#d95926")
THEMES = (LIGHT, DARK)


def _style(plt: Any, t: Theme) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONTS,
            "font.size": 12,
            "text.color": t.ink,
            "axes.labelcolor": t.muted,
            "axes.labelsize": 11.5,
            "axes.edgecolor": t.grid,
            "axes.facecolor": t.bg,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": t.grid,
            "grid.linewidth": 0.9,
            "xtick.color": t.muted,
            "ytick.color": t.muted,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "ytick.left": False,
            "xtick.major.size": 0,
            "figure.facecolor": t.bg,
            "savefig.facecolor": t.bg,
            "svg.fonttype": "path",
            "lines.solid_capstyle": "round",
        }
    )


def _pct_axis(ax: Any, lo: float, hi: float, step: float) -> None:
    ax.set_ylim(lo - (hi - lo) * 0.02, hi + (hi - lo) * 0.05)  # headroom so lines at the limit don't touch the edge
    ticks = np.arange(lo, hi + 1e-9, step)
    ax.set_yticks(ticks, [f"{round(v * 100)}%" for v in ticks])


def _header(fig: Any, t: Theme, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=17, fontweight="bold", color=t.ink)
    fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=12.5, color=t.muted)


def _footer(fig: Any, t: Theme, note: str = SOURCE) -> None:
    fig.text(0.012, 0.01, note, ha="left", va="bottom", fontsize=9.5, color=t.faint)


def _end_labels(ax: Any, items: Sequence[tuple[float, float, str, str]], min_gap: float) -> None:
    """Direct labels at the right end of lines, nudged apart vertically. Items: (x, y, text, colour)."""
    placed: list[float] = []
    for x, y, text, color in sorted(items, key=lambda i: i[1]):
        y_lab = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(y_lab)
        ax.annotate(
            text,
            xy=(x, y_lab),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=color,
            linespacing=1.0,
            annotation_clip=False,
        )


def _curves(
    data: Dataset, size: str, reward: str, series: Callable[[RunInfo], tuple[list[float], list[float]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Mean, min and max across seeds at the x values every seed has."""
    curves = [c for c in (series(i) for i in _seed_infos(data, size, reward)) if c[0]]
    if not curves:
        return None
    common = sorted(set.intersection(*(set(xs) for xs, _ in curves)))
    mat = np.array([[dict(zip(xs, ys, strict=True))[x] for x in common] for xs, ys in curves])
    return np.array(common), mat.mean(0), mat.min(0), mat.max(0)


def _train_series(key: str, window: int = 25) -> Callable[[RunInfo], tuple[list[float], list[float]]]:
    def f(info: RunInfo) -> tuple[list[float], list[float]]:
        if not info.train_log or key not in info.train_log[0]:  # older logs lack the off-policy grader
            return [], []
        xs = [float(r["step"]) for r in info.train_log]
        return xs, _rolling(np.array([r[key] for r in info.train_log], dtype=float), window).tolist()

    return f


def _eval_series(data: Dataset, metric: Metric) -> Callable[[RunInfo], tuple[list[float], list[float]]]:
    def f(info: RunInfo) -> tuple[list[float], list[float]]:
        steps, vals = _eval_curve(data, info, metric)
        return [float(s) for s in steps], vals

    return f


def _panels(plt: Any, n: int, height: float = 4.6) -> tuple[Any, list[Any]]:
    fig, axes = plt.subplots(1, n, figsize=(13, height), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.74, bottom=0.17, wspace=0.28)
    return fig, list(np.atleast_1d(axes))


def _size_title(ax: Any, t: Theme, size: str) -> None:
    ax.set_title(size.upper(), loc="left", fontsize=13, fontweight="bold", color=t.ink, pad=10)


def _over_steps(
    plt: Any,
    data: Dataset,
    t: Theme,
    *,
    series: Callable[[str, str], Callable[[RunInfo], tuple[list[float], list[float]]]],
    base: Metric | None,
    ylim: tuple[float, float, float],
    title: str,
    subtitle: str,
) -> Any:
    sizes = _sizes(data)
    fig, axes = _panels(plt, len(sizes))
    for ax, size in zip(axes, sizes, strict=True):
        _size_title(ax, t, size)
        labels = []
        for reward in REWARDS:
            c = _curves(data, size, reward, series(size, reward))
            if c is None:
                continue
            xs, mean, lo, hi = c
            col = t.reward(reward)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0)
            ax.plot(xs, mean, color=col, lw=2.6)
            labels.append((xs[-1], float(mean[-1]), reward.capitalize(), col))
        b = data.base.get(size)
        if base is not None and b is not None:
            y = b.mean(base)
            ax.axhline(y, color=t.faint, lw=1.4, ls=(0, (2, 2.5)))
            labels.append((1000.0, y, "Base", t.faint))
        _pct_axis(ax, *ylim)
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xlabel("Training step")
        _end_labels(ax, labels, min_gap=(ylim[1] - ylim[0]) * 0.075)
    _header(fig, t, title, subtitle)
    _footer(fig, t, SOURCE + " · line = mean of seeds, band = seed range")
    return fig


def fig_training_reward(plt: Any, data: Dataset, t: Theme) -> Any:
    """Each run's own training reward over steps."""
    return _over_steps(
        plt,
        data,
        t,
        series=lambda size, reward: _train_series("reward"),
        base=None,
        ylim=(0.0, 1.0, 0.25),
        title="The leaky reward is maxed out within 50 steps",
        subtitle="Training reward: the score each run is optimising, on its training tasks (25-step mean)",
    )


def fig_true_success(plt: Any, data: Dataset, t: Theme) -> Any:
    """OOD-test true success over steps."""
    return _over_steps(
        plt,
        data,
        t,
        series=lambda size, reward: _eval_series(data, "true_success"),
        base="true_success",
        ylim=(0.0, 1.0, 0.2),
        title="Robust training builds skill that transfers; leaky training stays at the untrained level",
        subtitle="True success on new kinds of task (OOD test), checked against the full simulator state",
    )


def fig_exploit(plt: Any, data: Dataset, t: Theme) -> Any:
    """OOD-test exploit rate over steps."""
    return _over_steps(
        plt,
        data,
        t,
        series=lambda size, reward: _eval_series(data, "exploit"),
        base="exploit",
        ylim=(0.0, 0.8, 0.2),
        title="Leaky-trained models fool the grader on about two in three new tasks",
        subtitle="Exploit rate on the OOD test: passes the surface-only grader but leaves the wrong state",
    )


def fig_early_warning(plt: Any, data: Dataset, t: Theme) -> Any:
    """Leaky runs: the reward they optimise vs the robust score on the same training batches."""
    sizes = _sizes(data)
    fig, axes = _panels(plt, len(sizes))
    for ax, size in zip(axes, sizes, strict=True):
        _size_title(ax, t, size)
        for key, col, text in (
            ("leaky_reward_mean", t.leaky, "Leaky reward (optimised)"),
            ("robust_reward_mean", t.ink, "Robust score (logged only)"),
        ):
            c = _curves(data, size, "leaky", _train_series(key, 50))
            if c is None:
                continue
            xs, mean, lo, hi = c
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(xs, mean, color=col, lw=2.6)
            if size == sizes[0]:
                y = float(np.interp(600, xs, mean))
                ax.text(600, y + 0.07, text, ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=col)
        _pct_axis(ax, 0.0, 1.0, 0.25)
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xlabel("Training step")
    _header(
        fig,
        t,
        "A second grader logged alongside shows the hack long before any eval",
        "Leaky-reward runs only, on their own training batches (50-step mean): what they optimise vs what is true",
    )
    _footer(fig, t, SOURCE + " · line = mean of seeds, band = seed range")
    return fig


def fig_final_by_size(plt: Any, an: Analysis, t: Theme) -> Any:
    """Final checkpoint: true success and exploit by size, robust vs leaky vs base."""
    data = an.data
    sizes = _sizes(data)
    cond = {(c.size, c.reward): c for c in an.conditions if c.cells}  # partial data: some conditions empty
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.76, bottom=0.15, wspace=0.3)
    x = np.arange(len(sizes))
    panels: tuple[tuple[Any, Metric, str], ...] = (
        (axes[0], "true_success", "True success"),
        (axes[1], "exploit", "Exploit rate"),
    )
    for ax, metric, label in panels:
        ax.set_title(label, loc="left", fontsize=13, fontweight="bold", color=t.ink, pad=10)
        labels = []
        for reward, off in (("robust", -0.07), ("leaky", 0.07)):
            col = t.reward(reward)
            pts = [(k + off, cond[(s, reward)].cells[metric]) for k, s in enumerate(sizes) if (s, reward) in cond]
            if not pts:
                continue
            ks = [k for k, _ in pts]
            ys = [c.point for _, c in pts]
            for k, c in pts:
                ax.plot([k, k], c.ci, color=col, lw=5, alpha=0.3 if t.name == "light" else 0.45, solid_capstyle="round")
            ax.plot(ks, ys, color=col, lw=2.4, marker="o", ms=9, markeredgecolor=t.bg, markeredgewidth=2)
            labels.append((ks[-1] + 0.08, ys[-1], reward.capitalize(), col))
        bks = [k for k, s in enumerate(sizes) if s in data.base]
        if bks:
            bys = [data.base[sizes[k]].mean(metric) for k in bks]
            ax.plot(bks, bys, color=t.faint, lw=1.6, ls=(0, (2, 2.5)), marker="o", ms=6, mfc=t.bg)
            labels.append((bks[-1] + 0.12, bys[-1], "Base", t.faint))
        if metric == "true_success":
            for k, s in enumerate(sizes):
                r, lk = cond.get((s, "robust")), cond.get((s, "leaky"))
                if r and lk:
                    gap = (r.cells[metric].point - lk.cells[metric].point) * 100
                    ax.annotate(
                        f"{gap:+.0f} pts",
                        xy=(k, r.cells[metric].ci[1]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=11,
                        color=t.muted,
                    )
        ax.set_xticks(x, [s.upper() for s in sizes])
        ax.set_xlim(-0.35, len(sizes) - 0.65)
        ax.set_xlabel("Model size")
        _pct_axis(ax, 0.0, 0.8, 0.2)
        _end_labels(ax, labels, min_gap=0.06)
    _header(
        fig,
        t,
        "The bigger the model, the more real skill a leaky reward costs",
        "Final checkpoint, OOD test. Labels: robust minus leaky true success. Dot = mean of 3 seeds, bar = 95% CI",
    )
    _footer(fig, t)
    return fig


FIGURES = ("1-training-reward", "2-true-success", "3-exploit", "4-final-by-size", "5-early-warning")


def make_figures(an: Analysis, out_dir: str | Path, themes: Sequence[Theme] = THEMES) -> list[Path]:
    """Write every figure in every theme as PNG (300 dpi) and SVG. Raises ImportError without matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    builders: dict[str, Callable[[Theme], Any]] = {
        "1-training-reward": lambda t: fig_training_reward(plt, an.data, t),
        "2-true-success": lambda t: fig_true_success(plt, an.data, t),
        "3-exploit": lambda t: fig_exploit(plt, an.data, t),
        "4-final-by-size": lambda t: fig_final_by_size(plt, an, t),
        "5-early-warning": lambda t: fig_early_warning(plt, an.data, t),
    }
    assert tuple(builders) == FIGURES
    paths = []
    for t in themes:
        _style(plt, t)
        for name, build in builders.items():
            fig = build(t)
            for ext in ("png", "svg"):
                p = out / f"{name}-{t.name}.{ext}"
                fig.savefig(p, dpi=300, metadata={"Date": None} if ext == "svg" else None)
                paths.append(p)
            plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Build the blog figures from the eval data.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        The process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", default="evals")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    an = analyze(load_dataset(a.evals, a.runs), n_boot=a.n_boot)
    for p in make_figures(an, a.out):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
