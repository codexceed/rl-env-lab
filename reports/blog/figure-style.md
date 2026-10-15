# Figure style decisions

Decided 2026-10-15 with the owner, for figures that serve both `reports/results.md` and the blog post.

## Options considered

The styles are modelled on well-known engineering and research blogs (from familiarity, not a fresh survey).

| | Style | Modelled on | Library | Verdict |
|---|---|---|---|---|
| **A** | Clean research | Distill.pub, AI-lab research posts | matplotlib + custom style | ✅ **Chosen** |
| **B** | Editorial | Our World in Data, FiveThirtyEight, The Economist | matplotlib | ✅ **Titles adopted** (each title states the finding) |
| C | Interactive | Observable, The Pudding, Hugging Face blog | Altair (Vega-Lite + vl-convert), Plotly | Not now: only works where the blog can embed HTML/JS. Revisit if the blog platform supports it |
| D | Paper | arXiv figures | SciencePlots (Nature / IEEE styles) | Rejected: dense and academic, poor fit for a blog |

**Themes:** light **and** dark, as separate files (chosen over light-only or one transparent file).

## Why the first plots failed

Tiny text, 0–1 axes with mostly empty space, a legend far from the lines, 25 tangled seed lines,
glaring white boxes in dark mode, and no title saying what to look at. Earlier still, each figure
grouped different measures (reward, exploit, true success) for one model; the owner wanted **one
measure per figure with models side by side**.

## The spec

| Element | Decision |
|---|---|
| One message per chart | Bold **title states the finding** ("The leaky reward is maxed out within 50 steps"); a muted subtitle says what is plotted |
| Layout | One measure per figure; one panel per model size (0.8B, 2B, 4B), shared y-axis, 13 × 4.6 in |
| Series | Robust vs leaky, plus the untrained base model as a dotted line where it applies |
| Seeds | **Line = mean of 3 seeds, shaded band = range across seeds** (min to max). Not a confidence interval |
| Labels | **Direct labels at the end of each line** (Robust / Leaky / Base), nudged apart; no legend box |
| Axes | Percent ticks; y-limit fitted to the measure (0–80% or 0–100%) with a little headroom; x ticks every 250 steps; horizontal grid only, no left spine |
| Final-by-size chart | Dot = mean of seeds, thick translucent bar = 95% CI, robust and leaky offset slightly; gap labels ("+22 pts") |
| Smoothing | Training curves: 25-step rolling mean (50 for the early-warning chart). Eval curves: raw, one point per 50 steps |
| Footer | Source line: model, steps, seeds, eval size, and what line/band mean |
| Colours | Robust blue, leaky orange, base grey. Light: `#2a78d6` / `#eb6834` on `#fcfcfb`. Dark: `#3987e5` / `#d95926` on `#1a1a19`. Both pairs pass the colour-blind separation and contrast checks (dataviz validator) |
| Text colours | Ink `#0b0b0b` / `#f0efec`, muted `#52514e` / `#c3c2b7`; text never takes a series colour except the direct labels |
| Font | Inter if installed, else Avenir Next, then Helvetica Neue. Title 17 pt bold, panel titles 13 pt bold, body 11–12.5 pt |
| Output | Per figure and theme: PNG at 300 dpi (3900 px wide) and SVG with text as outlines (renders the same everywhere) |
| Naming | `<n>-<name>-<light|dark>.<png|svg>`, numbered in story order |

## The figures

| File | Title | Used in `results.md` |
|---|---|---|
| `1-training-reward` | The leaky reward is maxed out within 50 steps | H2 |
| `2-true-success` | Robust training builds skill that transfers; leaky training stays at the untrained level | H2 |
| `3-exploit` | Leaky-trained models fool the grader on about two in three new tasks | H2 |
| `4-final-by-size` | The bigger the model, the more real skill a leaky reward costs | H5 |
| `5-early-warning` | A second grader logged alongside shows the hack long before any eval | Field notes |

## How to use and change them

- **Regenerate:** `make figures` (figures only) or `make report` (report + figures). Code:
  [`eval/figures.py`](../../eval/figures.py); style constants are at the top (`Theme`, `FONTS`).
- **Embed with theme switching** (GitHub and most markdown sites):

  ```html
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/2-true-success-dark.png">
    <img alt="..." src="figures/2-true-success-light.png">
  </picture>
  ```

- **Write alt text** that states the finding, not the chart type.
- **Inter:** installing it (rsms.me/inter) and rerunning `make figures` switches the font automatically.
- **New charts** should follow the spec above: a finding as title, direct labels, seed bands, both themes.
