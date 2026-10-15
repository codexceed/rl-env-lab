# Blog figures (v1, pre-registered runs)

Built by `make figures` (`eval/blog_figures.py`) from the same eval data as `reports/results.md`.
Each figure comes in a light and a dark theme, as a 300 dpi PNG and an SVG.

| File | Message |
|---|---|
| `1-training-reward` | The leaky reward is maxed out within 50 steps |
| `2-true-success` | Robust training builds skill that transfers; leaky training stays at the untrained level |
| `3-exploit` | Leaky-trained models fool the grader on about two in three new tasks |
| `4-final-by-size` | The bigger the model, the more real skill a leaky reward costs |
| `5-early-warning` | A second grader logged alongside shows the hack long before any eval |

Font: Inter if installed, else Avenir Next / Helvetica Neue. SVG text is converted to paths, so it renders the same everywhere.
