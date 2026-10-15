# Blog starter pack

Everything needed to start the blog post on this study without re-deriving anything.

| File | What it holds |
|---|---|
| [`figure-style.md`](figure-style.md) | Figure style decisions: options considered, what we chose and why, the exact spec |
| [`story.md`](story.md) | Suggested outline, the findings worth telling with their numbers and evidence labels, and which figure goes where |
| [`../figures/`](../figures/) | The figures: 5 charts × light/dark × PNG (300 dpi) / SVG. Regenerate with `make figures` |

Sources to quote from:

- [`../results.md`](../results.md): pre-registered verdicts, exploit gallery, limits, field notes, GRPO maths
- [`../learnings.md`](../learnings.md): every lesson with its evidence, tagged by version and source
- [`../v3.md`](../v3.md), [`../v2.md`](../v2.md), [`../sametype.md`](../sametype.md): exploratory follow-ups

**Ground rules for the post** (same as the reports):

- Say which numbers are **pre-registered** (H1, H2, H5 verdicts) and which are **exploratory**
  (H3, v2, v3, ID test). Never present an exploratory number as a verdict.
- Don't compare numbers across versions (v1 / v2 / v3) without saying they use different tasks or settings.
- No employer names or cloud-account details (teamspace, cluster, rates). Compute is stated in
  machine-hours ("~41 hours on 8× H200 machines").
