# Same-future PC/GEI comparison

These scripts implement a reusable, auditable comparison pipeline. Temporal
aggregation follows Section III.D and Table III of GEI_TITS_v1_6, not the
earlier single-instant or event-maximum experimental variants. They are not a
claim that historical paper numbers have been regenerated with this version.

## 1. Export one immutable joint-future bank

Use a NumPy NPZ file without object arrays or pickle. Required fields:

| Field | Shape | Meaning |
|---|---|---|
| a, b | (N,K,T,3) | Joint-indexed x/y/heading poses for each actor |
| times | (T,) | Seconds after the evaluation instant, starting at zero |
| weights | (N,K) | Joint weights, normalized separately for each frame |
| size_a, size_b | (N,2) | Length and width in metres |
| frame_key | (N,) string | Unique event-timestamp key |
| event_id | (N,) string | Source event identifier |
| label | (N,) integer | 0 non-crash, 1 crash |
| lead_time | (N,) float | Positive seconds before impact; use zero for negative events |

K is the number of **joint** hypotheses, e.g. 36 for independent six-by-six
actor modes. Current poses must agree across all K futures. No modes are added,
filtered, resampled or reweighted by the evaluator.

```bash
python experiments/evaluate_future_bank.py futures.npz outputs/scores
```

The manifest records input/output hashes, source fingerprints, solver controls,
environment and record counts. A failed run leaves a partial file and explicit
failure report; it does not produce a successful manifest.
The evaluator checks that the input-bank and GEI-source fingerprints stay
unchanged during scoring, and publishes the completed manifest last.

## 2. Analyze discrimination

Install the optional experiment dependencies:

```bash
python -m pip install -e ".[experiment]"
python experiments/analyze_discrimination.py outputs/scores/frame_scores.csv outputs/analysis
```

By default the analyzer requires the adjacent `manifest.json` to certify a
completed run and match the CSV's SHA-256, frame count and event count. Partial
files and directories containing `failure.json` are rejected. Every row must
include a consistent `status`, `pc` and `gei`; invalid, failed or current-contact
records stop the analysis rather than being silently discarded or capped.
Positive contact probability with zero GEI is valid when InDepth is zero.

For an independently checked external CSV that has no scoring manifest,
`--allow-unverified` is an explicit import option. The report is marked
`unverified_csv_import`; this option never overrides a failed run, a malformed
manifest or inconsistent scores. It is not a repair path for interrupted runs.

Protocol:

- Each negative event uses its maximum over the **complete supplied event**.
- Positive events must contain exactly one frame at each of 0.1,...,2.0 s before impact.
- Pool all five individual positive frame scores within each window: 0.1-0.5,
  0.6-1.0, 1.1-1.5 and 1.6-2.0 s. Do not replace them by a crash-wise maximum.
- Report four pooled-window AUPRC values and their arithmetic mean. This mean
  is not the mean of 20 single-instant AUPRC values.
- AUPRC means non-interpolated average precision.
- Source-event cluster bootstrap is stratified and paired; each sampled crash
  contributes all five frames. The same event multiplicities are used for both
  metrics and all four windows. Default: 10000 replicates.
- Confidence intervals are pointwise percentile intervals, not adjusted for
  multiple comparisons. No significance claims or universal superiority are
  inferred automatically.

The reader must supply the full negative events: the script cannot reconstruct
frames omitted upstream. Do not call a five-frame negative subset a complete
event. A separate registry completeness check is needed for dataset releases.

No classifier or threshold is fitted. No runtime measurements are included.
The outcome depends on the supplied future distribution; a lower GEI AUPRC must
not trigger selective exclusion or solver tuning.

In the paper's Table III cohort, each window has 1716 negative event scores and
1515 positive frame scores from 303 crashes. Both stochastic generators use
128 joint futures over 3 s. These counts and generator settings describe the
paper, not the required size of a user's bank. A six-by-six learned bank uses
the same evaluation interface but is a different experiment.
