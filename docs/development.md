# Development

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m build
python tools/check_release.py --dist dist
```

Tests cover analytical contact times, no-contact/current-contact boundaries,
contact between input knots, rotation-only contact, actor exchange, rigid
coordinate transformations, heading interpolation, weighted aggregation,
adapters and command-line examples. The zero-yaw recovery test uses an
independent closed-form translation reference, not a second call to the same
solver. Experiment tests distinguish pooled positive frames from event maxima
and per-instant averaging. They also reject partial/failed runs, mismatched
manifests, and inconsistent score/status combinations.

The release check installs the wheel into a temporary target and runs the tests
and examples from the extracted source archive. It checks that GEI is imported
from that wheel, rather than the editable checkout. Documentation-link tests
also verify that the framework image, GIFs and media notice ship together in
the source archive. The wheel intentionally contains only the library and its
package metadata; source examples and media are available in the repository
and source archive. Use a fresh build-output directory when checking a new version.

The public library must not depend on research directories, server paths,
PyTorch, plotting libraries or datasets at import time. Generated outputs belong
in `outputs/`; do not commit model weights, raw logs, bytecode or credentials.

For a solver change, add a deterministic regression case and update numerical
conventions. Do not tune numerical controls to obtain a desired AUPRC ranking.
Preserve frozen experiment results and their solver fingerprints.

Regenerate the README illustration:

```bash
python examples/plot_weighted_example.py
```

Use the same computed values as the runnable examples. The illustration uses
Arial when installed and a sans-serif fallback otherwise.
