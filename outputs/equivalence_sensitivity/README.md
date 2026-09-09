# Post-hoc interval-containment audit

Run `python analysis/audit_equivalence_intervals.py` from the repository root.
Uses unrounded released joint interval endpoints; source hashes are recorded.
No models are refitted and no running GPU experiment is modified.

The .01 absolute Macro-F1 tolerance was introduced after prior results were
seen. It is a one-percentage-point reporting sensitivity, not a preregistered
or clinically validated SESOI. Report .005/.01/.02 together, without choosing
the margin that produces a favorable conclusion. These margins are separate
from the compact-panel noninferiority tolerance.

Strict containment and zero inclusion are distinct, potentially overlapping
diagnostics. A boundary-touching interval is not called contained. Failure to
contain is inconclusive about equivalence, not proof of a meaningful effect.
Spanning both -epsilon and +epsilon indicates insufficient precision, not a
formal power calculation. Retained joint 95% intervals mix between-fit and
held-out-subject variability; do not convert their endpoints into TOST p-values.
There is no simultaneous family-wide equivalence guarantee.

The main paper summary uses twelve T=6400 comparisons and twelve adjacent
external frontier comparisons. CSV/JSON also retain the other completed rows,
including direct 32-to-128 comparisons, without silently counting them as
adjacent frontier steps. The ongoing GPU A-D experiment is not included.
