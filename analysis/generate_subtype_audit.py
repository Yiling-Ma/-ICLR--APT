#!/usr/bin/env python3
"""Generate subtype audit tables from pooled patient-disjoint predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_SPECS = [
    ("DropCascade", None, r"\ourmethod{}"),
    ("Flat FT-style Transformer", "matched:flat_ce", "Flat FT-style Transformer"),
    ("FT-style Transformer + HCE", "matched:hce", r"FT-style Transformer $+$ HCE"),
    ("XGBoost", "classical:xgboost", "XGBoost"),
    ("Logistic Regression", "classical:logistic_regression", "Logistic Regression"),
    ("Linear SVM", "classical:linear_svm", "Linear SVM"),
    ("Random Forest", "classical:random_forest", "Random Forest"),
    ("Reference Correlation", "classical:reference_correlation", "Reference Correlation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropcascade-predictions", type=Path, required=True)
    parser.add_argument("--classical-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/generated"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, str]:
    mapping = {}
    for line in path.read_text().splitlines():
        if line.startswith("  ") and ": " in line:
            subtype, lineage = line.strip().split(": ", 1)
            mapping[subtype] = lineage
    if not mapping:
        raise ValueError(f"No subtype mapping found in {path}")
    return mapping


def derive_mapping(frame: pd.DataFrame) -> dict[str, str]:
    pairs = frame[["fine_true", "coarse_true"]].astype(str).drop_duplicates()
    if pairs.groupby("fine_true")["coarse_true"].nunique().max() != 1:
        raise ValueError("Each subtype must map to exactly one lineage")
    return pairs.set_index("fine_true")["coarse_true"].to_dict()


def per_class_f1(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    scores = []
    for label in labels:
        true = frame["y_true"] == label
        pred = frame["y_pred"] == label
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return np.asarray(scores)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def partial_spearman(
    outcome: np.ndarray, predictor: np.ndarray, control: np.ndarray
) -> float:
    """Correlate rank residuals after linear adjustment for one control."""
    ranked = np.column_stack(
        [pd.Series(values).rank(method="average").to_numpy() for values in (outcome, predictor, control)]
    )
    design = np.column_stack([np.ones(len(ranked)), ranked[:, 2]])
    outcome_residual = ranked[:, 0] - design @ np.linalg.lstsq(
        design, ranked[:, 0], rcond=None
    )[0]
    predictor_residual = ranked[:, 1] - design @ np.linalg.lstsq(
        design, ranked[:, 1], rcond=None
    )[0]
    return float(np.corrcoef(outcome_residual, predictor_residual)[0, 1])


def patient_bootstrap_f1(
    frame: pd.DataFrame,
    labels: list[str],
    patients: list[str],
    draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return classwise percentile intervals from patient-clustered resamples."""
    patient_index = {patient: index for index, patient in enumerate(patients)}
    label_index = {label: index for index, label in enumerate(labels)}
    patient_ids = frame["sample_id"].astype(str).map(patient_index).to_numpy()
    true_ids = frame["y_true"].astype(str).map(label_index).to_numpy()
    pred_ids = frame["y_pred"].astype(str).map(label_index).to_numpy()
    if np.isnan(patient_ids).any() or np.isnan(true_ids).any() or np.isnan(pred_ids).any():
        raise ValueError("Bootstrap input contains an unknown patient or subtype")
    patient_ids = patient_ids.astype(int)
    true_ids = true_ids.astype(int)
    pred_ids = pred_ids.astype(int)

    shape = (len(patients), len(labels))
    tp = np.zeros(shape, dtype=np.float64)
    fp = np.zeros(shape, dtype=np.float64)
    fn = np.zeros(shape, dtype=np.float64)
    correct = true_ids == pred_ids
    np.add.at(tp, (patient_ids[correct], true_ids[correct]), 1)
    wrong = ~correct
    np.add.at(fp, (patient_ids[wrong], pred_ids[wrong]), 1)
    np.add.at(fn, (patient_ids[wrong], true_ids[wrong]), 1)

    sampled_tp = draws @ tp
    sampled_fp = draws @ fp
    sampled_fn = draws @ fn
    denominator = 2 * sampled_tp + sampled_fp + sampled_fn
    scores = np.divide(
        2 * sampled_tp,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    return np.nanpercentile(scores, 2.5, axis=0), np.nanpercentile(scores, 97.5, axis=0)


def hierarchy_error_null(
    frame: pd.DataFrame,
    mapping: dict[str, str],
    patients: list[str],
    draws: np.ndarray,
) -> dict[str, float]:
    """Compare observed cross-lineage errors with a fixed-marginal shuffle null."""
    errors = frame.loc[frame["y_true"].astype(str) != frame["y_pred"].astype(str)].copy()
    errors["true_lineage"] = errors["y_true"].astype(str).map(mapping)
    errors["pred_lineage"] = errors["y_pred"].astype(str).map(mapping)
    if errors[["true_lineage", "pred_lineage"]].isna().any().any():
        raise ValueError("Error-null input contains a subtype outside the hierarchy")

    patient_index = {patient: index for index, patient in enumerate(patients)}
    lineages = sorted(set(mapping.values()))
    lineage_index = {lineage: index for index, lineage in enumerate(lineages)}
    patient_ids = errors["sample_id"].astype(str).map(patient_index).to_numpy().astype(int)
    true_ids = errors["true_lineage"].map(lineage_index).to_numpy().astype(int)
    pred_ids = errors["pred_lineage"].map(lineage_index).to_numpy().astype(int)

    shape = (len(patients), len(lineages))
    true_counts = np.zeros(shape, dtype=np.float64)
    pred_counts = np.zeros(shape, dtype=np.float64)
    cross_counts = np.zeros(len(patients), dtype=np.float64)
    np.add.at(true_counts, (patient_ids, true_ids), 1)
    np.add.at(pred_counts, (patient_ids, pred_ids), 1)
    np.add.at(cross_counts, patient_ids, true_ids != pred_ids)

    def summarize(
        true_margin: np.ndarray,
        pred_margin: np.ndarray,
        cross: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        totals = true_margin.sum(axis=-1)
        observed = cross / totals
        null = 1 - (true_margin * pred_margin).sum(axis=-1) / totals**2
        difference = observed - null
        retention = (null - observed) / null
        return observed, null, difference, retention

    point = summarize(
        true_counts.sum(axis=0, keepdims=True),
        pred_counts.sum(axis=0, keepdims=True),
        np.asarray([cross_counts.sum()]),
    )
    sampled = summarize(draws @ true_counts, draws @ pred_counts, draws @ cross_counts)
    difference_ci = np.percentile(sampled[2], [2.5, 97.5])
    retention_ci = np.percentile(sampled[3], [2.5, 97.5])
    predicted_subtype = errors["y_pred"].astype(str).value_counts(normalize=True)
    true_subtype = errors["y_true"].astype(str).value_counts(normalize=True)
    wrong_conditioned_null = 0.0
    for true_label, true_weight in true_subtype.items():
        denominator = 1 - predicted_subtype.get(true_label, 0.0)
        cross_probability = sum(
            probability
            for predicted_label, probability in predicted_subtype.items()
            if mapping[predicted_label] != mapping[true_label]
        ) / denominator
        wrong_conditioned_null += true_weight * cross_probability
    return {
        "error_count": int(len(errors)),
        "observed_cross_lineage": float(point[0][0]),
        "null_cross_lineage": float(point[1][0]),
        "wrong_conditioned_null": float(wrong_conditioned_null),
        "observed_minus_null": float(point[2][0]),
        "difference_ci_lower": float(difference_ci[0]),
        "difference_ci_upper": float(difference_ci[1]),
        "normalized_retention": float(point[3][0]),
        "retention_ci_lower": float(retention_ci[0]),
        "retention_ci_upper": float(retention_ci[1]),
    }


def standardize(frame: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    return frame[["sample_id", true_col, pred_col]].rename(
        columns={true_col: "y_true", pred_col: "y_pred"}
    )


def load_predictions(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    raw_dropcascade = pd.read_csv(args.dropcascade_predictions)
    dropcascade = standardize(raw_dropcascade, "fine_true", "fine_pred")
    models = {"DropCascade": dropcascade}
    matched_predictions = {}
    for display_name, source, _ in MODEL_SPECS[1:]:
        source_type, directory = source.split(":", 1)
        if source_type == "matched":
            raw = pd.read_csv(
                args.matched_root / directory / "pooled_oof_predictions.csv"
            )
            models[display_name] = standardize(raw, "fine_true", "fine_pred")
            matched_predictions[display_name] = raw
        else:
            path = args.classical_root / directory / "pooled_oof_predictions.csv"
            models[display_name] = pd.read_csv(path)[
                ["sample_id", "y_true", "y_pred"]
            ]

    reference_counts = (
        dropcascade.groupby(["sample_id", "y_true"]).size().sort_index()
    )
    for name, frame in models.items():
        if len(frame) != len(dropcascade):
            raise ValueError(f"{name} has {len(frame)} rows; expected {len(dropcascade)}")
        model_counts = frame.groupby(["sample_id", "y_true"]).size().sort_index()
        if not model_counts.equals(reference_counts):
            raise ValueError(
                f"{name} does not use the same patient-by-subtype evaluation labels"
            )
    return models, raw_dropcascade, matched_predictions


def write_diagnostics_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & $\rho_{c}$ & $\rho_{p}$ & $\rho_{c\mid p}$ & $\rho_{p\mid c}$ & $G$ & $R_{\mathrm{sibling}}$ & \textbf{Cross-lineage} \\",
        r"\midrule",
    ]
    tex_names = dict((name, tex) for name, _, tex in MODEL_SPECS)
    for row in summary.itertuples(index=False):
        lines.append(
            f"{tex_names[row.model]} & {row.rho_cell:.3f} & {row.rho_patient:.3f} & "
            f"{row.partial_rho_cell:.3f} & {row.partial_rho_patient:.3f} & "
            f"{row.granularity_gap:.3f} & {row.sibling_error_rate:.3f} & "
            f"{row.cross_lineage_error_rate:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Exploratory subtype diagnostics on pooled patient-disjoint predictions. $\rho_c$ and $\rho_p$ are marginal Spearman associations of subtype F1 with log cell count and patient coverage; $\rho_{c\mid p}$ and $\rho_{p\mid c}$ are partial Spearman correlations computed by residualizing rank-transformed variables. $G=F_{1}^{\mathrm{coarse}}-F_{1}^{\mathrm{fine}}$. Among errors, $R_{\mathrm{sibling}}$ retains the true parent lineage and Cross-lineage is its complement. With only 27 subtypes and limited variation in patient coverage, these associations are descriptive and do not establish causality.}",
            r"\label{tab:subtype_diagnostics}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_per_subtype_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{longtable}{llrrrrrr}",
        r"\caption{Per-subtype support and pooled out-of-fold F1 with 95\% percentile intervals from 2{,}000 patient-clustered bootstrap resamples. Cells/patient is the mean among patients in which that subtype is observed. Head and Tail denote the seven most and seven least abundant subtypes; all others are Mid.}\label{tab:subtype_per_class} \\",
        r"\toprule",
        r"\textbf{Subtype} & \textbf{Lineage} & \textbf{Cells} & \textbf{Patients} & \textbf{Cells/patient} & \textbf{Regime} & \textbf{DropCascade F1 [95\% CI]} & \textbf{XGB F1 [95\% CI]} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Subtype} & \textbf{Lineage} & \textbf{Cells} & \textbf{Patients} & \textbf{Cells/patient} & \textbf{Regime} & \textbf{DropCascade F1 [95\% CI]} & \textbf{XGB F1 [95\% CI]} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.subtype} & {row.lineage} & {row.cell_count:,} & {row.patient_count} & "
            f"{row.mean_cells_per_patient:.1f} & {row.regime} & "
            f"{row.dropcascade_f1:.3f} [{row.dropcascade_ci_lower:.3f}, {row.dropcascade_ci_upper:.3f}] & "
            f"{row.xgboost_f1:.3f} [{row.xgboost_ci_lower:.3f}, {row.xgboost_ci_upper:.3f}] \\\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    path.write_text("\n".join(lines) + "\n")


def write_tail_table(summary: pd.DataFrame, path: Path) -> None:
    tex_names = dict((name, tex) for name, _, tex in MODEL_SPECS)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Tail Macro-F1} & \textbf{Head Macro-F1} \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"{tex_names[row.model]} & {row.tail_macro_f1:.3f} & {row.head_macro_f1:.3f} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Fine-subtype macro-F1 on the seven least and seven most abundant subtypes under pooled patient-disjoint evaluation. This aggregate is reported as a descriptive complement to the complete per-subtype audit in Table~\ref{tab:subtype_per_class}.}",
            r"\label{tab:tail_head}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_consistency_table(summary: pd.DataFrame, path: Path) -> None:
    tex_names = dict((name, tex) for name, _, tex in MODEL_SPECS)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Coarse--Fine Consistency} \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"{tex_names[row.model]} & {row.consistency:.3f} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Natural coarse--fine consistency on the same 361{,}792-cell, 40-patient pooled out-of-fold artifacts as Table~\ref{tab:main}. For each model, the hard fine prediction is mapped to its parent and compared with the corresponding hard coarse prediction. \emph{Unknown} is excluded. Consistency is supporting evidence rather than predictive superiority because it can be changed by constrained decoding.}",
            r"\label{tab:consistency}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_error_null_table(summary: pd.DataFrame, path: Path) -> None:
    tex_names = dict((name, tex) for name, _, tex in MODEL_SPECS)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Errors} & \textbf{Observed} & \textbf{Null} & \textbf{Wrong-cond. null} & \textbf{Obs.$-$Null [95\% CI]} & \textbf{Norm. retention} \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{tex_names[row.model]} & {row.error_count:,} & "
            f"{row.observed_cross_lineage:.3f} & {row.null_cross_lineage:.3f} & "
            f"{row.wrong_conditioned_null:.3f} & "
            f"{row.observed_minus_null:+.3f} [{row.difference_ci_lower:+.3f}, "
            f"{row.difference_ci_upper:+.3f}] & {row.normalized_retention:+.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Cross-lineage error against fixed-marginal nulls. Null permutes predicted subtype labels within each model's errors while preserving true- and predicted-subtype marginals; its expectation is exact. Wrong-cond. null renormalizes the predicted marginal after excluding each true subtype, preventing chance exact matches but no longer preserving the realized prediction marginal exactly. Confidence intervals for Observed$-$Null use 2{,}000 patient-clustered bootstrap resamples. Normalized retention is $(\mathrm{Null}-\mathrm{Observed})/\mathrm{Null}$; positive values indicate fewer cross-lineage errors than expected.}",
            r"\label{tab:hierarchy_error_null}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    models, raw_dropcascade, matched_predictions = load_predictions(args)
    mapping = load_mapping(args.mapping) if args.mapping else derive_mapping(raw_dropcascade)

    reference = models["DropCascade"]
    counts = reference.groupby("y_true").size().sort_values()
    patient_counts = reference.groupby("y_true")["sample_id"].nunique()
    labels = counts.index.tolist()
    tail = set(labels[:7])
    head = set(labels[-7:])

    per_subtype = pd.DataFrame(
        {
            "subtype": labels,
            "lineage": [mapping[label] for label in labels],
            "cell_count": counts.to_numpy(),
            "patient_count": patient_counts.loc[labels].to_numpy(),
        }
    )
    per_subtype["mean_cells_per_patient"] = (
        per_subtype["cell_count"] / per_subtype["patient_count"]
    )
    per_subtype["regime"] = [
        "Tail" if label in tail else "Head" if label in head else "Mid" for label in labels
    ]
    per_subtype["dropcascade_f1"] = per_class_f1(models["DropCascade"], labels)
    per_subtype["xgboost_f1"] = per_class_f1(models["XGBoost"], labels)

    patients = sorted(reference["sample_id"].astype(str).unique())
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.integers(
        0, len(patients), size=(args.n_boot, len(patients))
    )
    draws = np.stack(
        [np.bincount(indices, minlength=len(patients)) for indices in sampled_indices]
    )
    for model, prefix in (("DropCascade", "dropcascade"), ("XGBoost", "xgboost")):
        lower, upper = patient_bootstrap_f1(models[model], labels, patients, draws)
        per_subtype[f"{prefix}_ci_lower"] = lower
        per_subtype[f"{prefix}_ci_upper"] = upper

    model_rows = []
    for model_name, source, _ in MODEL_SPECS:
        frame = models[model_name]
        f1_values = per_class_f1(frame, labels)
        errors = frame.loc[frame["y_true"] != frame["y_pred"]]
        sibling_rate = float(
            (errors["y_true"].map(mapping) == errors["y_pred"].map(mapping)).mean()
        )
        if source is None:
            coarse_predictions = raw_dropcascade["coarse_pred"].astype(str)
            mapped_fine_predictions = raw_dropcascade["fine_pred"].astype(str).map(mapping)
            coarse_f1 = float(
                per_class_f1(
                    raw_dropcascade.rename(
                        columns={"coarse_true": "y_true", "coarse_pred": "y_pred"}
                    ),
                    sorted(raw_dropcascade["coarse_true"].unique()),
                ).mean()
            )
            fine_f1 = float(f1_values.mean())
        elif source.startswith("matched:"):
            matched = matched_predictions[model_name]
            coarse_predictions = matched["coarse_pred"].astype(str)
            mapped_fine_predictions = matched["fine_pred"].astype(str).map(mapping)
            coarse_f1 = float(
                per_class_f1(
                    standardize(matched, "coarse_true", "coarse_pred"),
                    sorted(matched["coarse_true"].unique()),
                ).mean()
            )
            fine_f1 = float(f1_values.mean())
        else:
            directory = source.split(":", 1)[1]
            coarse_frame = pd.read_csv(
                args.classical_root.parent
                / "coarse"
                / directory
                / "pooled_oof_predictions.csv"
            )
            if len(coarse_frame) != len(frame) or not coarse_frame[
                "sample_id"
            ].astype(str).reset_index(drop=True).equals(
                frame["sample_id"].astype(str).reset_index(drop=True)
            ):
                raise ValueError(f"{model_name} coarse/fine predictions are not row-aligned")
            mapped_truth = frame["y_true"].astype(str).map(mapping).reset_index(drop=True)
            if not coarse_frame["y_true"].astype(str).reset_index(drop=True).equals(
                mapped_truth
            ):
                raise ValueError(f"{model_name} uses an inconsistent subtype parent map")
            coarse_predictions = coarse_frame["y_pred"].astype(str).reset_index(drop=True)
            mapped_fine_predictions = (
                frame["y_pred"].astype(str).map(mapping).reset_index(drop=True)
            )
            coarse_f1 = float(
                per_class_f1(
                    coarse_frame[["sample_id", "y_true", "y_pred"]],
                    sorted(coarse_frame["y_true"].astype(str).unique()),
                ).mean()
            )
            fine_f1 = float(f1_values.mean())
        model_rows.append(
            {
                "model": model_name,
                "rho_cell": spearman(f1_values, np.log(counts.to_numpy())),
                "rho_patient": spearman(f1_values, patient_counts.loc[labels].to_numpy()),
                "partial_rho_cell": partial_spearman(
                    f1_values,
                    np.log(counts.to_numpy()),
                    patient_counts.loc[labels].to_numpy(),
                ),
                "partial_rho_patient": partial_spearman(
                    f1_values,
                    patient_counts.loc[labels].to_numpy(),
                    np.log(counts.to_numpy()),
                ),
                "coarse_macro_f1": coarse_f1,
                "fine_macro_f1": fine_f1,
                "granularity_gap": coarse_f1 - fine_f1,
                "sibling_error_rate": sibling_rate,
                "cross_lineage_error_rate": 1.0 - sibling_rate,
                "consistency": float(
                    (coarse_predictions.reset_index(drop=True) == mapped_fine_predictions).mean()
                ),
                "tail_macro_f1": float(per_class_f1(frame, list(counts.index[:7])).mean()),
                "head_macro_f1": float(per_class_f1(frame, list(counts.index[-7:])).mean()),
            }
        )
    model_summary = pd.DataFrame(model_rows)
    null_rows = []
    for model_name, _, _ in MODEL_SPECS:
        null_rows.append(
            {"model": model_name, **hierarchy_error_null(models[model_name], mapping, patients, draws)}
        )
    error_null = pd.DataFrame(null_rows)

    per_subtype.to_csv(args.output_dir / "subtype_per_class.csv", index=False)
    model_summary.to_csv(args.output_dir / "subtype_model_summary.csv", index=False)
    error_null.to_csv(args.output_dir / "hierarchy_error_null.csv", index=False)
    write_diagnostics_table(model_summary, args.table_dir / "subtype_diagnostics.tex")
    write_per_subtype_table(per_subtype, args.table_dir / "subtype_per_class.tex")
    write_tail_table(model_summary, args.table_dir / "tail_head.tex")
    write_consistency_table(model_summary, args.table_dir / "consistency.tex")
    write_error_null_table(error_null, args.table_dir / "hierarchy_error_null.tex")


if __name__ == "__main__":
    main()
