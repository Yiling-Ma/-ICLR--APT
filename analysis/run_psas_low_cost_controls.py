"""Run patient-level PSAS controls requested for the APT-Bench audit."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from apt_jepa.data.dataset import load_merged_dataframe
from apt_jepa.data.preprocessing import apply_standardizer, fit_standardizer


RANDOM_BUDGETS = [5, 10, 20]
CELLCOUNT_BUDGETS = [5, 10, 20, 293]
CELLCOUNTS = [100, 500, 1000, 5000, None]
N_FOLDS = 5
_STATE = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-dir", default="cell_JEPA/outputs/dropcascade_kfold5_splits")
    parser.add_argument("--output-dir", default="cell_JEPA/outputs/psas_controls_v2")
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--cell-seeds", type=int, default=50)
    parser.add_argument("--shap-cells-per-patient", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--sections",
        default="random,cellcount,shap",
        help="Comma-separated subset of random, cellcount, and shap.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use only B=5 for a quick code-path test.")
    return parser.parse_args()


def patient_vote(cell_pred: np.ndarray, sample_ids: np.ndarray) -> dict[str, int]:
    frame = pd.DataFrame({"sample_id": sample_ids, "pred": cell_pred})
    return frame.groupby("sample_id")["pred"].agg(lambda s: int(s.mode().iloc[0])).to_dict()


def patient_metrics(prediction: dict[str, int], truth: dict[str, int]) -> tuple[float, float]:
    patients = sorted(prediction)
    y_true = np.asarray([truth[p] for p in patients])
    y_pred = np.asarray([prediction[p] for p in patients])
    return (
        float(f1_score(y_true, y_pred, average="macro")),
        float(accuracy_score(y_true, y_pred)),
    )


def fit_predict(
    x_norm: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_idx: np.ndarray,
    model_name: str,
    seed: int,
) -> np.ndarray:
    if model_name == "LR":
        model = LogisticRegression(max_iter=500, random_state=seed)
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=1,
            random_state=seed,
            eval_metric="mlogloss",
            tree_method="hist",
        )
    else:
        raise ValueError(model_name)
    model.fit(x_norm[train_idx][:, feature_idx], y[train_idx])
    return model.predict(x_norm[test_idx][:, feature_idx]).astype(int)


def rank_lr(x_norm: np.ndarray, y: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    model = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=0.1,
        max_iter=300,
        random_state=42,
    )
    model.fit(x_norm[train_idx], y[train_idx])
    importance = np.abs(model.coef_).sum(axis=0)
    return np.argsort(-importance)


def rank_fold_task(fold: int) -> np.ndarray:
    return rank_lr(_STATE["x_norm"][fold], _STATE["y"], _STATE["train_idx"][fold])


def evaluate_panel(feature_by_fold: list[np.ndarray], model_name: str, seed: int) -> tuple[float, float]:
    state = _STATE
    pooled = {}
    for fold in range(N_FOLDS):
        pred = fit_predict(
            state["x_norm"][fold],
            state["y"],
            state["train_idx"][fold],
            state["test_idx"][fold],
            feature_by_fold[fold],
            model_name,
            seed,
        )
        pooled.update(patient_vote(pred, state["sample_ids"][state["test_idx"][fold]]))
    return patient_metrics(pooled, state["patient_truth"])


def random_panel_task(task: tuple[int, int]) -> list[dict]:
    budget, repeat = task
    state = _STATE
    if budget == state["n_features"]:
        feature_idx = np.arange(state["n_features"])
    else:
        rng = np.random.default_rng(100_000 + budget * 1_000 + repeat)
        feature_idx = np.sort(rng.choice(state["n_features"], size=budget, replace=False))
    panels = [feature_idx] * N_FOLDS
    rows = []
    for model_name in ["LR", "XGBoost"]:
        macro_f1, accuracy = evaluate_panel(panels, model_name, 10_000 + repeat)
        rows.append(
            {
                "B": budget,
                "repeat": repeat,
                "model": model_name,
                "patient_macro_f1": macro_f1,
                "patient_accuracy": accuracy,
                "aptamers": ";".join(state["feature_names"][feature_idx]),
            }
        )
    return rows


def empirical_percentile(random_scores: np.ndarray, observed: float) -> float:
    below = np.sum(random_scores < observed)
    equal = np.sum(random_scores == observed)
    return float(100.0 * (below + 0.5 * equal) / len(random_scores))


def run_random_controls(output_dir: Path, repeats: int, jobs: int) -> None:
    top_path = output_dir / "top_panel_performance.csv"
    if top_path.exists():
        top = pd.read_csv(top_path)
        print("Loaded cached Top-B performance", flush=True)
    else:
        print("Evaluating fold-specific Top-B panels...", flush=True)
        top_rows = []
        for budget in RANDOM_BUDGETS:
            fold_panels = [ranking[:budget] for ranking in _STATE["lr_rankings"]]
            for model_name in ["LR", "XGBoost"]:
                macro_f1, accuracy = evaluate_panel(fold_panels, model_name, 42)
                top_rows.append(
                    {
                        "B": budget,
                        "model": model_name,
                        "patient_macro_f1": macro_f1,
                        "patient_accuracy": accuracy,
                    }
                )
        top = pd.DataFrame(top_rows)
        top.to_csv(top_path, index=False)

    tasks = [
        (budget, repeat)
        for budget in RANDOM_BUDGETS
        for repeat in range(repeats)
    ]
    partial_path = output_dir / f"random_panel_scores_n{repeats}.partial.csv"
    completed = set()
    prior_rows = []
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        prior_rows = partial.to_dict("records")
        completed = set(zip(partial["B"], partial["repeat"]))
        tasks = [task for task in tasks if task not in completed]
        print(f"Resuming after {len(completed)} completed random panels", flush=True)
    if jobs == 1:
        nested_rows = []
        for index, task in enumerate(tasks, start=1):
            nested_rows.append(random_panel_task(task))
            if index % 10 == 0 or index == len(tasks):
                pd.DataFrame(prior_rows + [row for group in nested_rows for row in group]).to_csv(
                    partial_path, index=False
                )
                print(f"Random panels: {index}/{len(tasks)} tasks", flush=True)
    else:
        with mp.get_context("fork").Pool(processes=jobs) as pool:
            nested_rows = []
            for index, result in enumerate(pool.imap_unordered(random_panel_task, tasks, chunksize=1), start=1):
                nested_rows.append(result)
                if index % 10 == 0 or index == len(tasks):
                    pd.DataFrame(prior_rows + [row for group in nested_rows for row in group]).to_csv(
                        partial_path, index=False
                    )
                    print(f"Random panels: {index}/{len(tasks)} tasks", flush=True)
    random_df = pd.DataFrame(prior_rows + [row for group in nested_rows for row in group])
    random_df = random_df.sort_values(["B", "model", "repeat"]).reset_index(drop=True)
    random_df.to_csv(output_dir / f"random_panel_scores_n{repeats}.csv", index=False)

    summary_rows = []
    for (budget, model_name), group in random_df.groupby(["B", "model"], observed=True):
        scores = group["patient_macro_f1"].to_numpy()
        top_score = float(
            top.loc[(top["B"] == budget) & (top["model"] == model_name), "patient_macro_f1"].iloc[0]
        )
        summary_rows.append(
            {
                "B": budget,
                "model": model_name,
                "n_random": len(scores),
                "top_B_patient_macro_f1": top_score,
                "random_mean": float(np.mean(scores)),
                "random_empirical_2.5pct": float(np.quantile(scores, 0.025)),
                "random_empirical_97.5pct": float(np.quantile(scores, 0.975)),
                "top_B_random_percentile_midrank": empirical_percentile(scores, top_score),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / f"random_panel_summary_n{repeats}.csv", index=False)


def run_cellcount_controls(output_dir: Path, n_seeds: int) -> None:
    print("Running cell-count sensitivity...", flush=True)
    rows = []
    for budget in CELLCOUNT_BUDGETS:
        prediction_cache = {"LR": {}}
        for fold in range(N_FOLDS):
            features = _STATE["lr_rankings"][fold][:budget]
            test_idx = _STATE["test_idx"][fold]
            for model_name in prediction_cache:
                prediction_cache[model_name][fold] = fit_predict(
                    _STATE["x_norm"][fold],
                    _STATE["y"],
                    _STATE["train_idx"][fold],
                    test_idx,
                    features,
                    model_name,
                    42,
                )

        for seed in range(n_seeds):
            for target_count in CELLCOUNTS:
                for model_name in prediction_cache:
                    pooled = {}
                    rng = np.random.default_rng(500_000 + budget * 10_000 + seed * 100 + (target_count or 0))
                    for fold in range(N_FOLDS):
                        test_idx = _STATE["test_idx"][fold]
                        test_sids = _STATE["sample_ids"][test_idx]
                        test_pred = prediction_cache[model_name][fold]
                        for sample_id in np.unique(test_sids):
                            local = np.flatnonzero(test_sids == sample_id)
                            if target_count is not None and len(local) > target_count:
                                local = rng.choice(local, size=target_count, replace=False)
                            pooled[sample_id] = int(pd.Series(test_pred[local]).mode().iloc[0])
                    macro_f1, accuracy = patient_metrics(pooled, _STATE["patient_truth"])
                    rows.append(
                        {
                            "B": budget,
                            "n_cells": "all" if target_count is None else str(target_count),
                            "seed": seed,
                            "model": model_name,
                            "patient_macro_f1": macro_f1,
                            "patient_accuracy": accuracy,
                        }
                    )
        print(f"Cell-count sensitivity: B={budget} complete", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / f"cellcount_sensitivity_{n_seeds}seeds.csv", index=False)
    summary = (
        raw.groupby(["B", "n_cells", "model"], observed=True)["patient_macro_f1"]
        .agg(
            n_seeds="size",
            mean="mean",
            empirical_2_5pct=lambda s: s.quantile(0.025),
            empirical_97_5pct=lambda s: s.quantile(0.975),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    summary.to_csv(output_dir / f"cellcount_sensitivity_summary_{n_seeds}seeds.csv", index=False)


def shap_ranking(fold: int, cells_per_patient: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx = _STATE["train_idx"][fold]
    x_norm = _STATE["x_norm"][fold]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=8,
        random_state=42 + fold,
        eval_metric="mlogloss",
        tree_method="hist",
    )
    model.fit(x_norm[train_idx], _STATE["y"][train_idx])

    rng = np.random.default_rng(700_000 + fold)
    selected = []
    train_sids = _STATE["sample_ids"][train_idx]
    for sample_id in np.unique(train_sids):
        patient_idx = train_idx[train_sids == sample_id]
        if len(patient_idx) > cells_per_patient:
            patient_idx = rng.choice(patient_idx, size=cells_per_patient, replace=False)
        selected.append(patient_idx)
    selected_idx = np.concatenate(selected)
    contributions = model.get_booster().predict(
        xgb.DMatrix(x_norm[selected_idx]), pred_contribs=True, strict_shape=True
    )
    contributions = np.asarray(contributions)[..., :-1]
    cell_importance = np.abs(contributions).mean(axis=1)
    patient_importance = []
    selected_sids = _STATE["sample_ids"][selected_idx]
    for sample_id in np.unique(selected_sids):
        patient_importance.append(cell_importance[selected_sids == sample_id].mean(axis=0))
    importance = np.asarray(patient_importance).mean(axis=0)
    return np.argsort(-importance), importance


def run_shap_controls(output_dir: Path, cells_per_patient: int) -> None:
    print("Computing patient-balanced XGBoost TreeSHAP rankings...", flush=True)
    shap_rankings = []
    importance_rows = []
    overlap_rows = []
    for fold in range(N_FOLDS):
        ranking, importance = shap_ranking(fold, cells_per_patient)
        shap_rankings.append(ranking)
        print(f"TreeSHAP ranking: fold {fold + 1}/{N_FOLDS}", flush=True)
        for feature_idx, value in enumerate(importance):
            importance_rows.append(
                {
                    "outer_fold": fold,
                    "aptamer": _STATE["feature_names"][feature_idx],
                    "patient_balanced_mean_abs_shap": float(value),
                    "rank": int(np.flatnonzero(ranking == feature_idx)[0] + 1),
                }
            )
        for budget in [10, 20]:
            lr_set = set(_STATE["lr_rankings"][fold][:budget])
            shap_set = set(ranking[:budget])
            overlap_rows.append(
                {
                    "outer_fold": fold,
                    "B": budget,
                    "intersection": len(lr_set & shap_set),
                    "union": len(lr_set | shap_set),
                    "jaccard": len(lr_set & shap_set) / len(lr_set | shap_set),
                    "lr_aptamers": ";".join(_STATE["feature_names"][sorted(lr_set)]),
                    "shap_aptamers": ";".join(_STATE["feature_names"][ranking[:budget]]),
                }
            )
    pd.DataFrame(importance_rows).to_csv(output_dir / "xgboost_shap_rankings.csv", index=False)
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(output_dir / "lr_shap_jaccard_by_fold.csv", index=False)
    overlap.groupby("B", observed=True)["jaccard"].agg(["mean", "std", "min", "max"]).reset_index().to_csv(
        output_dir / "lr_shap_jaccard_summary.csv", index=False
    )

    performance_rows = []
    for ranking_name, rankings in [("LR coefficient", _STATE["lr_rankings"]), ("XGBoost-SHAP", shap_rankings)]:
        for budget in [10, 20]:
            panels = [ranking[:budget] for ranking in rankings]
            for model_name in ["LR", "XGBoost"]:
                macro_f1, accuracy = evaluate_panel(panels, model_name, 42)
                performance_rows.append(
                    {
                        "ranking": ranking_name,
                        "B": budget,
                        "classifier": model_name,
                        "patient_macro_f1": macro_f1,
                        "patient_accuracy": accuracy,
                    }
                )
    pd.DataFrame(performance_rows).to_csv(output_dir / "lr_vs_shap_panel_performance.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.smoke:
        RANDOM_BUDGETS[:] = [5]
        CELLCOUNT_BUDGETS[:] = [5]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, x, feature_names = load_merged_dataframe(
        args.data_dir,
        f"{args.data_dir}/metadata.csv",
        f"{args.data_dir}/cell_annotation.csv",
        subtype_col_in_annotation="Celltypes_new",
        drop_missing_subtype=True,
        drop_unknown=True,
        coarse_mapping_config="cell_JEPA/apt_jepa/configs/coarse_lineage_explicit.yaml",
    )
    disease_encoder = LabelEncoder().fit(merged["disease"].astype(str))
    y = disease_encoder.transform(merged["disease"].astype(str))
    sample_ids = merged["sample_id"].astype(str).to_numpy()
    patient_disease = merged[["sample_id", "disease"]].drop_duplicates().set_index("sample_id")["disease"]
    patient_truth = {
        patient: int(disease_encoder.transform([disease])[0]) for patient, disease in patient_disease.items()
    }

    train_indices, test_indices, x_norm_folds = [], [], []
    for fold in range(N_FOLDS):
        split = pd.read_csv(Path(args.split_dir) / f"fold{fold}_split.csv")
        membership = merged[["sample_id"]].merge(split[["sample_id", "split"]], on="sample_id", how="left")
        if membership["split"].isna().any():
            raise ValueError(f"Missing split membership in fold {fold}")
        train_idx = np.flatnonzero(membership["split"].isin(["train", "val"]).to_numpy())
        test_idx = np.flatnonzero((membership["split"] == "test").to_numpy())
        mean, std = fit_standardizer(x[train_idx])
        x_norm = apply_standardizer(x, mean, std).astype(np.float32)
        train_indices.append(train_idx)
        test_indices.append(test_idx)
        x_norm_folds.append(x_norm)
        print(f"Standardized fold {fold + 1}/{N_FOLDS}", flush=True)

    _STATE.update({"x_norm": x_norm_folds, "y": y, "train_idx": train_indices})
    ranking_cache = output_dir / "lr_rankings.npz"
    if ranking_cache.exists():
        cached = np.load(ranking_cache)
        rankings = [cached[f"fold{fold}"] for fold in range(N_FOLDS)]
        print("Loaded cached fold-specific LR rankings", flush=True)
    elif args.jobs == 1:
        rankings = [rank_fold_task(fold) for fold in range(N_FOLDS)]
    else:
        with mp.get_context("fork").Pool(processes=min(N_FOLDS, args.jobs)) as pool:
            rankings = pool.map(rank_fold_task, range(N_FOLDS))
    if not ranking_cache.exists():
        np.savez(ranking_cache, **{f"fold{fold}": ranking for fold, ranking in enumerate(rankings)})
        print("Computed and cached five fold-specific LR rankings", flush=True)

    _STATE.update(
        {
            "x_norm": x_norm_folds,
            "y": y,
            "sample_ids": sample_ids,
            "patient_truth": patient_truth,
            "train_idx": train_indices,
            "test_idx": test_indices,
            "lr_rankings": rankings,
            "feature_names": np.asarray(feature_names),
            "n_features": len(feature_names),
        }
    )
    with open(output_dir / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)

    sections = {section.strip() for section in args.sections.split(",") if section.strip()}
    if "random" in sections:
        print(f"Running {args.random_repeats} random panels per budget...", flush=True)
        run_random_controls(output_dir, args.random_repeats, args.jobs)
    if "cellcount" in sections:
        run_cellcount_controls(output_dir, args.cell_seeds)
    if "shap" in sections:
        run_shap_controls(output_dir, args.shap_cells_per_patient)


if __name__ == "__main__":
    main()
