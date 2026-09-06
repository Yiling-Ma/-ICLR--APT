"""Train matched flat-CE and two-level HCE FT-Transformer baselines.

Both variants use the exact APT Transformer tokenizer/backbone dimensions and
training budget of DropCascade. The HCE variant follows the reachability-matrix
probability aggregation of Cultrera di Montesano et al.; because APT-Bench
provides a fine and a coarse label for every cell, its objective includes both
observed levels of the two-level tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from apt_jepa.data.dataset import build_dataloaders, load_merged_dataframe
from apt_jepa.losses.hierarchical_cross_entropy import (
    build_two_level_reachability,
    hierarchical_nll,
    hierarchical_scores,
)
from apt_jepa.models.apt_transformer import APTTransformerBackbone
from apt_jepa.models.heads import make_mlp_head
from apt_jepa.scripts.train_soft_lineage_cascade import (
    apply_explicit_lineage_mapping,
    build_subtype_to_lineage,
)
from apt_jepa.train.utils import apply_overrides, load_yaml, set_seed


class MatchedFTTransformer(nn.Module):
    """One flat node classifier over coarse and fine labels."""

    def __init__(self, num_coarse: int, num_fine: int, model_cfg: dict):
        super().__init__()
        hidden_dim = model_cfg.get("hidden_dim", 256)
        dropout = model_cfg.get("dropout", 0.1)
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.encoder = APTTransformerBackbone(
            num_features=model_cfg.get("num_features", 293),
            hidden_dim=hidden_dim,
            n_layers=model_cfg.get("n_layers", 4),
            n_heads=model_cfg.get("n_heads", 8),
            ff_dim=model_cfg.get("ff_dim", 512),
            dropout=dropout,
            use_mask_embedding=True,
        )
        self.node_head = make_mlp_head(
            hidden_dim,
            num_coarse + num_fine,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.encoder(x, mask=None)
        return self.node_head(embedding)

    def split_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return logits[:, : self.num_coarse], logits[:, self.num_coarse :]


def compute_loss(
    model: MatchedFTTransformer,
    logits: torch.Tensor,
    y_coarse: torch.Tensor,
    y_fine: torch.Tensor,
    reachability: torch.Tensor,
    loss_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_logits, fine_logits = model.split_logits(logits)
    variant = loss_cfg.get("variant", "flat_ce")
    if variant == "flat_ce":
        fine_loss = F.cross_entropy(fine_logits, y_fine)
        coarse_loss = F.cross_entropy(coarse_logits, y_coarse)
    elif variant == "hce":
        epsilon = loss_cfg.get("epsilon", 1e-6)
        fine_targets = model.num_coarse + y_fine
        fine_loss = hierarchical_nll(
            logits, fine_targets, reachability, epsilon=epsilon
        )
        coarse_loss = hierarchical_nll(
            logits, y_coarse, reachability, epsilon=epsilon
        )
    else:
        raise ValueError(f"Unknown loss.variant={variant}")
    total = (
        loss_cfg.get("lambda_fine", 1.0) * fine_loss
        + loss_cfg.get("lambda_coarse", 0.5) * coarse_loss
    )
    return total, fine_loss, coarse_loss


def decode(
    model: MatchedFTTransformer,
    logits: torch.Tensor,
    reachability: torch.Tensor,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_logits, fine_logits = model.split_logits(logits)
    fine_pred = fine_logits.argmax(dim=1)
    if variant == "hce":
        coarse_pred = hierarchical_scores(logits, reachability)[
            :, : model.num_coarse
        ].argmax(dim=1)
    else:
        coarse_pred = coarse_logits.argmax(dim=1)
    return coarse_pred, fine_pred


def run_epoch(
    model,
    loader,
    device,
    reachability,
    loss_cfg,
    optimizer=None,
    scaler=None,
    grad_clip_norm=1.0,
):
    training = optimizer is not None
    model.train(training)
    variant = loss_cfg.get("variant", "flat_ce")
    losses = []
    store = {key: [] for key in ("fine_true", "fine_pred", "coarse_true", "coarse_pred")}
    metadata = {"sample_id": [], "disease": [], "subtype": []}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y_fine = batch["subtype_label"].to(device, non_blocking=True)
            y_coarse = batch["coarse_label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp = scaler is not None and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss, fine_loss, coarse_loss = compute_loss(
                    model, logits, y_coarse, y_fine, reachability, loss_cfg
                )
            if training:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
            coarse_pred, fine_pred = decode(model, logits, reachability, variant)
            losses.append((loss.item(), fine_loss.item(), coarse_loss.item()))
            store["fine_true"].append(y_fine.detach().cpu().numpy())
            store["fine_pred"].append(fine_pred.detach().cpu().numpy())
            store["coarse_true"].append(y_coarse.detach().cpu().numpy())
            store["coarse_pred"].append(coarse_pred.detach().cpu().numpy())
            metadata["sample_id"].extend(batch["sample_id"])
            metadata["disease"].extend(batch["disease_name"])
            metadata["subtype"].extend(batch["subtype_name"])
    result = {key: np.concatenate(values) for key, values in store.items()}
    result.update(metadata)
    mean_losses = np.asarray(losses).mean(axis=0)
    result.update(
        {
            "loss": float(mean_losses[0]),
            "fine_loss": float(mean_losses[1]),
            "coarse_loss": float(mean_losses[2]),
            "fine_macro_f1": float(
                f1_score(result["fine_true"], result["fine_pred"], average="macro", zero_division=0)
            ),
            "coarse_macro_f1": float(
                f1_score(result["coarse_true"], result["coarse_pred"], average="macro", zero_division=0)
            ),
        }
    )
    return result


def save_checkpoint(path: Path, model, optimizer, epoch: int, score: float, cfg: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
            "config": cfg,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cell_JEPA/apt_jepa/configs/matched_ft_transformer.yaml")
    parser.add_argument("--variant", choices=("flat_ce", "hce"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument(
        "--reuse-best",
        action="store_true",
        help="Skip encoder training and rerun validation-selected head calibration.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def resolve(path: str, root: Path) -> str:
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else root / candidate)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = apply_overrides(load_yaml(resolve(args.config, project_root)), args.override)
    if args.variant:
        cfg["loss"]["variant"] = args.variant
    if args.fold is not None:
        cfg["split"]["external_split_csv"] = (
            f"cell_JEPA/outputs/dropcascade_kfold5_splits/fold{args.fold}_split.csv"
        )
        cfg["train"]["output_dir"] = (
            f"cell_JEPA/outputs/matched_ft_transformer/{cfg['loss']['variant']}/fold{args.fold}"
        )
    set_seed(cfg["train"].get("seed", 42))
    output_dir = Path(resolve(cfg["train"]["output_dir"], project_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    data_cfg = cfg["data"]
    merged_df, x, _ = load_merged_dataframe(
        resolve(data_cfg["data_dir"], project_root),
        resolve(data_cfg["metadata_path"], project_root),
        resolve(data_cfg["annotation_path"], project_root),
        subtype_col_in_annotation=data_cfg.get("subtype_col_in_annotation"),
        drop_missing_subtype=data_cfg.get("drop_missing_subtype", True),
        drop_unknown=data_cfg.get("drop_unknown", True),
        coarse_mapping_config=None,
    )
    merged_df, explicit_mapping = apply_explicit_lineage_mapping(
        merged_df, resolve(data_cfg["coarse_mapping_config"], project_root)
    )
    split_df = pd.read_csv(resolve(cfg["split"]["external_split_csv"], project_root))
    split_df.to_csv(output_dir / "split.csv", index=False)
    bundle = build_dataloaders(
        merged_df,
        x,
        split_df,
        batch_size=cfg["train"].get("batch_size", 512),
        num_workers=cfg["train"].get("num_workers", 4),
        use_quantile_binning=data_cfg.get("use_quantile_binning", False),
        split_mode="sample",
    )
    subtype_to_lineage = build_subtype_to_lineage(bundle)
    num_coarse = len(bundle.coarse_encoder.classes_)
    num_fine = len(bundle.subtype_encoder.classes_)
    reachability = build_two_level_reachability(
        subtype_to_lineage, num_coarse, num_fine
    )
    label_mapping = {
        "subtypes": bundle.subtype_encoder.classes_.tolist(),
        "lineages": bundle.coarse_encoder.classes_.tolist(),
        "diseases": bundle.disease_encoder.classes_.tolist(),
        "explicit_mapping": explicit_mapping,
    }
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reachability = reachability.to(device)
    model = MatchedFTTransformer(num_coarse, num_fine, cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-5),
    )
    amp = cfg["train"].get("use_amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if args.reuse_best:
        if not (output_dir / "best.pt").exists():
            raise FileNotFoundError(f"Missing encoder checkpoint: {output_dir / 'best.pt'}")
        best_score = float(torch.load(output_dir / "best.pt", map_location="cpu")["score"])
    else:
        history = []
        best_score = -1.0
        patience = 0
        for epoch in range(1, cfg["train"].get("epochs", 50) + 1):
            train_result = run_epoch(
                model,
                bundle.train_loader,
                device,
                reachability,
                cfg["loss"],
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=cfg["train"].get("grad_clip_norm", 1.0),
            )
            val_result = run_epoch(model, bundle.val_loader, device, reachability, cfg["loss"])
            row = {
                "epoch": epoch,
                **{f"train_{k}": train_result[k] for k in ("loss", "fine_loss", "coarse_loss", "fine_macro_f1", "coarse_macro_f1")},
                **{f"val_{k}": val_result[k] for k in ("loss", "fine_loss", "coarse_loss", "fine_macro_f1", "coarse_macro_f1")},
            }
            history.append(row)
            print(json.dumps(row))
            score = val_result["fine_macro_f1"]
            if score > best_score:
                best_score = score
                patience = 0
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, score, cfg)
            else:
                patience += 1
            pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)
            if patience >= cfg["train"].get("patience", 10):
                break

    best = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    calibration_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg["schedule"].get("calibration_lr", 3e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-5),
    )
    calibration_score = best_score
    selected_calibration_epoch = 0
    calibrated_path = output_dir / "best_calibrated.pt"
    for calibration_epoch in range(1, cfg["schedule"].get("calibration_epochs", 5) + 1):
        run_epoch(
            model,
            bundle.train_loader,
            device,
            reachability,
            cfg["loss"],
            optimizer=calibration_optimizer,
            scaler=scaler,
            grad_clip_norm=cfg["train"].get("grad_clip_norm", 1.0),
        )
        calibration_val = run_epoch(
            model, bundle.val_loader, device, reachability, cfg["loss"]
        )
        score = calibration_val["fine_macro_f1"]
        print(
            json.dumps(
                {
                    "calibration_epoch": calibration_epoch,
                    "val_fine_macro_f1": score,
                    "val_coarse_macro_f1": calibration_val["coarse_macro_f1"],
                }
            )
        )
        if score > calibration_score:
            calibration_score = score
            selected_calibration_epoch = calibration_epoch
            save_checkpoint(
                calibrated_path,
                model,
                calibration_optimizer,
                best["epoch"],
                score,
                cfg,
            )

    if selected_calibration_epoch:
        model.load_state_dict(torch.load(calibrated_path, map_location=device)["model"])
    else:
        model.load_state_dict(best["model"])

    test = run_epoch(model, bundle.test_loader, device, reachability, cfg["loss"])
    metrics = {
        "variant": cfg["loss"]["variant"],
        "n_cells": int(len(test["fine_true"])),
        "n_patients": int(pd.Series(test["sample_id"]).nunique()),
        "fine_macro_f1": test["fine_macro_f1"],
        "fine_accuracy": float(accuracy_score(test["fine_true"], test["fine_pred"])),
        "coarse_macro_f1": test["coarse_macro_f1"],
        "coarse_accuracy": float(accuracy_score(test["coarse_true"], test["coarse_pred"])),
        "best_epoch": int(best["epoch"]),
        "best_val_fine_macro_f1": float(best_score),
        "selected_calibration_epoch": selected_calibration_epoch,
        "selected_val_fine_macro_f1": float(calibration_score),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(
        {
            "sample_id": test["sample_id"],
            "disease": test["disease"],
            "fine_true_id": test["fine_true"],
            "fine_pred_id": test["fine_pred"],
            "coarse_true_id": test["coarse_true"],
            "coarse_pred_id": test["coarse_pred"],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    save_checkpoint(
        output_dir / "final.pt",
        model,
        calibration_optimizer,
        best["epoch"],
        metrics["fine_macro_f1"],
        cfg,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
