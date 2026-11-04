"""The three frozen V0.2 losses and diagnostic reductions."""

from __future__ import annotations

from typing import Any

import torch

from .simulator import Realization


def per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().flatten(1).mean(1)


def subset_realization(realization: Realization, mask: torch.Tensor) -> Realization:
    return Realization(realization.epsilon_zernike[mask], realization.epsilon_tilt[mask])


def clean_loss_per_sample(prediction: Any, batch: dict[str, Any]) -> torch.Tensor:
    return per_sample_mse(prediction.clean, batch["clean"])


def trajectory_loss_per_sample(
    prediction: Any,
    batch: dict[str, Any],
    simulator: torch.nn.Module,
) -> torch.Tensor:
    rendered = simulator.render(batch["clean"].float(), prediction.realization, batch["s_T"]).output
    return per_sample_mse(rendered, batch["target_T"])


def bridge_loss_per_sample(
    prediction: Any,
    batch: dict[str, Any],
    simulator: torch.nn.Module,
) -> torch.Tensor:
    mask = batch["valid_h"]
    if not bool(mask.any().detach().cpu()):
        return torch.empty(0, device=batch["image"].device, dtype=torch.float32)
    # Render the whole fixed-size microbatch, then reduce only over valid
    # transitions.  Invalid rows carry h=g and target_h=image in stack_loaded;
    # their values are discarded here.  This is algebraically identical to
    # rendering the indexed subset, but avoids recompiling the Ascend graph for
    # every observed valid-subset cardinality.
    rendered = simulator.render(
        prediction.clean.float(),
        prediction.realization,
        batch["h"],
    ).output
    return per_sample_mse(rendered, batch["target_h"])[mask]


def trajectory_bridge_losses_per_sample(
    prediction: Any,
    batch: dict[str, Any],
    simulator: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the frozen L_T estimator and valid L_B rows in one render call."""

    count = batch["image"].shape[0]
    zernike = torch.cat(
        [
            prediction.realization.epsilon_zernike,
            prediction.realization.epsilon_zernike,
        ],
        dim=0,
    )
    tilt = torch.cat(
        [
            prediction.realization.epsilon_tilt,
            prediction.realization.epsilon_tilt,
        ],
        dim=0,
    )
    rendered = simulator.render(
        torch.cat([batch["clean"].float(), prediction.clean.float()], dim=0),
        Realization(zernike, tilt),
        torch.cat([batch["s_T"], batch["h"]], dim=0),
    ).output
    values = per_sample_mse(
        rendered,
        torch.cat([batch["target_T"], batch["target_h"]], dim=0),
    )
    return values[:count], values[count:][batch["valid_h"]]


def output_zero(prediction: Any) -> torch.Tensor:
    """A zero graph that marks every output branch as participating in DDP."""

    return (
        prediction.clean.float().sum()
        + prediction.realization.epsilon_zernike.float().sum()
        + prediction.realization.epsilon_tilt.float().sum()
    ) * 0.0
