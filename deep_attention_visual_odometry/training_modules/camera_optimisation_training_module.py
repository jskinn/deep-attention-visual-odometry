from typing import Literal
import torch
import torch.nn as nn
from lightning import LightningModule
from deep_attention_visual_odometry.base_types import CameraViewsAndPoints


class CameraOptmisationTrainingModule(LightningModule):
    def __init__(self, network: nn.Module, matmul_precision: Literal["medium", "high"] = "high"):
        super().__init__()
        self.network = network
        self.loss_fn = nn.MSELoss(reduction="mean")
        torch.set_float32_matmul_precision(matmul_precision)

    def training_step(self, batch: CameraViewsAndPoints, batch_idx):
        return self._step(batch, "Training")

    def validation_step(self, batch: CameraViewsAndPoints, batch_idx):
        return self._step(batch, "Validation", log_weights=True)

    def test_step(self, batch: CameraViewsAndPoints, batch_idx):
        return self._step(batch, "Test", log_weights=True)

    def _step(
        self,
        batch: CameraViewsAndPoints,
        step_name: str = "Training",
        log_weights: bool = False,
    ):
        predictions = self.network(batch.projected_points, batch.visibility_mask)
        focal_length_loss = self.loss_fn(
            predictions.focal_length, batch.camera_intrinsics[:, 0:1, 0]
        )
        cx_loss = self.loss_fn(predictions.cx, batch.camera_intrinsics[:, 2:3, 0])
        cy_loss = self.loss_fn(predictions.cy, batch.camera_intrinsics[:, 2:3, 1])
        self.log(f"{step_name} focal length loss", focal_length_loss)
        self.log(f"{step_name} cx loss", cx_loss)
        self.log(f"{step_name} cy loss", cy_loss)

        loss = focal_length_loss + cx_loss + cy_loss
        self.log(f"{step_name} loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-4)
        return optimizer

    def _weights_log(self, name: str, weights: torch.Tensor, step: str) -> None:
        self.logger.experiment.add_histogram(step + " " + name, weights)
