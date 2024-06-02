from typing import Literal
import torch
import torch.nn as nn
from lightning import LightningModule
from deep_attention_visual_odometry.base_types import CameraViewsAndPoints


class CameraOptmisationTrainingModule(LightningModule):
    def __init__(
        self, network: nn.Module, matmul_precision: Literal["medium", "high", "highest"] = "high",
            float_precision: Literal["32", "64"] = "32"
    ):
        super().__init__()
        self.network = network
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.float_precision = float_precision
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
        projected_points = batch.projected_points
        visibility_mask = batch.visibility_mask
        camera_intrinsics = batch.camera_intrinsics
        if self.float_precision == "64":
            projected_points = projected_points.to(torch.float64)
            visibility_mask = visibility_mask.to(torch.float64)
            camera_intrinsics = camera_intrinsics.to(torch.float64)

        predictions = self.network(projected_points, visibility_mask)
        error = predictions.get_error().mean()
        focal_length_loss = self.loss_fn(
            predictions.focal_length[:, 0], camera_intrinsics[:, 0]
        )
        cx_loss = self.loss_fn(predictions.cx[:, 0], camera_intrinsics[:, 1])
        cy_loss = self.loss_fn(predictions.cy[:, 0], camera_intrinsics[:, 2])
        self.log(f"{step_name} mean error ", error)
        self.log(f"{step_name} focal length loss", focal_length_loss)
        self.log(f"{step_name} cx loss", cx_loss)
        self.log(f"{step_name} cy loss", cy_loss)

        loss = focal_length_loss + cx_loss + cy_loss + error
        self.log(f"{step_name} loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-4)
        return optimizer

    def _weights_log(self, name: str, weights: torch.Tensor, step: str) -> None:
        self.logger.experiment.add_histogram(step + " " + name, weights)
