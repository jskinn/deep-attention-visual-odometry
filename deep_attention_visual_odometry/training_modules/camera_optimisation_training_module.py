from typing import Literal
import torch
import torch.nn as nn
from lightning import LightningModule
from deep_attention_visual_odometry.base_types import CameraViewsAndPoints


class CameraOptmisationTrainingModule(LightningModule):
    def __init__(
        self, network: nn.Module, matmul_precision: Literal["medium", "high", "highest"] = "high"
    ):
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
        error = predictions.get_error().mean()
        focal_length_loss = self.loss_fn(
            predictions.focal_length[:, 0], batch.camera_intrinsics[:, 0]
        )
        cx_loss = self.loss_fn(predictions.cx[:, 0], batch.camera_intrinsics[:, 1])
        cy_loss = self.loss_fn(predictions.cy[:, 0], batch.camera_intrinsics[:, 2])
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
