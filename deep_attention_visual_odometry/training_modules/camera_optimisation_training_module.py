from functools import partial
import torch
import torch.nn as nn
from lightning import LightningModule
from deep_attention_visual_odometry.base_types import CameraViewsAndPoints


class CameraOptmisationTrainingModule(LightningModule):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
        self.loss_fn = nn.L1Loss()

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
        predictions = self.network(batch.projected_points)
        predicted_parameters =

        # Take an l1 loss between the predicted and actual parameters
        loss = self.loss_fn(predicted_parameters, batch.parameters)
        self.log(f"{step_name} loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-4)
        return optimizer

    def _weights_log(self, name: str, weights: torch.Tensor, step: str) -> None:
        self.logger.experiment.add_histogram(step + " " + name, weights)
