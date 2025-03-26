# Copyright (C) 2024  John Skinner
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
import mlflow.pytorch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import ModelSummary

from deep_attention_visual_odometry.data_modules import CameraViewDataModule
from deep_attention_visual_odometry.training_modules import (
    CameraCalibrationTrainingModule,
)
from deep_attention_visual_odometry.networks import CalibrationNetwork


def main():
    experiment_description = """
    Learn a mapping from 2-D pixel matches across multiple views to the camera intrinsics and extrinsics.
    The pixel matches come from an oracle, they are assumed to be correct.
    """
    num_views = 4
    num_points = 8
    hidden_size = 8 * num_views * num_points
    network = CalibrationNetwork(
        num_views=num_views,
        num_points=num_points,
        hidden_size=hidden_size,
    )
    data_module = CameraViewDataModule(
        num_points=num_points,
        num_views=num_views,
        batch_size=64,
        num_workers=12,
        training_batches=128,
        validation_batches=16,
        test_batches=16,
    )
    training_module = CameraCalibrationTrainingModule(
        matmul_precision="highest",
        float_precision="64",
        network=network,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs/camera_calibration_oracle",
        name="bgfs_solver_mlp_guess",
    )
    mlflow_logger = MLFlowLogger(
        experiment_name="camera_calibration_oracle",
        run_name="bfgs solver mlp guess",
        tracking_uri="http://localhost:8080",
        tags={
            "project": "deep-attention-visual-odm",
            "mlflow.note.content": experiment_description,
        },
    )
    mlflow.pytorch.autolog()
    trainer = Trainer(
        max_epochs=50,
        logger=[tensorboard_logger, mlflow_logger],
        callbacks=[ModelSummary(max_depth=2)]
    )
    trainer.fit(model=training_module, datamodule=data_module)


if __name__ == "__main__":
    main()
