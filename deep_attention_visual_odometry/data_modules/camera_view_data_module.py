from lightning import LightningDataModule
from torch.utils.data import DataLoader
from deep_attention_visual_odometry.data import CameraAndParametersDataset


class CameraViewDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        num_points: int = 128,
        num_views: int = 4,
        training_batches: int = 128,
        validation_batches: int = 16,
        test_batches: int = 16,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.num_points = int(num_points)
        self.num_views = int(num_views)
        self.training_batches = int(training_batches)
        self.validation_batches = int(validation_batches)
        self.test_batches = int(test_batches)

    def train_dataloader(self):
        return DataLoader(
            CameraAndParametersDataset(
                num_points=self.num_points,
                num_views=self.num_views,
                epoch_length=self.training_batches * self.batch_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            CameraAndParametersDataset(
                num_points=self.num_points,
                num_views=self.num_views,
                epoch_length=self.validation_batches * self.batch_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            CameraAndParametersDataset(
                num_points=self.num_points,
                num_views=self.num_views,
                epoch_length=self.test_batches * self.batch_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
