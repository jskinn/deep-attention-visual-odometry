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
        training_length: int = 1000,
        validation_length: int = 100,
        test_length: int = 100,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.num_points = int(num_points)
        self.num_views = int(num_views)
        self.training_length = int(training_length)
        self.validation_length = int(validation_length)
        self.test_length = int(test_length)

    def train_dataloader(self):
        return DataLoader(
            CameraAndParametersDataset(
                num_points=self.num_points,
                num_views=self.num_views,
                epoch_length=self.training_length,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            CameraAndParametersDataset(
                num_points=self.num_points,
                num_views=self.num_views,
                epoch_length=self.validation_length
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            CameraAndParametersDataset(
                num_points=self.num_points,
                num_views=self.num_views,
                epoch_length=self.test_length
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
