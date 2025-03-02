import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from .wildtrack_dataset import Wildtrack
from .multiviewx_dataset import MultiviewX
from .pedestrian_dataset import PedestrianDataset
from .sampler import TemporalSampler


class PedestrianDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = "./datasets/MultiviewX",
            batch_size: int = 2,
            num_workers: int = 4,
            resolution: int = None,
                bounds: int = None,
    accumulate_grad_batches: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        dataset_config = dict(base = base, 
                        resolution = self.resolution, 
                            bounds = self.bounds)
        if stage == 'fit':
            self.data_train = PedestrianDataset(**dataset_config, is_train=True)

        if stage == 'fit' or stage == 'validate':
            self.data_val = PedestrianDataset(**dataset_config, is_train=False)

        if stage == 'test':
            self.data_test = PedestrianDataset(**dataset_config, is_train=False)

        if stage == 'predict':
            self.data_predict = PedestrianDataset(**dataset_config, is_train=False)

    def sampler(self, data):
        return TemporalSampler(data, 
                    batch_size = self.batch_size,
       accumulate_grad_batches = self.accumulate_grad_batches,
        )

    def train_dataloader(self):
        return DataLoader(
                          self.data_train,
             batch_size = self.batch_size,
            num_workers = self.num_workers,
                sampler = self.sampler(self.data_train),
             pin_memory = True,
        )

    def val_dataloader(self):
        return DataLoader(
                          self.data_val,
             batch_size = self.batch_size,
            num_workers = self.num_workers,
                sampler = self.sampler(self.data_val),
             pin_memory = True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size = 1,
            num_workers = 1,
        )

    def predict_dataloader(self):
        return DataLoader(
                        self.data_predict,
             batch_size=self.batch_size,
            num_workers=self.num_workers
        )
