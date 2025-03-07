"""
Reference:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html
"""
import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from .sampler import TemporalSampler
from .wildtrack import Wildtrack
from .multiviewx import MultiviewX
from .pedestrian import PedestrianDataset
from .synthehicle import SynthehicleDataset


class PedestrianDataLoader(pl.LightningDataModule):

    def __init__(
                self,
              data_dir: str,
             data_type: str,
           num_workers: int = 4,
            batch_size: int = 2,
      batch_grad_accum: int = 8,
            resolution = None,
                bounds = None,
    ):
        super().__init__()

        self.data_type = data_type
        self.data_dir = data_dir
        self.data_root = os.path.basename(self.data_dir)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_grad_accum = batch_grad_accum

        self.resolution = resolution
        self.bounds = bounds

        self.data_test = None
        self.data_val = None
        self.data_train = None
        self.data_predict = None

    def setup(self, stage: Optional[str] = None):

        if self.data_type == 'wildtrack':
            dataclass = Wildtrack
        elif self.data_type == 'multiviewx':
            dataset_base = MultiviewX
        else:
            raise ValueError(f'{self.data_type} is not supported! Choose `wildtrack` or `multiviewx`!')

        dataset_base = dataclass(self.data_dir)
        dataset_config = dict(base = dataset_base, 
                        resolution = self.resolution, 
                            bounds = self.bounds)
        if stage == 'fit':
            self.data_train = PedestrianDataset(**dataset_config, is_train=True)

        if stage == 'fit' \
        or stage == 'validate':
            self.data_val = PedestrianDataset(**dataset_config, is_train=False)

        if stage == 'test':
            self.data_test = PedestrianDataset(**dataset_config, is_train=False)

        if stage == 'predict':
            self.data_predict = PedestrianDataset(**dataset_config, is_train=False)

    def sampler(self, data):
        return TemporalSampler(data, batch_size = self.batch_size,
                                    batch_grad_accum = self.batch_grad_accum)

    def train_dataloader(self):
        return DataLoader(self.data_train,
             batch_size = self.batch_size,
            num_workers = self.num_workers,
                sampler = self.sampler(self.data_train),
             pin_memory = True)

    def val_dataloader(self):
        return DataLoader(self.data_val,
             batch_size = self.batch_size,
            num_workers = self.num_workers,
                sampler = self.sampler(self.data_val),
             pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size = 1,
                         num_workers = 1)

    def predict_dataloader(self):
        return DataLoader(self.data_predict,
                batch_size=self.batch_size,
                num_workers=self.num_workers)


class VehicleDataLoader(pl.LightningDataModule):

    def __init__(
                self,
              data_dir: str,
            test_split: str = 'test',
           num_workers: int = 8,
            batch_size: int = 6,
      batch_grad_accum: int = 8,
            resolution = None,
                bounds = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.test_split = test_split

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_grad_accum = batch_grad_accum

        self.resolution = resolution
        self.bounds = bounds

        self.data_test = None
        self.data_val = None
        self.data_train = None
        self.data_predict = None

    def setup(self, stage: Optional[str] = None):

        dataset_config = dict(root = self.data_dir, 
                        resolution = self.resolution, 
                            bounds = self.bounds)

        if stage == 'fit':
            self.data_train = SynthehicleDataset(**dataset_config, split='train')

        if stage == 'fit' \
        or stage == 'validate':
            self.data_val = SynthehicleDataset(**dataset_config, split='val', is_train=True)

        if stage == 'test':
            self.data_test = SynthehicleDataset(**dataset_config, split=self.test_split)

    def sampler(self, data):
        return TemporalSampler(data, batch_size = self.batch_size,
                                    batch_grad_accum = self.batch_grad_accum)

    def train_dataloader(self):
        return DataLoader(
                            self.data_train,
                 batch_size=self.batch_size,
                num_workers=self.num_workers,
                    sampler=self.sampler(self.data_train),
         persistent_workers=self.num_workers > 0,
                 pin_memory=True,
            )

    def val_dataloader(self):
        return DataLoader(
                            self.data_val,
                 batch_size=self.batch_size,
                num_workers=self.num_workers,
                    sampler=self.sampler(self.data_val),
         persistent_workers=self.num_workers > 0,
                 pin_memory=True,
            )

    def test_dataloader(self):
        return DataLoader(
                            self.data_test,
                num_workers=self.num_workers,
                batch_size=1,
            )

 