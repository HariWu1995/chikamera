import os.path as osp

import matplotlib.pyplot as plt

import numpy as np
import lightning as pl

import torch
import torch.nn.functional as F

from .loss import FocalLoss, compute_rot_loss
from .models import Segnet, MVDet, Liftnet, Bevformer
from .utils.vox import VoxelUtil
from .utils.misc import pack_seqdim, reduce_masked_mean, sigmoid
from .utils.pproc import postprocess
from .evaluation.mod import mod_metrics
from .evaluation.mot_bev import mot_metrics
from .tracking.multitracker import JDETracker


class WorldTrackModel(pl.LightningModule):

    def __init__(
            self,
            model_name: str = 'segnet',
            encoder_name: str = 'res18',
            learning_rate: float = 0.001,
            resolution: tuple = (200, 4, 200),
                bounds: tuple =(-75, 75, -75, 75, -1, 5),
                depth: tuple = (100, 2.0, 25),
        scene_centroid: tuple = (0.0, 0.0, 0.0),
                z_sign: int = 1,
            feat2d_dim: int = 128,
            num_cameras: int = None,
            num_classes: int = 1,
            max_detections: int = 60,
            conf_threshold: float = 0.5,
            use_temporal_cache: bool = True,
        ):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name

        self.learning_rate = learning_rate
        self.resolution = resolution
        
        self.Y, self.Z, self.X = self.resolution
        self.D, self.DMIN, self.DMAX = depth
        
        self.bounds = bounds
        self.max_detections = max_detections
        self.conf_threshold = conf_threshold
        
        # Loss
        self.center_loss_fn = FocalLoss()

        # Temporal cache
        self.use_temporal_cache = use_temporal_cache
        self.max_cache = 32
        self.temporal_cache_frames = -2 * torch.ones(self.max_cache, dtype=torch.long)
        self.temporal_cache = None

        # Test
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        self.mota_seq_gt_list, self.mota_seq_pred_list = [], []
        self.frame = 0
        self.test_tracker = JDETracker(conf_thres=self.conf_threshold)

        # Model
        num_cameras = None if num_cameras == 0 else num_cameras

        model_config = dict(
            Y=self.Y, Z=self.Z, X=self.X,
            num_cameras = num_cameras,
            num_classes = num_classes,
           encoder_type = self.encoder_name,
                 z_sign = z_sign,
             feat2d_dim = feat2d_dim,
        )

        if model_name == 'segnet':
            self.model = Segnet(**model_config)

        elif model_name == 'liftnet':
            self.model = Liftnet(**model_config, DMIN=self.DMIN, DMAX=self.DMAX, D=self.D)

        elif model_name == 'bevformer':
            del model_config['num_cameras']
            self.model = Bevformer(**model_config)

        elif model_name == 'mvdet':
            del model_config['z_sign']
            self.model = MVDet(**model_config)

        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        self.scene_centroid = torch.tensor(scene_centroid, device=self.device).reshape([1, 3])
        self.vox_util = VoxelUtil(self.Y, self.Z, self.X, scene_centroid=self.scene_centroid, bounds=self.bounds)
        
        self.save_hyperparameters()

    def forward(self, item):
        """
        Notation:
            B = batch size, 
            S = number of cameras, 
            C = 3, 
            D = discrete depth,
            H = img height, 
            W = img width

        Arguments:
                 rgb_cams: (B,S,C,H,W)
               pix_T_cams: (B,S,4,4)
            cams_T_global: (B,S,4,4)
             ref_T_global: (B,4,4)
                 vox_util: vox util object
        """
        prev_bev = self.load_cache(item['frame'].cpu())

        output = self.model(
            rgb_cams=item['img'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=prev_bev,
        )

        if self.use_temporal_cache:
            self.store_cache(item['frame'].cpu(), output['bev_raw'].clone().detach())

        return output

    def load_cache(self, frames):
        idx = []
        for frame in frames:
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            if i.nelement() == 1:
                idx.append(i.item())
        if len(idx) != len(frames):
            return None
        else:
            return self.temporal_cache[idx]

    def store_cache(self, frames, bev_feat):
        if self.temporal_cache is None:
            shape = list(bev_feat.shape)
            shape[0] = self.max_cache
            self.temporal_cache = torch.zeros(shape, device=bev_feat.device, dtype=bev_feat.dtype)

        for frame, feat in zip(frames, bev_feat):
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            
            # Choose unfilled cache slot
            if i.nelement() == 0:
                i = (self.temporal_cache_frames == -2).nonzero(as_tuple=True)[0]
            
            # Choose random cache slot
            if i.nelement() == 0:
                i = torch.randint(self.max_cache, (1, 1))

            self.temporal_cache[i[0]] = feat
            self.temporal_cache_frames[i[0]] = frame

    def loss(self, target, output):
        center_img_e = output['img_center']
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        valid_g = target['valid_bev']
        center_g = target['center_bev']
        offset_g = target['offset_bev']

        B, S = target['center_img'].shape[:2]

        center_img_g = pack_seqdim(target['center_img'], B)

        center_loss = self.center_loss_fn(sigmoid(center_e), center_g)

        offset_loss = torch.abs(offset_e[:, :2] - offset_g[:, :2]).sum(dim=1, keepdim=True)
        offset_loss = reduce_masked_mean(offset_loss, valid_g)

        tracking_loss = F.smooth_l1_loss(offset_e[:, 2:], offset_g[:, 2:], reduction='none')
        tracking_loss = tracking_loss.sum(dim=1, keepdim=True)
        tracking_loss = reduce_masked_mean(tracking_loss, valid_g)

        if 'size_bev' in target:
            size_g = target['size_bev']
            rotbin_g = target['rotbin_bev']
            rotres_g = target['rotres_bev']
            size_loss = torch.abs(size_e - size_g).sum(dim=1, keepdim=True)
            size_loss = reduce_masked_mean(size_loss, valid_g)
            rot_loss = compute_rot_loss(rot_e, rotbin_g, rotres_g, valid_g)
        else:
            size_loss = torch.tensor(0.)
            rot_loss = torch.tensor(0.)

        center_factor = 1 / torch.exp(self.model.center_weight)
        center_loss_weight = center_factor * center_loss
        center_uncertainty_loss = self.model.center_weight

        offset_factor = 1 / torch.exp(self.model.offset_weight)
        offset_loss_weight = offset_factor * offset_loss
        offset_uncertainty_loss = self.model.offset_weight

        size_factor = 1 / torch.exp(self.model.size_weight)
        size_loss_weight = size_factor * size_loss
        size_uncertainty_loss = self.model.size_weight

        rot_factor = 1 / torch.exp(self.model.rot_weight)
        rot_loss_weight = rot_factor * rot_loss
        rot_uncertainty_loss = self.model.rot_weight

        tracking_factor = 1 / torch.exp(self.model.tracking_weight)
        tracking_loss_weight = tracking_factor * tracking_loss
        tracking_uncertainty_loss = self.model.tracking_weight

        # img loss
        center_img_loss = self.center_loss_fn(sigmoid(center_img_e), center_img_g) / S

        loss_dict = {
              'center_loss':   center_loss * 10,
              'offset_loss':   offset_loss * 10,
            'tracking_loss': tracking_loss,
                'size_loss':     size_loss,
                 'rot_loss':      rot_loss,
             'center_img': center_img_loss,
        }

        loss_weight_dict = {
              'center_loss':   center_loss_weight * 10,
              'offset_loss':   offset_loss_weight * 10,
            'tracking_loss': tracking_loss_weight,
                'size_loss':     size_loss_weight,
                 'rot_loss':      rot_loss_weight,
             'center_img': center_img_loss,
        }

        stats_dict = {
          'center_uncertainty_loss':   center_uncertainty_loss,
          'offset_uncertainty_loss':   offset_uncertainty_loss,
        'tracking_uncertainty_loss': tracking_uncertainty_loss,
            'size_uncertainty_loss':     size_uncertainty_loss,
             'rot_uncertainty_loss':      rot_uncertainty_loss,
        }
    
        total_loss = sum(loss_weight_dict.values()) + sum(stats_dict.values())
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        if batch_idx % 100 == 1:
            self.plot_data(target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log('val_center', loss_dict['center_loss'], batch_size=B, sync_dist=True)
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        # ref_T_global = item['ref_T_global']
        # global_T_ref = torch.inverse(ref_T_global)

        # output on bev plane
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        xy_e, xy_prev_e, scores_e, \
              classes_e, sizes_e, rzs_e = postprocess(center_e.sigmoid(), offset_e, size_e, 
                                                    rz_e=rot_e, K=self.max_detections)

        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = self.vox_util.Mem2Ref(mem_xyz, self.Y, self.Z, self.X)[..., :2]

        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy_prev = self.vox_util.Mem2Ref(mem_xyz_prev, self.Y, self.Z, self.X)[..., :2]

        # detection
        moda_zip = zip(item['frame'], item['grid_gt'], ref_xy, scores_e)

        for frame, grid_gt, xy, score in moda_zip:
            frame = int(frame.item())
            valid = score > self.conf_threshold

            self.moda_gt_list.extend([[frame, x.item(), y.item()] for x, y, _ in grid_gt[grid_gt.sum(1) != 0]])
            self.moda_pred_list.extend([[frame, x.item(), y.item()] for x, y in xy[valid]])

        # tracking
        mota_zip = zip(item['sequence_num'], item['frame'], item['grid_gt'], 
                       ref_xy.cpu(), ref_xy_prev.cpu(), scores_e.cpu())

        for seq_num, frame, grid_gt, bev_det, bev_prev, score in mota_zip:
            frame = int(frame.item())
            output_stracks = self.test_tracker.update(bev_det, bev_prev, score)

            self.mota_gt_list.extend([
                [seq_num.item(), frame, i.item(), -1, -1, -1, -1, 1, x.item(),  y.item(), -1]
                for x, y, i in grid_gt[grid_gt.sum(1) != 0]
            ])
            self.mota_pred_list.extend([
                [seq_num.item(), frame, s.track_id, -1, -1, -1, -1, s.score.item()] + s.xy.tolist() + [-1]
                for s in output_stracks
            ])

    def on_test_epoch_end(self):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else './cache'

        # detection
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        gt_path   = osp.join(log_dir, 'moda_gt.txt')

        np.savetxt(pred_path, np.array(self.moda_pred_list), fmt='%f')
        np.savetxt(  gt_path, np.array(self.moda_gt_list  ), fmt='%d')

        recall, precision, moda, modp = mod_metrics(osp.abspath(pred_path), osp.abspath(gt_path))
        self.log(f'detect/recall', recall)
        self.log(f'detect/precision', precision)
        self.log(f'detect/moda', moda)
        self.log(f'detect/modp', modp)

        # tracking
        scale = 1 if self.X == 150 else 0.025

        pred_path = osp.join(log_dir, 'mota_pred.txt')
        gt_path   = osp.join(log_dir, 'mota_gt.txt')

        np.savetxt(pred_path, np.array(self.mota_pred_list), fmt='%f', delimiter=',')
        np.savetxt(  gt_path, np.array(self.mota_gt_list  ), fmt='%f', delimiter=',')

        summary = mot_metrics(osp.abspath(pred_path), osp.abspath(gt_path), scale)
        summary = summary.loc['OVERALL']

        for key, value in summary.to_dict().items():
            if value >= 1 and key[:3] != 'num':
                value /= summary.to_dict()['num_unique_objects']
            value = value * 100 if value < 1 else value
            value = 100 - value if key == 'motp' else value
            self.log(f'track/{key}', value)

    def plot_data(self, target, output, batch_idx=0):
        center_e = output['instance_center']
        center_g = target['center_bev']

        # save plots to tensorboard in eval loop
        writer = self.logger.experiment

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        ax1.imshow(center_g[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax2.imshow(center_e[-1].amax(0).sigmoid().squeeze().cpu().numpy())

        ax1.set_title('center_g')
        ax2.set_title('center_e')
        plt.tight_layout()

        writer.add_figure(f'plot/{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import OneCycleLR

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, 
                                    total_steps=self.trainer.estimated_stepping_batches,)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step",
            },
        }


if __name__ == '__main__':

    # Reference: 
    #   https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html
    #   https://www.restack.io/p/pytorch-lightning-answer-cli-example-cat-ai

    from lightning.pytorch.cli import LightningCLI

    torch.set_float32_matmul_precision('medium')

    class TrackLightningCLI(LightningCLI):

        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.bounds", "data.init_args.bounds")
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")

    cli = TrackLightningCLI(model_class=WorldTrackModel)

    """
    Test:
        python -m libs.TrackTacular.world_track test \
                -c "./checkpoints/TrackTacular/mvx_liftnet/config.yaml" \
            --ckpt "./checkpoints/TrackTacular/mvx_liftnet/model.ckpt" \
            --data.batch_size 1
    """

