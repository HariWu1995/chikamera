import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.evaluation.evaluate import evaluate


eps = 1e-4


class BaseTrainer(object):

    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):

    def __init__(self, model, criterion, logdir, denormalize, cls_thresh=0.4, alpha=1.0):
        super(BaseTrainer, self).__init__()
        self.logdir = logdir
        self.model = model
        self.alpha = alpha
        self.criterion = criterion
        self.cls_thresh = cls_thresh
        self.denormalize = denormalize

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()

        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()

        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0

        print("\n\nTraining ...")

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, (data, map_gt, imgs_gt, _) in pbar:
            optimizer.zero_grad()
            map_res, imgs_res = self.model(data)
            t_f = time.time()
            t_forward += t_f - t_b
            loss = 0
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                   loss / len(imgs_gt) * self.alpha
            loss.backward()
            optimizer.step()
            losses += loss.item()

            pred = (map_res > self.cls_thresh).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive =   pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + eps)
            recall    = true_positive / (true_positive + false_negative + eps)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()

            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0

            pbar.set_description(
                f'Epoch: {epoch}, Batch: {batch_idx+1}, ' + \
                f'Loss: {loss.item():.6f}, ' + \
                f'prec: {precision*100:.2f}%, recall: {recall*100:.2f}%, ' + \
                f'maxima: {map_res.max():.3f}'
            )

        t1 = time.time()
        t_all = t1 - t0
        print(
            f'Train #Epoch: {epoch}, #Batch: {len(data_loader)}, ' + \
            f'Loss: {losses/len(data_loader):.5f}, ' + \
            f'Precision: {precision_s.avg*100:.2f}%, Recall: {recall_s.avg*100:.1f}%, ' + \
            f'Time: {t_all:.2f}s'
        )

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        self.model.eval()

        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()

        if res_fpath is not None:
            assert gt_fpath is not None

        print("\n\nTesting ...")

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, (data, map_gt, imgs_gt, frame) in pbar:
            with torch.no_grad():
                map_res, imgs_res = self.model(data)

            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thresh].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thresh).nonzero()

                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = 0
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
            loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                   loss / len(imgs_gt) * self.alpha
            losses += loss.item()
            
            pred = (map_res > self.cls_thresh).int().to(map_gt.device)

            # Calculate True (T) / False (F) - Positive (P) / Negative (N)
            # TP = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            # FP =   pred.sum().item() - TP
            # FN = map_gt.sum().item() - TP

            TP = torch.sum((map_gt == 1) & (pred == 1)).item()
            TN = torch.sum((map_gt == 0) & (pred == 0)).item()
            FP = torch.sum((map_gt == 0) & (pred == 1)).item()
            FN = torch.sum((map_gt == 1) & (pred == 0)).item()

            precision = TP / (TP + FP + eps)
            recall    = TP / (TP + FN + eps)
            precision_s.update(precision)
            recall_s.update(recall)

            pbar.set_description(
                f'Loss: {loss.item():.6f}, ' + \
                f'prec: {precision*100:.2f}%, ' + \
                f'recall: {recall*100:.2f}%, ' + \
                f'maxima: {map_res.max():.3f}'
            )

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            self.visualize(data, imgs_res, map_res, map_gt, data_loader.dataset.map_kernel)

        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(
                os.path.abspath(
                os.path.dirname(res_fpath)) + '/all_res.txt', 
                all_res_list.numpy(), 
                fmt=['%d','%d','%d','%.8f']
            )

            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), 
                                                     os.path.abspath(gt_fpath),
                                                     data_loader.dataset.base.__name__)

            # FIXME: Unofiicial python evaluation tool for some purposes.
            # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), 
            #                                             os.path.abspath(gt_fpath),
            #                                             data_loader.dataset.base.__name__)

            print('MODA: {:.2f}%, MODP: {:.2f}%, precision: {:.2f}%, recall: {:.2f}%'.
                  format(moda, modp, precision, recall))

        return losses / len(data_loader), precision_s.avg * 100, moda

    def visualize(self, data, imgs_res, map_res, map_gt, map_kernel):

        fig = plt.figure()
        subplt0 = fig.add_subplot(211, title="output")
        subplt1 = fig.add_subplot(212, title="target")
        subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
        subplt1.imshow(self.criterion._target_transform(map_res, map_gt, map_kernel)
                        .cpu().detach().numpy().squeeze())
        plt.savefig(os.path.join(self.logdir, 'map.jpg'))
        plt.close(fig)

        # visualizing the heatmap for per-view estimation
        heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
        heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()

        img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
        img0 = Image.fromarray((img0 * 255).astype('uint8'))

        head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
        foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)

        head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
        foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))


class BBOXTrainer(BaseTrainer):

    def __init__(self, model, criterion, cls_thresh):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thresh = cls_thresh

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()

        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()

            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()

        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)

            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()

            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thresh
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)

