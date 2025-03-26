from __future__ import division, print_function, absolute_import

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
from tabulate import tabulate

import numpy as np

import torch
from torch.cuda import amp

from torchreid.constants import *
from ..engine import Engine
from ... import metrics
from ...losses.GiLt_loss import GiLtLoss
from ...losses.body_part_attention_loss import BodyPartAttentionLoss
from ...metrics.distance import compute_distance_matrix_using_bp_features
from ...utils import plot_body_parts_pairs_distance_distribution, \
                    plot_pairs_distance_distribution, re_ranking
from ...utils.tools import extract_test_embeddings
from ...utils.torchtools import collate
from ...utils.visualization.feature_map_visualization import display_feature_maps


class ImagePartBasedEngine(Engine):
    r"""Training/testing engine for part-based image-reid.
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        writer,
        loss_name,
        config,
        dist_combine_strat,
        batch_size_pairwise_dist_matrix,
        engine_state,
        margin=0.3,
        scheduler=None,
        use_gpu=True,
        save_model_flag=False,
        mask_filtering_training=False,
        mask_filtering_testing=False,
    ):
        super(ImagePartBasedEngine, self).__init__(
            config,
            datamanager,
            writer,
            engine_state,
            use_gpu=use_gpu,
            save_model_flag=save_model_flag,
            detailed_ranking=config.test.detailed_ranking,
        )

        self.model = model
        self.register_model("model", model, optimizer, scheduler)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parts_num = self.config.model.kpr.masks.parts_num
        self.mask_filtering_training = mask_filtering_training
        self.mask_filtering_testing = mask_filtering_testing
        self.dist_combine_strat = dist_combine_strat
        self.batch_size_pairwise_dist_matrix = batch_size_pairwise_dist_matrix
        self.losses_weights = self.config.loss.part_based.weights
        self.mixed_precision = self.config.train.mixed_precision

        self.scaler = amp.GradScaler() if self.mixed_precision else None

        # Losses
        self.GiLt = GiLtLoss(
            self.losses_weights,
            use_visibility_scores=self.mask_filtering_training,
            triplet_margin=margin,
            loss_name=loss_name,
            writer=self.writer,
            use_gpu=self.use_gpu,
            num_classes=datamanager.num_train_pids,
        )

        self.body_part_attention_loss = BodyPartAttentionLoss(
            loss_type=self.config.loss.part_based.ppl,
            use_gpu=self.use_gpu,
            best_pred_ratio=self.config.loss.part_based.best_pred_ratio,
            num_classes=self.parts_num+1,
        )

        # Timers
        self.feature_extraction_timer = self.writer.feature_extraction_timer
        self.loss_timer = self.writer.loss_timer
        self.optimizer_timer = self.writer.optimizer_timer

    def forward_backward(self, data):
        imgs, target_masks, prompt_masks, keypoints_xyc, pids, imgs_path, cam_id = self.parse_data_for_train(data)

        with amp.autocast(enabled=self.mixed_precision):
            # feature extraction
            self.feature_extraction_timer.start()
            embeddings_dict, visibility_scores_dict, \
            id_cls_scores_dict, pixels_cls_scores, spatial_features, masks = self.model(imgs, target_masks=target_masks, 
                                                                                              prompt_masks=prompt_masks, 
                                                                                              keypoints_xyc=keypoints_xyc, 
                                                                                              cam_label=cam_id)
            display_feature_maps(embeddings_dict, spatial_features, masks[PARTS], imgs_path, pids)
            self.feature_extraction_timer.stop()

            # loss
            self.loss_timer.start()
            loss, loss_summary = self.combine_losses(
                visibility_scores_dict,
                embeddings_dict,
                id_cls_scores_dict,
                pids,
                pixels_cls_scores,
                target_masks,
                bpa_weight=self.losses_weights[PIXELS]["ce"],
            )
            self.loss_timer.stop()

        # optimization step
        self.optimizer_timer.start()
        self.optimizer.zero_grad()
        if self.scaler is None:
            loss.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer_timer.stop()

        return loss, loss_summary

    def combine_losses(
        self,
        visibility_scores_dict,
        embeddings_dict,
        id_cls_scores_dict,
        pids,
        pixels_cls_scores=None,
        target_masks=None,
        bpa_weight=0,
    ):
        # 1. ReID objective:
        # GiLt loss on holistic and part-based embeddings
        loss, loss_summary = self.GiLt(
            embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids
        )

        # 2. Part prediction objective:
        # Body part attention loss on spatial feature map
        if (
            pixels_cls_scores is not None
            and target_masks is not None
            and bpa_weight > 0
        ):
            # resize external masks to fit feature map size
            # target_masks = nn.functional.interpolate(  # FIXME should be useless
            #     target_masks,
            #     pixels_cls_scores.shape[2::],
            #     mode="bilinear",
            #     align_corners=True,
            # )
            # compute target part index for each spatial location, i.e. each spatial location (pixel) value indicate
            # the (body) part that spatial location belong to, or 0 for background.
            pixels_cls_score_targets = target_masks.argmax(dim=1)  # [N, Hf, Wf]  # TODO check first indices are prioretized if equality
            # compute the classification loss for each pixel
            bpa_loss, bpa_loss_summary = self.body_part_attention_loss(
                pixels_cls_scores, pixels_cls_score_targets
            )
            loss += bpa_weight * bpa_loss
            loss_summary = {**loss_summary, **bpa_loss_summary}

        return loss, loss_summary

    def _feature_extraction(self, data_loader, feature_dir: str = './'):

        keys = ['features','part_masks','pixels_cls_scores','parts_vis_scores',
                'samples','pedestrian_ids','camera_ids']
        for d in keys:
            if os.path.isdir(osp.join(feature_dir, d)) is False:
                os.makedirs(osp.join(feature_dir, d))

        cnt = 0
        for idx, samples in enumerate(tqdm(data_loader, desc=f"FeaturExtraction")):
            cnt += 1
            
            if all([os.path.isfile(
                    osp.join(feature_dir, k, f'{idx:07d}.{ext}'))
                                      for k, ext in zip(keys, ['pt']*4 + ['npy']*3)]):
                continue

            imgs, target_masks, prompt_masks, \
                 keypoints_xyc, pids, camids = self.parse_data_for_eval(samples)
            if self.use_gpu:
                if target_masks is not None:
                    target_masks = target_masks.cuda()
                if prompt_masks is not None:
                    prompt_masks = prompt_masks.cuda()
                imgs = imgs.cuda()

            self.writer.test_batch_timer.start()
            model_output = self.model(imgs, target_masks=target_masks, 
                                            prompt_masks=prompt_masks, 
                                            keypoints_xyc=keypoints_xyc, 
                                                cam_label=camids)
            features, visibility_scores, \
            parts_masks, pixels_cls_scores = extract_test_embeddings(
                model_output, self.config.model.kpr.test_embeddings
            )
            self.writer.test_batch_timer.stop()

            features = features.data.cpu().half()
            parts_masks = parts_masks.data.cpu()
            
            torch.save(features, osp.join(feature_dir, 'features', f'{idx:07d}.pt'))
            torch.save(parts_masks, osp.join(feature_dir, 'part_masks', f'{idx:07d}.pt'))

            if self.mask_filtering_testing:
                parts_vis_scores = visibility_scores.data.cpu().half()
                torch.save(parts_vis_scores, osp.join(feature_dir, 'parts_vis_scores', f'{idx:07d}.pt'))

            if pixels_cls_scores is not None:
                pixels_cls_scores = pixels_cls_scores.data.cpu().half()
                torch.save(pixels_cls_scores, osp.join(feature_dir, 'pixels_cls_scores', f'{idx:07d}.pt'))
            
            np.save(osp.join(feature_dir, 'pedestrian_ids', f'{idx:07d}.npy'), pids)
            np.save(osp.join(feature_dir,     'camera_ids', f'{idx:07d}.npy'), camids)
            # np.save(osp.join(feature_dir,        'samples', f'{idx:07d}.npy'), samples)

        return cnt

    @torch.no_grad()
    def _evaluate(
        self,
        epoch,
        dataset_name="",
        query_loader=None,
        gallery_loader=None,
        dist_metric="euclidean",
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        visrank_q_idx_list=[],
        visrank_count=10,
        save_dir="",
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        save_feats_and_ids=False,
    ):
        save_feats_and_ids = True
        
        queries_dir = osp.join(save_dir, "query")
        if os.path.isdir(queries_dir) is False:
            os.makedirs(queries_dir)

        gallery_dir = osp.join(save_dir, "gallery")
        if os.path.isdir(gallery_dir) is False:
            os.makedirs(gallery_dir)

        print("\nExtracting features from query set ...")
        q_count = self._feature_extraction(query_loader, queries_dir)
        print(f"Done, obtained {q_count} records")

        print("\nExtracting features from gallery set ...")
        g_count = self._feature_extraction(gallery_loader, gallery_dir)
        print(f"Done, obtained {g_count} records")

        print("\nFeature extraction speed: {:.4f} sec / batch".format(self.writer.test_batch_timer.avg))

        # Re-load
        q_feats = torch.cat([torch.load(f) for f in glob(f'{queries_dir}/features/*.pt')], dim=0)
        g_feats = torch.cat([torch.load(f) for f in glob(f'{gallery_dir}/features/*.pt')], dim=0)

        torch.save(q_feats, osp.join(save_dir, "query_features.pt"))
        torch.save(g_feats, osp.join(save_dir, "gallery_features.pt"))

        # q_feats = torch.load(osp.join(save_dir, "query_features.pt"))
        # g_feats = torch.load(osp.join(save_dir, "gallery_features.pt"))

        self.writer.performance_evaluation_timer.start()

        if normalize_feature:
            print("\nNormalizing features with L2-norm ...")
            q_feats = self.normalize(q_feats)
            g_feats = self.normalize(g_feats)

            torch.save(q_feats, osp.join(save_dir, "query_features_norm.pt"))
            torch.save(g_feats, osp.join(save_dir, "gallery_features_norm.pt"))

            # q_feats = torch.load(osp.join(save_dir, "query_features_norm.pt"))
            # g_feats = torch.load(osp.join(save_dir, "gallery_features_norm.pt"))

        qf_parts_visibility = torch.cat([torch.load(f) for f in glob(f'{queries_dir}/parts_vis_scores/*.pt')], dim=0)
        gf_parts_visibility = torch.cat([torch.load(f) for f in glob(f'{gallery_dir}/parts_vis_scores/*.pt')], dim=0)

        torch.save(qf_parts_visibility, osp.join(save_dir, "qf_parts_visibility.pt"))
        torch.save(gf_parts_visibility, osp.join(save_dir, "gf_parts_visibility.pt"))

        # qf_parts_visibility = torch.load(osp.join(save_dir, "qf_parts_visibility.pt"))
        # gf_parts_visibility = torch.load(osp.join(save_dir, "gf_parts_visibility.pt"))

        q_pids = np.concatenate([np.load(f) for f in glob(f'{queries_dir}/pedestrian_ids/*.npy')], axis=0)
        g_pids = np.concatenate([np.load(f) for f in glob(f'{gallery_dir}/pedestrian_ids/*.npy')], axis=0)

        np.save(osp.join(save_dir, "q_pids.npy"), q_pids)
        np.save(osp.join(save_dir, "g_pids.npy"), g_pids)

        # q_pids = np.load(osp.join(save_dir, "q_pids.npy"))
        # g_pids = np.load(osp.join(save_dir, "g_pids.npy"))

        q_camids = np.concatenate([np.load(f) for f in glob(f'{queries_dir}/camera_ids/*.npy')], axis=0)
        g_camids = np.concatenate([np.load(f) for f in glob(f'{gallery_dir}/camera_ids/*.npy')], axis=0)

        np.save(osp.join(save_dir, "q_camids.npy"), q_camids)
        np.save(osp.join(save_dir, "g_camids.npy"), g_camids)

        # q_camids = np.load(osp.join(save_dir, "q_camids.npy"))
        # g_camids = np.load(osp.join(save_dir, "g_camids.npy"))

        q_anns = None
        g_anns = None

        print("\nComputing distance matrix with metric = {} ...".format(dist_metric))
        distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(
            q_feats,
            g_feats,
            qf_parts_visibility,
            gf_parts_visibility,
            self.dist_combine_strat,
            self.batch_size_pairwise_dist_matrix,
            self.use_gpu,
            dist_metric,
        )
        distmat = distmat.numpy()
        body_parts_distmat = body_parts_distmat.numpy()

        if rerank:
            print("\nApplying person re-ranking ...")
            distmat_qq, body_parts_distmat_qq = compute_distance_matrix_using_bp_features(
                                                    q_feats,
                                                    q_feats,
                                                    qf_parts_visibility,
                                                    qf_parts_visibility,
                                                    self.dist_combine_strat,
                                                    self.batch_size_pairwise_dist_matrix,
                                                    self.use_gpu,
                                                    dist_metric,
                                                )
            distmat_gg, body_parts_distmat_gg = compute_distance_matrix_using_bp_features(
                                                    g_feats,
                                                    g_feats,
                                                    gf_parts_visibility,
                                                    gf_parts_visibility,
                                                    self.dist_combine_strat,
                                                    self.batch_size_pairwise_dist_matrix,
                                                    self.use_gpu,
                                                    dist_metric,
                                                )
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        eval_metric = self.datamanager.test_loader[dataset_name]["query"].dataset.eval_metric

        print("\nComputing CMC and mAP for eval metric '{}' ...".format(eval_metric))
        eval_metrics = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_anns=q_anns,
            g_anns=g_anns,
            eval_metric=eval_metric,
            max_rank=np.array(ranks).max(),
            use_cython=False,
        )

        mAP = eval_metrics["mAP"]
        cmc = eval_metrics["cmc"]
        print("\n\n** Results **")
        print("mAP: {:.2%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))

        for metric in eval_metrics.keys():
            if metric not in {"mAP", "cmc", "all_AP", "all_cmc"}:
                print("{:<20}: {}".format(metric, eval_metrics[metric]))

        # Parts ranking
        if self.detailed_ranking:
            self.display_individual_parts_ranking_performances(
                body_parts_distmat,
                cmc,
                g_camids,
                g_pids,
                mAP,
                q_camids,
                q_pids,
                eval_metric,
            )

        # TODO move below to writer
        plot_body_parts_pairs_distance_distribution(body_parts_distmat, q_pids, g_pids, "Query-gallery")
        print("\nEvaluate distribution of distances of pairs with same id vs different ids")
        (
            same_ids_dist_mean,
            same_ids_dist_std,
            different_ids_dist_mean,
            different_ids_dist_std,
            ssmd,
        ) = plot_pairs_distance_distribution(distmat, q_pids, g_pids, "Query-gallery")  
        # TODO separate ssmd from plot, put plot in writer
        print("Positive pairs distance distribution mean: {:.3f}".format(same_ids_dist_mean))
        print("Positive pairs distance distribution std: {:.3f}".format(same_ids_dist_std))
        print("Negative pairs distance distribution mean: {:.3f}".format(different_ids_dist_mean))
        print("Negative pairs distance distribution std: {:.3f}".format(different_ids_dist_std))
        print("SSMD = {:.4f}".format(ssmd))

        # if groundtruth target body masks are provided, compute part prediction accuracy
        avg_pxl_pred_accuracy = 0.0
        if (
                "target_masks" in q_anns
            and "target_masks" in g_anns
            and q_pxl_scores_ is not None
            and g_pxl_scores_ is not None
        ):
            q_pxl_pred_accuracy = self.compute_pixels_cls_accuracy(torch.from_numpy(q_anns["target_masks"]), q_pxl_scores_)
            g_pxl_pred_accuracy = self.compute_pixels_cls_accuracy(torch.from_numpy(g_anns["target_masks"]), g_pxl_scores_)
            avg_pxl_pred_accuracy = (
                  q_pxl_pred_accuracy * len(q_parts_masks)
                + g_pxl_pred_accuracy * len(g_parts_masks)
              ) / (len(q_parts_masks) + len(g_parts_masks))
            print(
                "\nPixel prediction accuracy for query = {:.2f}% and for gallery = {:.2f}% and on average = {:.2f}%".format(
                    q_pxl_pred_accuracy, g_pxl_pred_accuracy, avg_pxl_pred_accuracy
                )
            )

        if visrank:
            self.writer.visualize_rank(
                self.datamanager.test_loader[dataset_name],
                dataset_name,
                distmat,
                save_dir,
                visrank_topk,
                visrank_q_idx_list,
                visrank_count,
                body_parts_distmat,
                qf_parts_visibility,
                gf_parts_visibility,
                q_parts_masks,
                g_parts_masks,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                q_anns,
                g_anns,
                eval_metrics,
            )

        self.writer.visualize_embeddings(
            q_feats,
            g_feats,
            q_pids,
            g_pids,
            self.datamanager.test_loader[dataset_name],
            dataset_name,
            qf_parts_visibility,
            gf_parts_visibility,
            mAP,
            cmc[0],
        )
        self.writer.performance_evaluation_timer.stop()
        return cmc, mAP, ssmd, avg_pxl_pred_accuracy

    def compute_pixels_cls_accuracy(self, target_masks, pixels_cls_scores):
        if pixels_cls_scores.is_cuda:
            target_masks = target_masks.cuda()
        # target_masks = nn.functional.interpolate(
        #     target_masks,
        #     pixels_cls_scores.shape[2::],
        #     mode="bilinear",
        #     align_corners=True,
        # )  # Best perf with bilinear here and nearest in resize transform
        pixels_cls_score_targets = target_masks.argmax(dim=1)  # [N, Hf, Wf]
        pixels_cls_score_targets = pixels_cls_score_targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(
            0, 2
        )  # [N*Hf*Wf, M]
        accuracy = metrics.accuracy(pixels_cls_scores, pixels_cls_score_targets)[0]
        return accuracy.item()

    def display_individual_parts_ranking_performances(
        self,
        body_parts_distmat,
        cmc,
        g_camids,
        g_pids,
        mAP,
        q_camids,
        q_pids,
        eval_metric,
    ):
        print("Parts embeddings individual rankings :")
        bp_offset = 0
        if GLOBAL in self.config.model.kpr.test_embeddings:
            bp_offset += 1
        if FOREGROUND in self.config.model.kpr.test_embeddings:
            bp_offset += 1
        table = []
        for bp in range(
            0, body_parts_distmat.shape[0]
        ):  # TODO DO NOT TAKE INTO ACCOUNT -1 DISTANCES!!!!
            perf_metrics = metrics.evaluate_rank(
                body_parts_distmat[bp],
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric,
                max_rank=10,
                use_cython=False,
            )
            title = "p {}".format(bp - bp_offset)
            if bp < bp_offset:
                if bp == 0:
                    if GLOBAL in self.config.model.kpr.test_embeddings:
                        title = GLOBAL
                    else:
                        title = FOREGROUND
                if bp == 1:
                    title = FOREGROUND
            mAP = perf_metrics["mAP"]
            cmc = perf_metrics["cmc"]
            table.append([title, mAP, cmc[0], cmc[4], cmc[9]])
        headers = ["embed", "mAP", "R-1", "R-5", "R-10"]
        print(tabulate(table, headers, tablefmt="fancy_grid", floatfmt=".3f"))

    def parse_data_for_train(self, data):
        imgs = data["image"]
        imgs_path = data["img_path"]
        target_masks = data.get("target_masks", None)
        prompt_masks = data.get("prompt_masks", None)
        keypoints_xyc = data.get("keypoints_xyc", None)
        pids = data["pid"]
        cam_id = data["camid"]


        if self.use_gpu:
            imgs = imgs.cuda()
            cam_id = cam_id.cuda()
            if target_masks is not None:
                target_masks = target_masks.cuda()
            if prompt_masks is not None:
                prompt_masks = prompt_masks.cuda()
            if keypoints_xyc is not None:
                keypoints_xyc = keypoints_xyc.cuda()
            pids = pids.cuda()

        if target_masks is not None:
            assert target_masks.shape[1] == (
                self.config.model.kpr.masks.parts_num + 1
            ), f"masks.shape[1] ({target_masks.shape[1]}) != parts_num ({self.config.model.kpr.masks.parts_num + 1})"

        return imgs, target_masks, prompt_masks, keypoints_xyc, pids, imgs_path, cam_id

    def parse_data_for_eval(self, data):
        imgs = data["image"]
        target_masks = data.get("target_masks", None)
        prompt_masks = data.get("prompt_masks", None)
        keypoints_xyc = data.get("keypoints_xyc", None)
        pids = data["pid"]
        camids = data["camid"]
        return imgs, target_masks, prompt_masks, keypoints_xyc, pids, camids
