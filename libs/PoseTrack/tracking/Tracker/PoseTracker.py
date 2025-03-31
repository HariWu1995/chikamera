from functools import partial
from tqdm import tqdm
from time import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# NOTE: remember to build & install aic_cpp
import aic_cpp

import os
import sys
sys.path.append("./src")

from utils.timeout import timeout   # timeout decorator, return KeyboardInterrupt

from Tracker.states import TrackState, Track2DState
from Tracker.filter import KalmanFilterBbox as BboxFilter
from Tracker.matching import IoUs
from Tracker.PoseTrack import PoseTrack
from Solver.bip_solver import GLPKSolver
from Cameras.process import find_view_for_cluster


eps = 1e-5

time_solver_lim = int(os.environ.get("GLPK_TIME_LIMIT_MS", 30_000) / 1_000)
time_other_lim = 60
time_runs_out = time_solver_lim + time_other_lim


class PoseTracker:

    def __init__(self, cameras, **kwargs):
        self.cameras = cameras
        self.num_cam = len(cameras)
        self.num_keypoints = 17

        self.solver = GLPKSolver(min_affinity = -kwargs.get('affinity_range', 1_000), 
                                 max_affinity =  kwargs.get('affinity_range', 1_000))
        self.tracks = []
        self.bank_size = kwargs.get('bank_size', 100)
        self.decay_weight = kwargs.get('decay_weight', 0.5)

        self.thresh_p2l_3d = kwargs.get('thresh_p2l_3d', 0.3)
        self.thresh_2d = kwargs.get('thresh_2d', 0.3)
        self.thresh_epi = kwargs.get('thresh_epi', 0.2)
        self.thresh_homo = kwargs.get('thresh_homo', 1.5)
        self.thresh_bbox = kwargs.get('thresh_bbox', 0.8)
        self.thresh_kpts = kwargs.get('thresh_kpts', 0.7)
        self.thresh_reid = kwargs.get('thresh_reid', 0.5)

        self.main_joints = np.array([5, 6, 11, 12, 13, 14, 15, 16])
        self.upper_body = np.array([5, 6, 11, 12])
        self.feet_idx = np.array([15, 16])

    def compute_reid_aff(self, entity_list_mv, avail_tracks):

        n_track = len(avail_tracks)

        reid_sim_mv = []
        reid_weight = []

        for v in range(self.num_cam):
            reid_sim_sv    = np.zeros((len(entity_list_mv[v]), n_track))
            reid_weight_sv = np.zeros((len(entity_list_mv[v]), n_track)) + eps
    
            for s_id, entity in enumerate(entity_list_mv[v]):
                if entity.bbox[-1] < self.thresh_bbox:
                    continue
                for t_id, track in enumerate(avail_tracks):
                    if not len(track.track2ds[v].state):
                        continue
                    if (Track2DState.Occluded not in track.track2ds[v].state) \
                    and (Track2DState.Missing not in track.track2ds[v].state):
                        continue 
                    reid_sim = track.feat_bank @ entity.reid
                    reid_sim = reid_sim[reid_sim > 0]
                    if reid_sim.size:
                        reid_sim_sv[s_id, t_id] = np.max(reid_sim) - self.thresh_reid
                        reid_weight_sv[s_id, t_id] = 1

            reid_sim_mv.append(reid_sim_sv)
            reid_weight.append(reid_weight_sv)

        return reid_sim_mv, reid_weight

    def compute_3dkp_aff(self, entity_list_mv, avail_tracks):
        aff_mv = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            aff_sv = np.zeros((len(entity_list_mv[v]), n_track))
            cam = self.cameras[v]

            for s_id, entity in enumerate(entity_list_mv[v]):
    
                joints_h = np.vstack((entity.kpts[:, :-1].T, np.ones((1, self.num_keypoints))))
                joints_rays = cam.project_inv @ joints_h
                joints_rays /= joints_rays[-1]
                joints_rays = joints_rays[:-1]
                joints_rays -= np.repeat(cam.pos.reshape(3, 1), self.num_keypoints, axis=1)
                joints_rays = joints_rays / (np.linalg.norm(joints_rays, axis=0) + eps)
                joints_rays = joints_rays.T # 17*3

                for t_id, track in enumerate(avail_tracks):
    
                    aff = np.zeros(self.num_keypoints)
                    kp_3d = track.kpts_3d
    
                    k_idx = np.where(entity.kpts[:, -1] < self.thresh_kpts)[0]
                    aff[k_idx] = Point2LineDist(kp_3d[k_idx, :-1], cam.pos, joints_rays[k_idx])
                    
                    aff  = 1 - aff / self.thresh_p2l_3d
                    aff = aff * entity.kpts[:, -1] * np.exp(-track.age_3D)

                    valid = (entity.kpts[:, -1] > self.thresh_kpts) * (kp_3d[:, -1] > 0)
                    aff_sv[s_id, t_id] = np.sum(aff) / (np.sum(valid * np.exp(-track.age_3D)) + eps)

            aff_mv.append(aff_sv)

        return aff_mv
    
    def compute_2dkp_aff(self, entity_list_mv, avail_tracks):
        aff_mv = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            aff_sv = np.zeros((len(entity_list_mv[v]), n_track))

            for s_id, entity in enumerate(entity_list_mv[v]):
                joints_s = entity.kpts
                
                for t_id, track in enumerate(avail_tracks):
                    joints_t = track.keypoints_mv[v]
                    dist = np.linalg.norm(joints_t[:, :-1] - joints_s[:, :-1], axis=1)
                    aff = 1 - dist / (self.thresh_2d * (np.linalg.norm(track.bbox_mv[v][2:4] - track.bbox_mv[v][:2]) + eps))
                    valid = (joints_t[:, -1] > self.thresh_kpts) * (joints_s[:, -1] > self.thresh_kpts)
                    aff = aff * valid * np.exp(-track.age_2D[v])
                    aff_sv[s_id, t_id] = np.sum(aff) / (np.sum(valid * np.exp(-track.age_2D[v])) + eps)

            aff_mv.append(aff_sv)

        return aff_mv
    
    def compute_epi_homo_aff(self, entity_list_mv, avail_tracks):
        aff_mv = []
        aff_homo = [] 
        age_2D_thresh = 1

        n_track = len(avail_tracks)
        mv_rays = self.calculate_joint_rays(entity_list_mv)

        for v in range(self.num_cam):
            pos = self.cameras[v].pos
            cam = self.cameras[v]
            sv_rays = mv_rays[v]

            aff_sv      = np.zeros((len(entity_list_mv[v]), n_track))
            aff_homo_sv = np.zeros((len(entity_list_mv[v]), n_track))

            for s_id, entity in enumerate(entity_list_mv[v]):
                
                joints_s = entity.kpts
                feet_valid_s = np.all(joints_s[self.feet_idx, -1] > self.thresh_kpts)

                feet_s    = aic_cpp.compute_feet_s(joints_s, self.feet_idx, cam.homo_feet_inv)
                box_pos_s = aic_cpp.compute_box_pos_s(entity.bbox, cam.homo_inv)
                box_valid_s = True

                for t_id, track in enumerate(avail_tracks):
                    joints_t = track.keypoints_mv
                    aff_sv     [s_id, t_id], \
                    aff_homo_sv[s_id, t_id] = aic_cpp.loop_t_homo_full(
                                                    joints_t,
                                                    joints_s,
                                                    track.age_bbox,
                                                    track.age_2D,
                                                    feet_s,
                                                    feet_valid_s,
                                                    v,
                                                    self.thresh_epi,
                                                    self.thresh_homo,
                                                    self.thresh_kpts,
                                                    age_2D_thresh,
                                                    sv_rays[s_id],
                                                    self.cameras,
                                                    box_pos_s,
                                                    box_valid_s,
                                                    track.bbox_mv
                                                )
                    continue

                    # ⚠️ Why continue if there are more code
                    aff_ss = []
                    aff_homo_ss = []
                    if feet_valid_s:
                        feet_valid_t = (joints_t[:, self.feet_idx[0], -1] > self.thresh_kpts) \
                                     & (joints_t[:, self.feet_idx[1], -1] > self.thresh_kpts)

                    valid = (joints_t[:, :, -1] > self.thresh_kpts) \
                          & (joints_s[   :, -1] > self.thresh_kpts)
                    
                    for vj in range(self.num_cam):
                        if v == vj or track.age_bbox[vj] >= 2:
                            continue

                        pos_j = self.cameras[vj].pos
                        track_rays_sv = track.calculate_target_rays(vj)

                        aff_temp = aic_cpp.epipolar_3d_score_norm(pos, sv_rays[s_id], 
                                                                pos_j, track_rays_sv, self.thresh_epi)
                        _aff_ss = aic_cpp.aff_sum(aff_temp, valid[vj], track.age_2D[vj], 1)
                        if _aff_ss != 0:
                            aff_ss.append(_aff_ss)

                        if feet_valid_s and feet_valid_t[vj]:
                            _aff_homo_ss = aic_cpp.compute_feet_distance(joints_t[vj], self.feet_idx, 
                                                        self.cameras[vj].homo_feet_inv, feet_s, self.thresh_homo)
                            aff_homo_ss.append(_aff_homo_ss)

                    aff_homo_sv[s_id, t_id] = sum(aff_homo_ss) / (len(aff_homo_ss) + eps)
                    aff_sv     [s_id, t_id] = sum(aff_ss)      / (len(aff_ss)      + eps)

            aff_homo.append(aff_homo_sv)
            aff_mv.append(aff_sv)

        return aff_mv, aff_homo                            

    def compute_bbox_iou_aff(self, entity_list_mv, avail_tracks):
        aff_mv = []
        iou_mv = []
        ovr_det_mv = []
        ovr_tgt_mv = []

        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            iou     = np.zeros((len(entity_list_mv[v]), n_track))
            ovr_det = np.zeros((len(entity_list_mv[v]), len(entity_list_mv[v])))

            if iou.size == 0:
                aff_mv.append(iou)
                iou_mv.append(iou)
                ovr_det_mv.append(ovr_det)
                ovr_tgt_mv.append(iou)
                continue
            
            detection_bboxes = np.stack([detection.bbox for detection in entity_list_mv[v]])[:, :5]

            multi_mean       = np.stack([track.bbox_filter[v].mean.copy()       if track.bbox_filter[v].mean       is not None else np.array([1,1,1,1,0,0,0,0]) for track in avail_tracks])
            multi_covariance = np.stack([track.bbox_filter[v].covariance.copy() if track.bbox_filter[v].covariance is not None else np.eye(8)                   for track in avail_tracks])
            
            multi_mean,\
            multi_covariance = avail_tracks[0].bbox_filter[v].multi_predict(multi_mean, multi_covariance)
            
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                if avail_tracks[i].bbox_filter[v].mean is not None:
                    avail_tracks[i].bbox_filter[v].mean = mean
                    avail_tracks[i].bbox_filter[v].covariance = cov

            detection_score  = detection_bboxes[:, -1]
            detection_bboxes = detection_bboxes[:, :4]

            track_bboxes = self.xyah_to_ltrb(multi_mean[:, :4].copy())
            for i in range(len(track_bboxes)):
                if avail_tracks[i].bbox_filter[v].mean is None:
                    track_bboxes[i] = avail_tracks[i].bbox_mv[v][:4]

            iou = IoUs(detection_bboxes.copy(),
                           track_bboxes.copy())
            iou[np.isnan(iou)] = 0

            age = np.array([track.age_bbox[v] for track in avail_tracks])
            
            ovr     = aic_cpp.bbox_overlap_rate(detection_bboxes.copy(),     track_bboxes.copy())
            ovr_det = aic_cpp.bbox_overlap_rate(detection_bboxes.copy(), detection_bboxes.copy())

            ovr_tgt_mv.append(ovr * (age <= 15))            
            ovr_det_mv.append(ovr_det)
            
            iou_mv.append(iou * (age <= 15))             
            iou = (((iou-0.5) * (age <= 15)).T * detection_score).T
            aff_mv.append(iou)
        
        return aff_mv, iou_mv, ovr_det_mv, ovr_tgt_mv

    def calculate_joint_rays(self, entity_list_mv):
        mv_rays = []
        for v in range(self.num_cam):
            cam = self.cameras[v]
            sv_rays = []
            n_detect = len(entity_list_mv[v])
            entity_sv = entity_list_mv[v]
            for s_id, entity in enumerate(entity_sv):
                joints_h = np.vstack((entity.kpts[:,:-1].T, np.ones((1, self.num_keypoints)))) # 3*n
                joints_rays =  cam.project_inv @ joints_h
                joints_rays /= joints_rays[-1]
                joints_rays = joints_rays[:-1]
                joints_rays -= np.repeat(cam.pos.reshape(3,1), self.num_keypoints, axis=1)
                joints_rays_norm = joints_rays / (np.linalg.norm(joints_rays, axis=0) + eps)
                joints_rays_norm = joints_rays_norm.T
                sv_rays.append(joints_rays_norm) # 17*3
            mv_rays.append(sv_rays)
        return mv_rays
    
    def match_with_miss_tracks(self, new_track, miss_tracks):
        if len(miss_tracks) == 0:
            self.tracks.append(new_track)
            return
        
        reid_sim = np.zeros(len(miss_tracks))
        for t_id, track in enumerate(miss_tracks):
            if new_track.feat_count == 0 \
                or track.feat_count == 0:
                continue
            if track.feat_count >= self.bank_size:
                bank = track.feat_bank
            else:
                bank =     track.feat_bank[:    track.feat_count % self.bank_size]
            new_bank = new_track.feat_bank[:new_track.feat_count % self.bank_size]            
            reid_sim[t_id] = np.max(new_bank @ bank.T)
        
        t_id = np.argmax(reid_sim)

        if reid_sim[t_id] > 0.5:
            miss_tracks[t_id].reactivate(new_track)
            print("reactivate", miss_tracks[t_id].id, miss_tracks[t_id].valid_views)
        else:
            self.tracks.append(new_track)
            print("new init",new_track.id, new_track.valid_views)

    def target_init(self, entity_list_mv, miss_tracks, iou_det_mv, ovr_det_mv, ovr_tgt_mv):
        cam_idx_map = []    # cam_idx_map for per det
        det_count = []      # per view det count
        det_all_count = [0]

        for v in range(self.num_cam):
            det_count.append(len(entity_list_mv[v]))
            det_all_count.append(det_all_count[-1] + det_count[-1])
            cam_idx_map += [v] * det_count[-1]
        
        if det_all_count[-1] == 0:
            return self.tracks

        det_num = det_all_count[-1]

        aff_homo = np.ones((det_num, det_num)) * (-10_000)
        aff_epi = np.ones((det_num, det_num)) * (-10_000)

        t1 = time()
        mv_rays = self.calculate_joint_rays(entity_list_mv)
        t2 = time()
        print(f'\n\nCalculated joint rays in {t2-t1:.2f} seconds')

        pbar = tqdm()
        for vi in range(self.num_cam):
            entities_vi = entity_list_mv[vi]
            pos_i = self.cameras[vi].pos
            for vj in range(vi, self.num_cam):
                if vi == vj:
                    continue
                else:
                    pos_j = self.cameras[vj].pos
                    entities_vj = entity_list_mv[vj]

                    aff_temp      = np.zeros((det_count[vi], det_count[vj]))
                    reid_sim_temp = np.zeros((det_count[vi], det_count[vj]))
                    aff_homo_temp = np.zeros((det_count[vi], det_count[vj]))
                    
                    # calculate for each det pair
                    for a in range(det_count[vi]):
                        entity_a = entities_vi[a]
                        feet_valid_a = np.all(entity_a.kpts[self.feet_idx, -1] > self.thresh_kpts)
                        if feet_valid_a:
                            feet_a = np.mean(entity_a.kpts[self.feet_idx, :-1], axis=0)
                            feet_a = self.cameras[vi].homo_feet_inv @ np.array([feet_a[0], feet_a[1], 1])
                        else:
                            feet_a = np.array([(entity_a.bbox[0] + entity_a.bbox[0]) / 2, entity_a.bbox[3]])
                            feet_a = self.cameras[vi].homo_inv @ np.array([feet_a[0], feet_a[1], 1])
                        feet_a = feet_a[:-1] / feet_a[-1]
                        feet_valid_a = True
                        
                        for b in range(det_count[vj]):
                            entity_b = entities_vj[b]
                            aff = np.zeros(self.num_keypoints)
                            valid_kp = (entity_a.kpts[:, -1] > self.thresh_kpts) \
                                     & (entity_b.kpts[:, -1] > self.thresh_kpts)
                            j_id = np.where(valid_kp)[0]

                            aff[j_id] = aic_cpp.epipolar_3d_score_norm(pos_i, mv_rays[vi][a][j_id, :], 
                                                                       pos_j, mv_rays[vj][b][j_id, :], self.thresh_epi)
                            pbar.set_description(f"View {vi} -> View {vj} / Person {a} -> Person {b}")

                            if feet_valid_a and np.all(entity_b.kpts[self.feet_idx, -1] > self.thresh_kpts):
                                feet_b = np.mean(entity_b.kpts[self.feet_idx, :-1], axis=0)
                                feet_b = self.cameras[vj].homo_feet_inv @ np.array([feet_b[0], feet_b[1], 1])
                            else:
                                feet_b = np.array([(entity_b.bbox[0] + entity_b.bbox[0]) / 2, entity_b.bbox[3]])
                                feet_b = self.cameras[vj].homo_inv @ np.array([feet_b[0], feet_b[1], 1])
                            feet_b = feet_b[:-1] / feet_b[-1]
                
                            aff_homo_temp[a, b] = 1 - np.linalg.norm(feet_b - feet_a)/ self.thresh_homo
                            aff_temp     [a, b] = np.sum(aff * entity_a.kpts[:, -1] * entity_b.kpts[:, -1]) / \
                                            (np.sum(valid_kp * entity_a.kpts[:, -1] * entity_b.kpts[:, -1]) + eps)
                            
                            reid_sim_temp[a, b] = (entity_a.reid @ entity_b.reid)
                    
                    aff_epi [det_all_count[vi] : det_all_count[vi+1], det_all_count[vj] : det_all_count[vj+1]] = aff_temp
                    aff_homo[det_all_count[vi] : det_all_count[vi+1], det_all_count[vj] : det_all_count[vj+1]] = aff_homo_temp

        aff_ = 2 * aff_epi + aff_homo
        aff_[aff_ < -1000] = -np.inf
        
        t1 = time()
        clusters, sol_matrix = self.solver.solve(aff_, True)
        t2 = time()
        print(f'\n\nFound {len(clusters)} clusters in {t2-t1:.2f} seconds')

        for cluster in tqdm(clusters):

            if len(cluster) == 1:
               view_list, number_list = find_view_for_cluster(cluster, det_all_count)
               det = entity_list_mv[view_list[0]][number_list[0]]
               
               if det.bbox[-1] > 0.9 \
               and all(det.kpts[self.main_joints, -1] > 0.5) \
               and np.sum(iou_det_mv[view_list[0]][number_list[0]] > 0.15) < 1 \
               and np.sum(ovr_det_mv[view_list[0]][number_list[0]] > 0.30) < 2:
                   new_track = PoseTrack(self.cameras)
                   new_track.single_view_init(det, id=len(self.tracks)+1)
                   self.match_with_miss_tracks(new_track, miss_tracks)

            else:
                view_list, \
                number_list = find_view_for_cluster(cluster, det_all_count)
                entity_list = [entity_list_mv[view_list[idx]][number_list[idx]] for idx in range(len(view_list))]

                for i, entity in enumerate(entity_list):
                    if entity.bbox[-1] > self.thresh_bbox \
                    and all(entity.kpts[self.main_joints, -1] > 0.5) \
                    and np.sum(iou_det_mv[view_list[i]][number_list[i]] > 0.15) < 1 \
                    and np.sum(ovr_det_mv[view_list[i]][number_list[i]] > 0.30) < 2:
                        new_track = PoseTrack(self.cameras)
                        for j in range(len(view_list)):
                            new_track.iou_mv    [view_list[j]] = iou_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_mv    [view_list[j]] = ovr_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_tgt_mv[view_list[j]] = ovr_tgt_mv[view_list[j]][number_list[j]]
                        new_track.multi_view_init(entity_list, id=len(self.tracks)+1)
                        self.match_with_miss_tracks(new_track, miss_tracks)
                        break
    
    def xyah_to_ltrb(self, ret):
        ret[...,  2] *= ret[...,  3]
        ret[..., :2] -= ret[..., 2:] / 2
        ret[..., 2:] += ret[..., :2]
        return ret

    @timeout(s=time_runs_out)
    def update_mv(self, entity_list_mv, frame_id=None, pbar=None, desc=""):

        um_iou_det_mv = []
        um_ovr_det_mv = []
        um_ovr_tgt_mv = []

        a_epi = 1
        a_box = 5
        a_homo = 1
        a_reid = 5
        
        # vide the valid view list
        for track in self.tracks :
            track.valid_views = []

        # 1st step, matching with confirmed and unconfirmed tracks
        if pbar:
            pbar.set_description(f"{desc} - [Step 1] Matching")
        avail_tracks = [track for track in self.tracks if track.state < TrackState.Missing]
        avail_idx = np.array([track.id for track in avail_tracks])

        if pbar:
            pbar.set_description(f"{desc} - [Step 2] Computing ReID aff")
        aff_reid, reid_weight = self.compute_reid_aff(entity_list_mv, avail_tracks)

        if pbar:
            pbar.set_description(f"{desc} - [Step 3] Computing Epi-homo aff")
        aff_epi, aff_homo = self.compute_epi_homo_aff(entity_list_mv, avail_tracks)

        if pbar:
            pbar.set_description(f"{desc} - [Step 4] Computing bbox-IoU aff")
        aff_box, iou_mv, \
        ovr_det_mv, ovr_tgt_mv = self.compute_bbox_iou_aff(entity_list_mv , avail_tracks)

        updated_tracks = set()
        unmatched_det = list()
        match_result = list()

        for v in range(self.num_cam):

            if pbar:
                pbar.set_description(f"{desc} - [Step 5] Matching view {v}")

            iou_sv     =     iou_mv[v]
            ovr_det_sv = ovr_det_mv[v]
            ovr_tgt_sv = ovr_tgt_mv[v]

            matched_det_sv = set()            

            lim = -a_box + 0.5 * a_box # upper limit
            aff_epi [v][ aff_epi[v] < lim] = lim
            aff_homo[v][aff_homo[v] < lim] = lim

            norm = a_epi  * (aff_epi [v] != 0).astype(float) + \
                   a_box  * (aff_box [v] != 0).astype(float) + \
                   a_homo * (aff_homo[v] != 0).astype(float)

            aff_ = (a_epi  * aff_epi [v] + \
                    a_box  * aff_box [v] + \
                    a_homo * aff_homo[v] + \
                    a_reid * aff_reid[v] * reid_weight[v]) / (1 + reid_weight[v])

            idx = np.where(norm > 0)
            aff_[idx] -= (a_box - norm[idx]) * 0.1
            aff_[aff_ < 0] = 0

            entity_list_sv = entity_list_mv[v]

            row_idxs, col_idxs = linear_sum_assignment(-aff_)
            match_result.append((row_idxs, col_idxs))

            if iou_sv.size:
                
                col_max     = iou_sv.max(0)
                col_max_arg = iou_sv.argmax(0)

                occlusion_row = set()
                for i in range(iou_sv.shape[1]):
                    if i not in col_idxs:
                        if col_max[i] > 0.5:
                            state = Track2DState.Occluded
                            occlusion_row.add(col_max_arg[i])
                        else:
                            state = Track2DState.Missing
                        if len(avail_tracks[i].track2ds[v].state) == 10:
                            avail_tracks[i].track2ds[v].state.pop()
                        avail_tracks[i].track2ds[v].state = [state] + avail_tracks[i].track2ds[v].state

            elif len(iou_sv) == 0:
                for i in range(iou_sv.shape[1]):
                    if len(avail_tracks[i].track2ds[v].state) == 10:
                        avail_tracks[i].track2ds[v].state.pop()
                    avail_tracks[i].track2ds[v].state = [Track2DState.Missing] + avail_tracks[i].track2ds[v].state

            for row, col in zip(row_idxs, col_idxs):
                
                # only update 2D info
                if row in occlusion_row:
                    state = Track2DState.Occluded
                    if len(avail_tracks[col].track2ds[v].state) == 10:
                        avail_tracks[col].track2ds[v].state.pop()
                    avail_tracks[col].track2ds[v].state = [state] + avail_tracks[col].track2ds[v].state
                
                if aff_[row,col] <= 0:
                    continue

                iou     =     iou_sv[row]
                ovr_det = ovr_det_sv[row]
                ovr_tgt = ovr_tgt_sv[row]

                # if True:
                avail_tracks[col].single_view_2D_update(v, entity_list_sv[row], iou, ovr_det, ovr_tgt, avail_idx)
                updated_tracks.add(col)
                matched_det_sv.add(row)
                    
            unmatched_det_sv = list(set(range(len(entity_list_sv))) - matched_det_sv)
            unmatched_sv = [entity_list_sv[u] for u in unmatched_det_sv]
            unmatched_det.append(unmatched_sv)

            unmatched_iou_sv = iou_sv[unmatched_det_sv]
            um_iou_det_mv.append(unmatched_iou_sv)

            unmatched_ovr_det_sv = ovr_det_sv[unmatched_det_sv]
            um_ovr_det_mv.append(unmatched_ovr_det_sv)

            unmatched_ovr_tgt_sv = ovr_tgt_sv[unmatched_det_sv]
            um_ovr_tgt_mv.append(unmatched_ovr_tgt_sv)
    
        if pbar:
            pbar.set_description(f"{desc} - [Step 6] Update track")

        for t_id in updated_tracks:
            corr_v = avail_tracks[t_id].multi_view_3D_update(avail_tracks)

        for t_id, track in enumerate(avail_tracks):
            track.valid_views = [v for v in range(self.num_cam) if track.age_bbox[v] == 0]
            if track.state == TrackState.Unconfirmed and (t_id not in updated_tracks):
                track.state = TrackState.Deleted
            if track.update_age >= 15:
                track.state = TrackState.Missing
            if track.state == TrackState.Confirmed:
                track.get_output()

        # perform association for unmatched detections and matching with missing tracks
        if pbar:
            pbar.set_description(f"{desc} - [Step 7] Association for unmatched and missed tracks")

        miss_tracks = [track for track in self.tracks if track.state == TrackState.Missing]
        if len(unmatched_det):
            if pbar:
                pbar.set_description(f"{desc} - [Step 7] Association - init target")
            self.target_init(unmatched_det, miss_tracks, um_iou_det_mv, um_ovr_det_mv, um_ovr_tgt_mv)

        feat_cnts = []
        for t, track in enumerate(self.tracks):

            if pbar:
                pbar.set_description(f"{desc} - [Step 7] Association - track {t} / {len(self.tracks)}")

            for v in range(self.num_cam):
                if track.age_bbox[v] >= 15:
                    track.bbox_filter[v] = BboxFilter()

            track.age_2D[track.age_2D >= 3] = np.inf
            track.age_3D[track.age_3D >= 3] = np.inf
            track.age_bbox[track.age_bbox >= 15] = np.inf
            track.dura_bbox[track.age_bbox >= 15] = 0

            track.age_2D += 1
            track.age_3D += 1
            track.age_bbox += 1
            track.update_age += 1
            if track.state  == TrackState.Confirmed:
                feat_cnts.append((track.id, track.feat_count))

    def output(self, frame_id):
        frame_results = []
        for track in self.tracks:
            if (track.state != TrackState.Confirmed) \
            or (track.update_age != 1):
                continue
            for v in track.valid_views:
                bbox = track.bbox_mv[v]
                record = np.array([[self.cameras[v].idx, 
                                    track.id, 
                                    frame_id, 
                                    bbox[0], 
                                    bbox[1], 
                                    bbox[2] - bbox[0], 
                                    bbox[3] - bbox[1], 
                                    track.output_coord[0], 
                                    track.output_coord[1], 
                                    track.output_coord[2]]])
                frame_results.append(record)
        return frame_results
    
            
