

from states import TrackState, Track2DState
from Tracker.kalman_filter import KalmanFilterBbox


class PoseTrack2D:
    
    def __init__(self):
        self.state = []
        self.cam_id = None
        self.bbox = None
        self.kpts = None
        self.reid = None

    def init_with_det(self, entity):
        if len(self.state) == 10:
            self.state.pop()
        self.state = [Track2DState.Detected] + self.state
        self.cam_id = entity.cam_id
        self.bbox = entity.bbox
        self.kpts = entity.kpts
        self.reid = entity.reid


class PoseTrack:

    def __init__(self, cameras):
        self.cameras = cameras
        self.num_cam = len(cameras)

        self.time_left = 0
        self.valid_views = [] # valid view input at current time step
        
        self.bank_size = 100
        self.feat_bank = np.zeros((self.bank_size, 2048))
        self.feat_count = 0

        self.track2ds = [PoseTrack2D() for i in range(self.num_cam)]
        self.bbox_filters = [KalmanFilterBbox() for i in range(self.num_cam)]

        self.entities = []
        self.output_coord = np.zeros(3)

        self.num_keypoints = 17
        self.kpts_thresh = 0.7
        self.kpts_3d = np.zeros((self.num_keypoints, 4))
        self.kpts_mv = np.zeros((self.num_cam, self.num_keypoints, 3))

        self.update_age = 0
        self.decay_weight = 0.5
        self.thresh_reid = 0.5
        self.thresh_reid_conf = 0.95
        self.unit = np.full((self.num_keypoints, 3), 1. / np.sqrt(3))

        self.bbox_mv = np.zeros((self.num_cam, 5))
        self.age_2D = np.ones((self.num_cam, self.num_keypoints)) * np.inf
        self.age_3D = np.ones((self.num_keypoints)) * np.inf
        self.age_bbox = np.ones((self.num_cam)) * np.inf
        self.dura_bbox = np.zeros((self.num_cam))

        self.output_priority = [[5, 6], [11, 12], [15, 16]]
        self.main_joints = np.array([5, 6, 11, 12, 13, 14, 15, 16])
        self.upper_body = np.array([5, 6, 11, 12])
        self.feet_idx = np.array([15, 16])

        self.iou_mv = [0 for i in range(self.num_cam)]
        self.ovr_mv = [0 for i in range(self.num_cam)]
        self.ovr_tgt_mv = [0 for i in range(self.num_cam)]
        self.oc_idx = [[] for i in range(self.num_cam)]
        self.oc_state = [False for i in range(self.num_cam)]

    def reactivate(self, newtrack):

        self.update_age = 0
        for attr in ['state','valid_views','track2ds','age_2D','age_3D','age_bbox','bbox_filters',
                    'bbox_mv','time_left','kpts_3d','kpts_mv','output_coord','dura_bbox']:
            setattr(self, attr, getattr(newtrack, attr))

        if newtrack.feat_count >= 1:
            if self.feat_count >= self.bank_size:
                bank = self.feat_bank
            else:
                bank = self.feat_bank[:self.feat_count % self.bank_size]
            new_bank = newtrack.feat_bank[:newtrack.feat_count % self.bank_size]

            sim = np.max((new_bank @ bank.T), axis=-1)
            sim_idx = np.where(sim < self.thresh_reid)[0]
            for id in sim_idx:
                self.feat_bank[self.feat_count % self.bank_size] = new_bank[id].copy()
                self.feat_count+=1

    def switch_view(self, track, v):
        print(f"Switching {track.id} 🡘 {self.id} with view {v} ...")
        self.track2ds[v],       track.track2ds[v]       = track.track2ds[v],        self.track2ds[v]
        self.age_2D[v],         track.age_2D[v]         = track.age_2D[v],          self.age_2D[v]
        self.kpts_mv[v],        track.kpts_mv[v]        = track.kpts_mv[v],         self.kpts_mv[v]
        self.age_bbox[v],       track.age_bbox[v]       = track.age_bbox[v],        self.age_bbox[v]
        self.bbox_mv[v],        track.bbox_mv[v]        = track.bbox_mv[v],         self.bbox_mv[v]
        self.dura_bbox[v],      track.dura_bbox[v]      = track.dura_bbox[v],       self.dura_bbox[v]
        self.oc_state[v],       track.oc_state[v]       = track.oc_state[v],        self.oc_state[v]
        self.oc_idx[v],         track.oc_idx[v]         = track.oc_idx[v],          self.oc_idx[v]
        self.bbox_filters[v],   track.bbox_filters[v]   = track.bbox_filters[v],    self.bbox_filters[v]
        self.iou_mv[v],         track.iou_mv[v]         = track.iou_mv[v],          self.iou_mv[v]
        self.ovr_mv[v],         track.ovr_mv[v]         = track.ovr_mv[v],          self.ovr_mv[v]
        self.ovr_tgt_mv[v],     track.ovr_tgt_mv[v]     = track.ovr_tgt_mv[v],      self.ovr_tgt_mv[v]

    def get_output(self):

        # 3D keypoints output
        for comb in self.output_priority:
            if all(self.age_3D[comb] == 0):
                self.output_coord = np.concatenate((np.mean(self.kpts_3d[comb, :2], axis=0), [3]))
                return self.output_coord 
        
        # if no 3D keypoints combination, choose single-view feet 
        feet_idxs = self.output_priority[-1]
        for v in self.valid_views:
            if all(self.kpts_mv[v][feet_idxs, -1] > 0.7) \
            and all(self.age_2D[v][feet_idxs] == 0):
                feet_pos = np.mean(self.kpts_mv[v][feet_idxs, :2], axis=0)
                feet_homo = self.cameras[v].homo_feet_inv @ np.array([feet_pos[0], feet_pos[1], 1])
                feet_homo = feet_homo[:-1] / feet_homo[-1]
                self.output_coord = np.concatenate((feet_homo,[2]))
                return self.output_coord

        # if no single-view feet, then choose bbox bottom point
        bottom_points = []
        for v in self.valid_views:
            bbox = self.bbox_mv[v]
            bp = self.cameras[v].homo_inv @ np.array([(bbox[2] + bbox[0]) / 2, bbox[3], 1])
            bp = bp[:-1] / bp[-1]
            if bbox[3] > 1078:
                bottom_points.append(bp)
                continue
            self.output_coord = np.concatenate((bp, [1]))            
            return self.output_coord
            
        # others
        bottom_points = np.array(bottom_points).reshape(-1,2)
        self.output_coord = np.concatenate((np.mean(bottom_points, axis=0), [1]))
        return self.output_coord

    def single_view_init(self, entity, id):
        # if initilized only with 1 view 
        self.state = TrackState.Unconfirmed
        self.time_left = 2

        cam_id = entity.cam_id
        self.valid_views.append(cam_id)

        track2d = self.track2ds[cam_id]
        track2d.init_with_det(entity)

        self.kpts_mv[cam_id] = entity.kpts
        self.bbox_mv[cam_id] = entity.bbox
        self.age_bbox[cam_id] = 0
        self.dura_bbox[cam_id] = 1

        self.bbox_filters[cam_id].update(entity.bbox[:4].copy())

        self.feat_bank[0] = track2d.reid
        self.feat_count +=1

        self.id = id
        self.update_age = 0
    
    def multi_view_init(self, entity_list, id):

        self.state = TrackState.Confirmed
        self.kpts_3d, self.kpts_mv, \
        self.age_3D, self.age_2D = self.triangulation(entity_list)

        for entity in entity_list:
            cam_id = entity.cam_id
            self.valid_views.append(cam_id)

            track2d = self.track2ds[cam_id]
            track2d.init_with_det(entity)

            self.bbox_mv[cam_id] = entity.bbox
            self.bbox_filters[cam_id].update(entity.bbox[:4].copy())

            if all(entity.kpts[self.upper_body, -1] > 0.5) \
               and entity.bbox[4] > 0.9 \
               and np.sum(self.iou_mv[cam_id] > 0.15) < 1 \
               and np.sum(self.ovr_mv[cam_id] > 0.30) < 2:
                if self.feat_count:
                    bank = self.feat_bank[:self.feat_count]
                    sim = bank @ entity.reid
                    if np.max(sim) < (self.thresh_reid + 0.1):
                        self.feat_bank[self.feat_count % self.bank_size] = entity.reid
                        self.feat_count += 1
                else:
                    self.feat_bank[0] = track2d.reid
                    self.feat_count +=1

            self.age_bbox[cam_id] = 0
            self.dura_bbox[cam_id] = 1

        self.update_age = 0
        self.id = id
        self.iou_mv = [0 for i in range(self.num_cam)]

        self.valid_views = sorted(self.valid_views)
        self.get_output()

    def triangulation(self, entity_list):
        kpts_mv = np.zeros((self.num_cam, self.num_keypoints, 3))
        kpts_3d = np.zeros((self.num_keypoints, 4))

        age_2D = np.ones((self.num_cam, self.num_keypoints)) * np.inf
        age_3D = np.ones((self.num_keypoints)) * np.inf

        for entity in entity_list:
            kpts_mv[entity.cam_id] = entity.kpts

        valid_joint_mask = (kpts_mv[:,:,2] > self.kpts_thresh)

        for j_idx in range(self.num_keypoints):
            if np.sum(valid_joint_mask[:, j_idx]) < 2:
                joint_3d = np.zeros(4)
            else:
                A = np.zeros((2 * self.num_keypoints, 4))
                for v_idx in range(self.num_cam):
                    A[2 * v_idx  ,:] = kpts_mv[v_idx, j_idx, 2] * (kpts_mv[v_idx, j_idx, 0] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[0,:])
                    A[2 * v_idx+1,:] = kpts_mv[v_idx, j_idx, 2] * (kpts_mv[v_idx, j_idx, 1] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[1,:])

                u, sigma, vt = np.linalg.svd(A)
                joint_3d = vt[-1] / vt[-1][-1]
                age_3D[j_idx] = 0
    
            kpts_3d[j_idx] = joint_3d
            age_2D[valid_joint_mask[:, j_idx]] = 0

        return kpts_3d, kpts_mv, age_3D, age_2D
    
    def single_view_2D_update(self, v, entity,iou, ovr, ovr_tgt, avail_idx):
        
        if np.sum(    iou > 0.5) >= 2 \
        or np.sum(ovr_tgt > 0.5) >= 2:
            oc_idx = avail_idx[np.where((iou > 0.5) | (ovr_tgt > 0.5))]
            self.oc_idx[v] = list(set(self.oc_idx[v] + [i for i in oc_idx if i != self.id]))
            self.oc_state[v] = True
            
        valid_joints = entity.kpts[:, -1] > self.kpts_thresh
        self.kpts_mv[v][valid_joints] = entity.kpts[valid_joints]
        self.age_2D[v][valid_joints] = 0

        self.bbox_mv[v] = entity.bbox
        self.age_bbox[v] = 0
        self.track2ds[v].init_with_det(entity)
            
        self.bbox_filters[v].update(entity.bbox[:4].copy())
        
        self.ovr_tgt_mv[v] = ovr_tgt
        self.dura_bbox[v] +=1
        self.iou_mv[v] = iou
        self.ovr_mv[v] = ovr

    def multi_view_3D_update(self, avail_tracks):
        valid_views = [v for v in range(self.num_cam) if (self.age_bbox[v] == 0)]
        if self.feat_count >= self.bank_size:
            bank = self.feat_bank
        else:
            bank = self.feat_bank[:self.feat_count % self.bank_size]

        for v in valid_views:
            if self.oc_state[v] \
            and self.bbox_mv[v][-1] > 0.9 \
            and np.sum(    self.iou_mv[v] > 0.15) < 2 \
            and np.sum(self.ovr_tgt_mv[v] > 0.30) < 2:
                if self.feat_count == 0:
                    self.oc_state[v] = False
                    self.oc_idx[v] = []
                    continue
                
                self.oc_state[v] = False
                oc_tracks = []

                print("Leaving OC", self.id, v, self.iou_mv[v], self.ovr_tgt_mv[v],self.oc_idx[v])                

                self_sim = np.max((self.track2ds[v].reid @ bank.T))
                print("self_sim:", self_sim)

                if self_sim > 0.5:
                    self.oc_idx[v] = []
                    continue
        
                for t_id, track in enumerate(avail_tracks):
                    if track.id in self.oc_idx[v]:
                        oc_tracks.append(track)
        
                if len(oc_tracks) == 0:
                    self.oc_idx[v] = []
                    print("Miss oc track")
                    continue

                reid_sim = np.zeros(len(oc_tracks))
                for t_id, track in enumerate(oc_tracks):
                    if track.feat_count == 0:
                        continue
                    if track.feat_count >= track.bank_size:
                        oc_bank = track.feat_bank
                    else:
                        oc_bank = track.feat_bank[:track.feat_count % track.bank_size]
                    reid_sim[t_id] = np.max(self.track2ds[v].reid @ oc_bank.T)
                print("reid_sim:", reid_sim)

                max_idx = np.argmax(reid_sim)
                self.oc_idx[v] = []
                if  reid_sim[max_idx] > self_sim \
                and reid_sim[max_idx] > 0.5:
                    self.switch_view(oc_tracks[max_idx], v)
                    
        valid_joint_mask = (self.kpts_mv[:,:,2] > self.kpts_thresh) & (self.age_2D == 0)
        corr_v = []

        for j_idx in range(self.num_keypoints):
            if np.sum(valid_joint_mask[:,j_idx]) < 2:
                joint_3d = np.zeros(4)
                continue
            else:
                A = np.zeros((2 * self.num_keypoints, 4))
                for v_idx in range(self.num_cam):
                    if valid_joint_mask[v_idx, j_idx]:
                        A[2 * v_idx    , :] = self.kpts_mv[v_idx, j_idx, 2] * (self.kpts_mv[v_idx, j_idx, 0] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[0,:])
                        A[2 * v_idx + 1, :] = self.kpts_mv[v_idx, j_idx, 2] * (self.kpts_mv[v_idx, j_idx, 1] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[1,:])

                u, sigma, vt = np.linalg.svd(A)
                joint_3d = vt[-1] / vt[-1][-1]

                # false matching correction
                if (joint_3d[2] < -1 \
                or  joint_3d[2] > 2.5) \
                or (j_idx in self.feet_idx and (joint_3d[2] < -1 or joint_3d[2] > 1)):
                    if np.min(self.dura_bbox[self.age_bbox == 0]) >= 10:
                        continue

                    # views to be corrected are often new entering people with the minimum bbox tracking duration
                    v_cand = [v for v in range(self.num_cam) 
                                if (self.dura_bbox[v] == np.min(self.dura_bbox[self.age_bbox == 0]))]
                        
                    for v in  v_cand:
                        if valid_joint_mask[v,j_idx]:                            
                            self.age_bbox[v] = np.inf
                            self.dura_bbox[v] = 0
                            self.kpts_mv[v] = 0
                            self.age_2D[v] = np.inf
                            valid_joint_mask[v] = 0
                            corr_v.append(v)
                            break

                self.age_3D[j_idx] = np.min(self.age_2D[valid_joint_mask[:, j_idx], j_idx])

            self.kpts_3d[j_idx] = joint_3d

        valid_views = [v for v in range(self.num_cam) 
                          if (self.age_bbox[v] == 0 and (not v in corr_v))]
        self.update_age = 0

        for v in valid_views:
            if self.feat_count >= self.bank_size:
                bank = self.feat_bank
            else:
                bank = self.feat_bank[:self.feat_count]
            
            entity = self.track2ds[v]
            if all(entity.kpts[self.upper_body,-1] > 0.5) \
            and entity.bbox[4] > 0.9 \
            and np.sum(self.iou_mv[v] > 0.15) < 2 \
            and np.sum(self.ovr_mv[v] > 0.30) < 2:
                if self.feat_count == 0:
                    self.feat_bank[0] = entity.reid
                    self.feat_count += 1
                else:
                    sim = bank @ entity.reid
                    if np.max(sim) < (self.thresh_reid + 0.1):
                        self.feat_bank[self.feat_count%self.bank_size] = entity.reid
                        self.feat_count += 1

        if self.state == TrackState.Unconfirmed:
            if any(self.bbox_mv[self.age_bbox==0][:, -1] > 0.9):
                self.time_left -= 1
                if self.time_left <= 0:
                    self.state = TrackState.Confirmed
        
        self.iou_mv = [0 for i in range(self.num_cam)]
        self.ovr_mv = [0 for i in range(self.num_cam)]
        return corr_v

    def calculate_target_rays(self, v):
        if self.age_bbox[v] > 1:
            return self.unit
        cam = self.cameras[v]
        return aic_cpp.compute_joints_rays(self.kpts_mv[v], cam.project_inv, cam.pos)

