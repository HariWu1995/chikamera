import numpy as np


eps = 1e-5


def find_view(human_number, human_counts_all_views):
    if len(human_counts_all_views) == 1:
        print("no human in any view")
        return -1
    else:
        for i in range(len(human_counts_all_views)-1):
            if  human_number >  human_counts_all_views[i] - 1 \
            and human_number <= human_counts_all_views[i+1] - 1:
                return i, human_number - human_counts_all_views[i]
            else:
                continue
        raise Exception("human array searching out of range") 


def find_view_for_cluster(cluster, human_counts_all_views):
    view_list = []
    number_list = []
    for human_number in cluster:
        view, number = find_view(human_number, human_counts_all_views)
        view_list.append(view)
        number_list.append(number)
    return view_list, number_list


def calculate_rays_sv(kpts, cam):
    joints_h = np.vstack((kpts[:,:-1].T, np.ones((1,kpts.shape[0])))) # 3 * n
    joints_rays =  cam.project_inv @ joints_h
    joints_rays /= joints_rays[-1]
    joints_rays = joints_rays[:-1]
    joints_rays -= np.repeat(cam.pos.reshape(3,1), kpts.shape[0], axis=1)
    joints_rays_norm = joints_rays / (np.linalg.norm(joints_rays, axis=0) + eps)
    joints_rays_norm = joints_rays_norm.T
    return joints_rays_norm

            
