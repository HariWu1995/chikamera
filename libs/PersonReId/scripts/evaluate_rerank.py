import time

import torch
import numpy as np
import scipy.io

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from re_ranking import re_ranking


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index
    n_good = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(n_good):
        d_recall = 1.0 / n_good
        precision = (i+1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def evaluate(score, ql, qc, gl, gc):
    index = np.argsort(score)  # from small to large
    # index = index[::-1]
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


if __name__ == "__main__":

    # Unit-test
    model_name = "ResNet50"
    data_dir = f"F:/__Datasets__/Market1501/preprocessed/results_{model_name}"

    result_path = f'{data_dir}/result.mat'
    multi_path = f'{data_dir}/multi_query.mat'

    result = scipy.io.loadmat(result_path)

    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    query_feature = result['query_feat']

    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
    gallery_feature = result['gallery_feat']

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    # re-ranking
    print('\n\nCalculating initial distance ...')
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))

    since = time.time()
    print('\n\nRe-ranking ...')
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    time_elapsed = time.time() - since
    print('\t ... completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('\n\nEvaluating ...')
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(re_rank[i,:], query_label[i], query_cam[i],
                                                gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)
    mAP =  ap / len(query_label)
    print('Rank@1: %f Rank@5: %f Rank@10: %f mAP: %f' % (CMC[0], CMC[4], CMC[9], mAP))
