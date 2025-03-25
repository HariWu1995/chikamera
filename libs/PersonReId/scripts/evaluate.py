import os
# import time
from tqdm import tqdm

import torch
import numpy as np
import scipy.io


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
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


def evaluate(qf, ql, qc, gf, gl, gc):
    score = np.dot(gf, qf)
    # predict index
    index = np.argsort(score)[::-1]  # from small to large
    # index = index[0:2000]
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

    print('\n\nEvaluating ...')
    result = scipy.io.loadmat(result_path)

    query_feature = result['query_feat']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]

    gallery_feature = result['gallery_feat']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
        
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.

    pbar = tqdm(range(len(query_label)))
    for i in pbar:
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
        pbar.set_description(f"AP = {ap_tmp:.5f}")
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)
    mAP =  ap / len(query_label)
    print('Rank@1: %f Rank@5: %f Rank@10: %f mAP: %f' % (CMC[0], CMC[4], CMC[9], mAP))

    # multiple-query
    multi = os.path.isfile(multi_path)
    if not multi:
        quit()

    print('\n\nEvaluating multi-query ...')
    m_result = scipy.io.loadmat(multi_path)
    mquery_feature = m_result['mquery_feat']
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.

    pbar = tqdm(range(len(query_label)))
    for i in pbar:
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
        mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        mq_feature = np.mean(mquery_feature[mquery_index, :], axis=0)
        ap_tmp, CMC_tmp = evaluate(mq_feature, query_label[i], query_cam[i],
                              gallery_feature, gallery_label, gallery_cam)
        pbar.set_description(f"AP = {ap_tmp:.5f}")
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)
    mAP =  ap / len(query_label)
    print('Rank@1: %f Rank@5: %f Rank@10: %f mAP: %f' % (CMC[0], CMC[4], CMC[9], mAP))
