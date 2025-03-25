import os
import argparse

import torch
from torchvision import datasets

import numpy as np
import scipy.io

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


def imshow(path, title=None):
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def ranking(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)[::-1]  # from small to large
    # index = index[0:2000]
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc) # same camera

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


if __name__ == "__main__":

    # Unit-test
    model_name = "ResNet50"
    data_dir = f"F:/__Datasets__/DukeMTMC/preprocessed"

    # Options
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--query_index', default=[123, 456, 789], nargs='+', help='test_image_index')
    parser.add_argument('--data_dir', default=data_dir,type=str, help='./test_data')
    parser.add_argument('--model_name', default=model_name,type=str, help='./test_data')
    opt = parser.parse_args()

    data_dir = opt.data_dir
    result_dir = f"{opt.data_dir}/results_{opt.model_name}"

    result_path = f'{result_dir}/result.mat'
    multi_path = f'{result_dir}/multi_query.mat'

    image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x)) 
        for x in ['gallery','query']
    }

    result = scipy.io.loadmat(result_path)

    query_feature = torch.FloatTensor(result['query_feat'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]

    gallery_feature = torch.FloatTensor(result['gallery_feat'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    multi = os.path.isfile(multi_path)
    if multi:
        m_result = scipy.io.loadmat(multi_path)
        mquery_feature = torch.FloatTensor(m_result['mquery_feat'])
        mquery_cam = m_result['mquery_cam'][0]
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()

    for i in opt.query_index:

        rank_idx = ranking(query_feature[i], query_label[i], query_cam[i],
                        gallery_feature, gallery_label, gallery_cam)

        # Visualize the rank result
        q_path, _ = image_datasets['query'].imgs[i]
        q_label = query_label[i]
        print('\n\n', q_path)
        print('Top 10 images are as follow:')

        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(q_path, 'query')
        for j in range(10):
            ax = plt.subplot(1, 11, j+2)
            ax.axis('off')
            img_path, _ = image_datasets['gallery'].imgs[rank_idx[j]]
            label = gallery_label[rank_idx[j]]
            imshow(img_path)
            if label == q_label:
                ax.set_title('%d' % (j+1), color='green')
            else:
                ax.set_title('%d' % (j+1), color='red')
            print(img_path)

        fig.savefig(f"{result_dir}/ranking_sample_{i}.png")
