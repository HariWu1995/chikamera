import numpy as np

from .CLEAR_MOD_HUN import CLEAR_MOD_HUN
# from evaluation.CLEAR_MOD_HUN import CLEAR_MOD_HUN


def mod_metrics(res_fpath, gt_fpath):
    """
    This is python simple translation from a MATLAB Evaluation tool
        used to evaluate detection result created by P. Dollar.

    This API allows the project to run purely in Python without using MATLAB Engine.

    Some critical information to notice before you use this API:

        1. This API is only tested in MVDet project: https://github.com/hou-yz/MVDet, 
            and might be incompatible with other projects.

        2. The detection result using this API is a little lower 
            (~ 0-2% decrease in MODA, MODP) than MATLAB evaluation tool, 
            the reason might be that the Hungarian Algorithm implemented in
            sklearn.utils.linear_assignment_.linear_assignment is different 
            with the one implemented by P. Dollar. 
            Please use official MATLAB API if you want to obtain the same result in the paper. 
    
        3. The training process would not be affected by this API.

    @param 
        res_fpath: detection result file path
        gt_fpath: ground truth result file path

    @return: 
        recall, precision, MODA, MODP
    """

    gtRaw = np.loadtxt(gt_fpath)
    detRaw = np.loadtxt(res_fpath)

    frame_ctr = 0
    frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0

    if detRaw is None or detRaw.shape[0] == 0:
        MODP, MODA, recall, precision = 0, 0, 0, 0
        return MODP, MODA, recall, precision

    for t in frames:
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)

        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in gtRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in gtRaw[idx, 2]])

        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr), axis=0)

        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)

        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in detRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in detRaw[idx, 2]])

        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr), axis=0)
        frame_ctr += 1

    recall, precision, MODA, MODP = CLEAR_MOD_HUN(gtAllMatrix, detAllMatrix)
    return recall, precision, MODA, MODP
