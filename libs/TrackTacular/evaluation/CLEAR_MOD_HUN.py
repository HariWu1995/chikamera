import math
import numpy as np

from scipy.optimize import linear_sum_assignment


def distance(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


def CLEAR_MOD_HUN(gt, det):
    """
    @param 
        gt: the groundtruth result matrix
        det: the detection result matrix

    @return: 
        MODA, MODP, recall, precision

    compute CLEAR Detection metrics 
    according to PERFORMANCE EVALUATION PROTOCOL 
             for FACE, PERSON and VEHICLE DETECTION and TRACKING 
              in VIDEO ANALYSIS and CONTENT EXTRACTION (VACE-II)
    
    CLEAR = CLASSIFICATION of EVENTS, ACTIVITIES and RELATIONSHIPS
    
    Submitted to Advanced Research and Development Activity

    metrics contains the following
    [1]   recall	- recall    = percentage of           detected targets
    [2]   precision	- precision = percentage of correctly detected targets
    [3]	    MODA    - N-MODA
    [4]	    MODP    - N-MODP
    """
    td = 50 / 2.5  # distance threshold

    F = int(max( gt[:, 0])) + 1
    N = int(max(det[:, 1])) + 1
    Fgt = int(max(gt[:, 0])) + 1
    Ngt = int(max(gt[:, 1])) + 1

    M = np.zeros((F, Ngt))

    c  = np.zeros((1, F))
    fp = np.zeros((1, F))
    m  = np.zeros((1, F))
    g  = np.zeros((1, F))

    d = np.zeros((F, Ngt))
    distances = np.inf * np.ones((F, Ngt))

    for t in range(1, F + 1):
        GTsInFrames  = np.where( gt[:, 0] == t - 1)
        DetsInFrames = np.where(det[:, 0] == t - 1)
        GTsInFrame  =  GTsInFrames[0]
        DetsInFrame = DetsInFrames[0]
        GTsInFrame  = np.reshape( GTsInFrame, (1,  GTsInFrame.shape[0]))
        DetsInFrame = np.reshape(DetsInFrame, (1, DetsInFrame.shape[0]))

        Ngtt = GTsInFrame.shape[1]
        Nt  = DetsInFrame.shape[1]
        g[0, t-1] = Ngtt

        if (DetsInFrame is not None) \
        and (GTsInFrame is not None):

            dist = np.inf * np.ones((Ngtt, Nt))
            for o in range(1, Ngtt+1):
                GT = gt[GTsInFrame[0][o-1]][2:4]
                for e in range(1, Nt+1):
                    E = det[DetsInFrame[0][e-1]][2:4]
                    dist[o-1, e-1] = distance(GT[0], GT[1], E[0], E[1])

            tmpai = dist
            tmpai = np.array(tmpai)

            # NOTE: 
            #   price / distance are set to 100_000 (1e6) instead of np.inf, 
            #   since the Hungarian Algorithm implemented in sklearn will 
            #   suffer from long calculation time if we use np.inf.
            tmpai[tmpai > td] = 1e6
            if not tmpai.all() == 1e6:
                HUN_res = np.array(linear_sum_assignment(tmpai)).T
                HUN_res = HUN_res[tmpai[HUN_res[:, 0], 
                                        HUN_res[:, 1]] < td]
                u, v = HUN_res[HUN_res[:, 1].argsort()].T
                for mmm in range(1, len(u)+1):
                    M[t-1, u[mmm-1]] = v[mmm-1] + 1

        curdetected, = np.where(M[t-1, :])
        c[0][t-1] = curdetected.shape[0]

        for ct in curdetected:
            eid = M[t-1, ct] - 1

            gtX = gt[GTsInFrame[0][ct], 2]
            gtY = gt[GTsInFrame[0][ct], 3]

            stX = det[DetsInFrame[0][int(eid)], 2]
            stY = det[DetsInFrame[0][int(eid)], 3]

            distances[t-1, ct] = distance(gtX, gtY, stX, stY)

        fp[0][t-1] =    Nt    - c[0][t-1]
        m[0][t-1] = g[0][t-1] - c[0][t-1]

    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100

    R = np.sum(c) / np.sum(g) * 100
    P = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100

    MODP = max(MODP, 0)
    MODA = max(MODA, 0)
    R = max(R, 0)
    P = max(P, 0)

    return R, P, MODA, MODP
