import numpy as np

import lap

import scipy
from scipy.spatial.distance import cdist

from . import kalman_filter


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))

    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)
    matches = indices[matched_mask]

    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
                np.empty((0, 2), dtype=int), 
            tuple(range(cost_matrix.shape[0])), 
            tuple(range(cost_matrix.shape[1])),
        )
   
    matches, unmatched_a, unmatched_b = [], [], []

    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    matches = np.asarray(matches)

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU

    Arguments:
        atlbrs: list[tlbr] | np.ndarray
        atlbrs: list[tlbr] | np.ndarray

    Return:
        ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    # ious = bbox_ious(
    #     np.ascontiguousarray(atlbrs, dtype=np.float32),
    #     np.ascontiguousarray(btlbrs, dtype=np.float32)
    # )
    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU

    Arguments:
        atracks: list[STrack]
        btracks: list[STrack]

    Return:
        cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
    or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


def center_distance(atracks, btracks):
    """
    Compute cost based on center point distance

    Arguments:
        atracks: list[STrack]
        btracks: list[STrack]

    Return:
        cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    atracks = np.stack(atracks)
    btracks = np.stack(btracks)

    cost_matrix = cdist(atracks, btracks, 'euclidean')
    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    Arguments:
            tracks: list[STrack]
        detections: list[BaseTrack]
            metric:

    Return:
        cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features   = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    
    cost_matrix = cdist(track_features, det_features, metric)
    cost_matrix = np.maximum(0.0, cost_matrix)
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position: bool = False):
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]

    measurements = np.asarray([det.to_xyah() for det in detections])
    
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=True, 
                lambda_=0.98, gating_threshold=1000):
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    # gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # gating_threshold = 1000

    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance * 0.1
    return cost_matrix
