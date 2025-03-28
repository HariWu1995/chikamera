import numpy as np


def format_openpose(candidates, scores, width, height):
    num_candidates, _, locs = candidates.shape

    candidates[..., 0] /= float(width)
    candidates[..., 1] /= float(height)

    bodies = candidates[:, :18].copy()
    bodies = bodies.reshape(num_candidates * 18, locs)

    body_scores = scores[:, :18]
    for i in range(len(body_scores)):
        for j in range(len(body_scores[i])):
            if body_scores[i][j] > 0.3:
                body_scores[i][j] = int(18 * i + j)
            else:
                body_scores[i][j] = -1

    faces = candidates[:, 24:92]
    faces_scores = scores[:, 24:92]

    hands = np.vstack([candidates[:, 92:113], candidates[:, 113:]])
    hands_scores = np.vstack([scores[:, 92:113], scores[:, 113:]])

    pose = dict(
        bodies=bodies,
        body_scores=body_scores,
        hands=hands,
        hands_scores=hands_scores,
        faces=faces,
        faces_scores=faces_scores,
    )

    return pose
