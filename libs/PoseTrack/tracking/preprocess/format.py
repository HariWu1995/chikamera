from copy import deepcopy


det_formats = [
    '%d','%d',                  # frame id, class id
    '%d','%d','%d','%d','%.3e', # 4-point bbox & score
]

kpt_formats = deepcopy(det_formats[1:])
for k in range(133):
    kpt_formats.extend(['%d','%d','%.3e'])  # keypoint (x, y, score)

reid_formats = deepcopy(det_formats[1:])
reid_formats.extend(['%.10e'] * 2048)       # re-identification features


