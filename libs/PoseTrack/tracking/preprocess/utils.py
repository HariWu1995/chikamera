import numpy as np
import pandas as pd


fid_columns = ['frame_id', 'x1', 'y1', 'x2', 'y2']
det_columns = ['frame_id', 'class_id', 'x1', 'y1', 'x2', 'y2', 'bbox_score']

kpt_columns = fid_columns + ['bbox_score']
for i in range(133):
    kpt_columns.extend([f'kpt_{i}_x', f'kpt_{i}_y', f'kpt_{i}_score'])

reid_columns = fid_columns + ['bbox_score'] + [f'feat_{i}' for i in range(2048)]


def reconcile(det_data, kpt_data, reid_data):

    det_df = pd.DataFrame.from_records(det_data)
    kpt_df = pd.DataFrame.from_records(kpt_data)
    reid_df = pd.DataFrame.from_records(reid_data)

    det_df.columns = det_columns
    kpt_df.columns = kpt_columns
    reid_df.columns = reid_columns

    kpt_df = kpt_df.drop(columns=['bbox_score'])
    reid_df = reid_df.drop(columns=['bbox_score'])

    id_dtypes = {k: 'int32' for k in fid_columns}
    det_df = det_df.astype(id_dtypes)
    kpt_df = kpt_df.astype(id_dtypes)
    reid_df = reid_df.astype(id_dtypes)

    all_df = det_df.merge(kpt_df, how='left', on=fid_columns)
    all_df = all_df.merge(reid_df, how='left', on=fid_columns)
    pre_cnt = len(all_df)

    all_df = all_df.dropna(axis=0, how='any')
    post_cnt = len(all_df)

    print(f"\n\nThere are {post_cnt-pre_cnt} missing records ({pre_cnt} -> {post_cnt})")
    
    return (
        all_df[det_columns].values,
        all_df[kpt_columns].values,
        all_df[reid_columns].values,        
    )


def load_preprocessed_file(file, delimiter: str = ',', 
                                 use_numpy: bool = False, 
                                use_memmap: bool = False, 
                                apply_norm: bool = False):
    if use_numpy:
        data = np.load(file, mmap_mode = 'r+' if use_memmap else None)
    else:
        data = np.loadtxt(file, delimiter = delimiter)
    if apply_norm:
        data[:, -2048:] = data[:, -2048:] / \
            np.linalg.norm(data[:, -2048:], axis=1, keepdims=True)
    return data


