import os
from shutil import copyfile
from tqdm import tqdm


# You only need to change this line to your dataset download path
download_path = 'F:/__Datasets__/DukeMTMC'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/preprocessed'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#---------------------------------------
# query
print('\n\nPreparing query ...')
query_load_path = download_path + '/query'
query_save_path = download_path + '/preprocessed/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_load_path, topdown=True):
    for name in tqdm(files):
        if not name.endswith('.jpg'):
            continue
        ID  = name.split('_')
        src_path = query_load_path + '/' + name
        dst_path = query_save_path + '/' + ID[0] 
        if os.path.isdir(dst_path) is False:
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
# multi-query
# for dukemtmc-reid, we do not need multi-query
query_load_path = download_path + '/gt_bbox'
query_save_path = download_path + '/preprocessed/multi-query'

if os.path.isdir(query_load_path):
    print('\n\nPreparing multi-query ...')
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_load_path, topdown=True):
        for name in tqdm(files):
            if not name.endswith('.jpg'):
                continue
            ID  = name.split('_')
            src_path = query_load_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
# gallery
print('\n\nPreparing gallery ...')
gallery_load_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/preprocessed/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_load_path, topdown=True):
    for name in tqdm(files):
        if not name.endswith('.jpg'):
            continue
        ID  = name.split('_')
        src_path = gallery_load_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if os.path.isdir(dst_path) is False:
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
# train_all
print('\n\nPreparing train-all ...')
train_load_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/preprocessed/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_load_path, topdown=True):
    for name in tqdm(files):
        if not name.endswith('.jpg'):
            continue
        ID  = name.split('_')
        src_path = train_load_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if os.path.isdir(dst_path) is False:
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
# train_val
print('\n\nPreparing train-val ...')
train_load_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/preprocessed/train'
val_save_path   = download_path + '/preprocessed/val'
if os.path.isdir(train_save_path) is False:
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_load_path, topdown=True):
    for name in tqdm(files):
        if not name.endswith('.jpg'):
            continue
        ID  = name.split('_')
        src_path = train_load_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if os.path.isdir(dst_path) is False:
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
