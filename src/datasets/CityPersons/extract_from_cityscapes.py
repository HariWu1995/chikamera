import os
import shutil

from tqdm import tqdm

import random as rd
import numpy as np
import pandas as pd

import cv2
import scipy.io

import matplotlib.pyplot as plt


DATA_DIR = dict(
    cityscapes = 'F:/__Datasets__/CityScapes',
    citypersons = 'F:/__Datasets__/CityPersons',
)

CLASS_DICT = {
    0: 'ignore regions' ,
    1: 'pedestrians',
    2: 'riders',
    3: 'sitting persons',
    4: 'other persons unusual postures',
    5: 'group of people',
}

H = 1024
W = 2048


# Convert the Mat files to Pandas.DataFrame
print('\n\nConverting ...')

DataFrames = dict()
columns = [
    'city','image_id',                                  # metadata
    'class_label','x1','y1','w','h',                    # for detection
    'instance_id','x1_vis','y1_vis','w_vis','h_vis',    # for segmentation
]

for subset in ['train', 'val']:
    print('\n\t', subset)
    mat = scipy.io.loadmat(f"{DATA_DIR['citypersons']}/raw/anno_{subset}.mat")

    all_records = []
    for img_data in tqdm(mat[f'anno_{subset}_aligned'][0]):
        for instance in img_data[0][0][2]:
            all_records.append([img_data[0][0][0][0],
                                img_data[0][0][1][0], 
                                instance[0], instance[1], instance[2], instance[3], instance[4],
                                instance[5], instance[6], instance[7], instance[8], instance[9]])

    DataFrames[subset] = pd.DataFrame(all_records, columns=columns)
    print(DataFrames[subset]['class_label'].value_counts())


# Check if all the images are available
print('\n\nChecking ...')

for subset in ['train', 'val']:
    print('\n\t', subset)

    all_imgs = DataFrames[subset]['image_id'].unique()
    for img_path in tqdm(all_imgs):
        path = img_path.split('_')[0] + '/' + img_path
        if not os.path.exists(f"{DATA_DIR['cityscapes']}/leftImg8bit/train/{path}"):
            print(path)


# Remove class_label = `ignore regions`
print('\n\nCleaning ...')

for subset in ['train', 'val']:
    print('\n\t', subset)
    DataFrames[subset] = DataFrames[subset].drop(
                        DataFrames[subset][DataFrames[subset]['class_label']==0].index)
    DataFrames[subset]['class_label'] = DataFrames[subset]['class_label'].replace([2,3,4,5],[0,0,0,0])


# Visualize
print('\n\nVisualizing ...')

def draw(image, splitted_boxes):
    
    for box in splitted_boxes:
        c,x,y,w,h = box
        x_min = x
        x_max = x+w
        y_min = y
        y_max = y+h
        X, Y, _ = image.shape

        isClosed = True

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 5

        # Using cv2.polylines() method
        # Draw a Blue polygon with thickness of 1 px
        image = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0), 4)
        image = cv2.putText(image, CLASS_DICT[c], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0))

    return image

plt.figure(figsize=(20, 16), dpi=80)

subset = 'train'
i = rd.randint(0, 2500)
df = DataFrames[subset]

all_ids = df['image_id'].unique()
c_box = df[df['image_id']==all_ids[i]][['class_label','x1','y1','w','h']].values

img_path = f"{DATA_DIR['cityscapes']}/leftImg8bit/{subset}/{all_ids[i].split('_')[0]}/{all_ids[i]}"
img = cv2.imread(img_path)
print(img.shape)
plt.imshow(draw(img, c_box))
plt.show()


# Preprocess
print('\n\nPreprocessing ...')

def preprocess(df, src_dir, img_dir, label_dir):

    id_list = list(df['image_id'].unique())

    for idd in tqdm(id_list):
        if os.path.exists(f"{img_dir}/{idd}"):
            continue
        shutil.copy(f"{src_dir}/{idd.split('_')[0]}/{idd}", 
                    f"{img_dir}/{idd}")

    if label_dir is None:
        return 

    for idd in tqdm(id_list):

        id_label_path = f"{label_dir}/{idd.split('.')[0]}.txt"
        if os.path.exists(id_label_path):
            continue

        types = df[df['image_id']==idd]['class_label'].values
        cords = df[df['image_id']==idd][['x1','y1','w','h']].values
        
        cords_new = np.zeros(cords.shape)    
        cords_new[:,0] = (cords[:,0] + cords[:,2] / 2) / W
        cords_new[:,1] = (cords[:,1] + cords[:,3] / 2) / H
        cords_new[:,2] =               cords[:,2]      / W
        cords_new[:,3] =               cords[:,3]      / H

        all_cord_list = ''
        for typ, cord in zip(types, cords_new):
            trial = [typ] + list(cord)
            prv = ''
            for s in [str(x) for x in trial]:
                prv = prv + ' ' + s

            all_cord_list = all_cord_list + prv + '\n'

        with open(id_label_path, 'w') as f:
            f.writelines(all_cord_list)


for subset in ['train', 'val']:
    print('\n\t', subset)

    df = DataFrames[subset]

    src_dir = f"{DATA_DIR['cityscapes']}/leftImg8bit/{subset}"
    img_dir = f"{DATA_DIR['citypersons']}/{subset}/images"
    anno_dir = f"{DATA_DIR['citypersons']}/{subset}/labels"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    preprocess(df, src_dir, img_dir, anno_dir)


# Check the annotations
print('\n\nChecking ...')

subset = 'train'
idd = list(DataFrames[subset]['image_id'].unique())[rd.randint(0, 2000)]

image = cv2.imread(f"{DATA_DIR['citypersons']}/{subset}/images/{idd}")
h, w = image.shape[:2]
print('Image shape:', image.shape)  # (800, 1360, 3)

with open(f"{DATA_DIR['citypersons']}/{subset}/labels/{idd.split('.')[0]}.txt") as f:
    lst = []
    for line in f:
        lst += [line.rstrip()]
        print('\t', line)

bbox_color = [172 , 10, 127]
bbox_thick = 2
bbox_font = cv2.FONT_HERSHEY_COMPLEX

for i in range(len(lst)):
    # Getting current bounding box coordinates, its width and height
    bb_current = lst[i].split()
    x_center ,  y_center  = int(float(bb_current[1]) * w), int(float(bb_current[2]) * h)
    box_width, box_height = int(float(bb_current[3]) * w), int(float(bb_current[4]) * h)
    
    # For YOLO format, we can get top-left corner coordinates that are x_min and y_min
    x_min = int(x_center - (box_width / 2))
    y_min = int(y_center - (box_height / 2))

    # Drawing bounding box on the original 
    cv2.rectangle(image, (x_min, y_min), 
                         (x_min + box_width, y_min + box_height), bbox_color, bbox_thick)

    # Preparing text with label and confidence for current bounding box
    class_current = 'Class: {}'.format(bb_current[0])

    # Putting text with label and confidence on the original image
    cv2.putText(image, class_current, (x_min, y_min - 5), bbox_font, 0.7, bbox_color, bbox_thick)

fig = plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Image 00003.jpg with Traffic Signs', fontsize=18)
plt.show()

# fig.savefig('example.png')
# plt.close()

