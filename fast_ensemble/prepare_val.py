# -*- coding: GBK -*-
import os
import argparse
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--val_data_path')
parser.add_argument('--dev_kit_path')
args = parser.parse_args()

annotation_path = os.path.join(args.dev_kit_path, 'data', 'ILSVRC2011_validation_ground_truth.txt')
mat_path = os.path.join(args.dev_kit_path, 'data', 'meta.mat')
with open(annotation_path) as f:
    categories = f.readlines()
val_images = os.listdir(args.val_data_path)
val_images.sort()

val_paths = [os.path.join(args.val_data_path, val_image) for val_image in val_images]
category_ids = [name.strip() for name in categories]

for i, category_id in enumerate(category_ids):
    img_path = os.path.join(args.val_data_path, 'ILSVRC2011_val_{}.JPEG'.format(str(i+1).zfill(8)))
    img_folder = os.path.join(args.val_data_path, category_id)
    new_img_path = os.path.join(img_folder, 'ILSVRC2011_val_{}.JPEG'.format(str(i+1).zfill(8)))
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    os.system('mv {0} {1}'.format(img_path, img_folder))


mat = scipy.io.loadmat(mat_path)['synsets']
for i in range(1000):
    cat_id = mat[i][0][0][0][0]
    synset_id = mat[i][0][1][0]
    origin_folder = os.path.join(args.val_data_path, str(cat_id))
    new_folder = os.path.join(args.val_data_path, synset_id)
    os.system('mv {0} {1}'.format(origin_folder, new_folder))
