import os
from config import *


def download_imagenet(url):
    pass


def prepare_dataset():
    imagenet_path = os.path.join(DATA_PATH, 'imagenet')
    img_path = os.path.join(imagenet_path, 'images')
    train_path = os.path.join(img_path, 'train')

    os.system('cd {0}'.format(img_path))
    tar_files = [file for file in os.listdir(img_path) if os.path.splitext(file)[1] == '.tar']
    cls_ids = [file_name.split('.')[0] for file_name in tar_files]
    for tar_file, cls_id in zip(tar_files, cls_ids):
        image_folder = os.path.join(train_path, cls_id)
        if not os.path.exists:
            os.makedirs(image_folder)
        os.system('mv {0} {1}'.format(os.path.join(img_path, tar_file), image_folder))
        print('moving {0} to {1}'.format(tar_file, image_folder))
        os.system('cd {0}'.format(image_folder))
        os.system('tar -xvf {0} -C {1} -m'.format(os.path.join(image_folder, tar_file), image_folder))
        print('extracting {0} to {1}'.format(os.path.join(image_folder, tar_file), image_folder))
        os.system('cd ..')
    return


if __name__ == '__main__':
    prepare_dataset()