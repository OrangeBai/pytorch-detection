import piexif
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str)
args = parser.parse_args()
counter = 0
for root, d, files in os.walk(args.data_root):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.splitext(file_path)[-1] == '.JPEG':
            try:
                piexif.remove(file_path)
                Image.open(file_path)
            except:
                os.remove(file_path)
                print('File {0} is corrupted, deleting'.format(file_path))
        if counter % 500 == 0:
            print(counter)
        counter += 1
