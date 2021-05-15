import argparse
import math
import os
import pathlib
import pickle

import cv2 as cv
import numpy as np

from utils import load_picture

parser = argparse.ArgumentParser(description='Extract iris positions.')
parser.add_argument('dataset_path', type=str)
args = parser.parse_args()

eye_files = list(str(p) for p in pathlib.Path(args.dataset_path).rglob('*.jpg'))

with open('{}.iris_info.pck'.format(os.path.basename(args.dataset_path)), 'rb') as f:
    metadata = pickle.load(f)

metadata = {k: v for k, v in metadata}

for file in eye_files:
    save_path = os.path.join('extracted', file)
    img = load_picture(file)
    tgt = metadata[file]

    width = tgt[3]
    radius = tgt[2]
    rectangle = np.zeros((width, int(2 * radius * math.pi)), dtype='float32')
    for row in range(rectangle.shape[0]):
        for col in range(rectangle.shape[1]):
            t = math.pi * 2 / rectangle.shape[1] * (col + 1)
            rho = radius - row - 1
            x = int(tgt[0] + rho * math.sin(t) + 0.5) - 1
            y = int(tgt[1] - rho * math.cos(t) + 0.5) - 1
            rectangle[row, col] = img[y, x]

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clh = clahe.apply((rectangle * 255).astype(np.uint8))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv.imwrite(save_path + '.png', clh)
    print('processed:', file)
