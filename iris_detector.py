import argparse
import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_tensor, resize

from iris_detector_manual import preview_detection
from models import IrisDetector
from utils import load_picture

detector = IrisDetector().cuda()
detector.load_state_dict(torch.load('iris.detector.pck'))
detector.eval()

parser = argparse.ArgumentParser(description='Automatically detect iris positions.')
parser.add_argument('dataset_path', type=str)
args = parser.parse_args()

eye_files = list(str(p) for p in pathlib.Path(args.dataset_path).rglob('*.jpg'))

metadata = []
dump_test = True

with torch.no_grad():
    for file in eye_files:
        save_path = os.path.join('test', os.path.basename(file))
        img = load_picture(file)
        offset = (img.shape[1] - img.shape[0]) // 2
        reshaped = img[:, offset:offset + img.shape[0]]
        ratio = detector.INPUT_SIZE[0] / reshaped.shape[0]
        reshaped = resize(to_tensor(reshaped), detector.INPUT_SIZE)
        tgt = detector(reshaped.unsqueeze(0).cuda()).cpu().numpy()[0]
        tgt /= ratio
        tgt[0] += offset
        out = preview_detection(img, tgt)
        tgt = (int(tgt[0]), int(tgt[1]), int(tgt[2]), int(tgt[3]))
        if dump_test:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.imsave(save_path, out)
        metadata.append((file, tgt))
        print('processed:', file)

with open('{}.iris_info.pck2'.format(os.path.basename(args.dataset_path)), 'wb') as f:
    pickle.dump(metadata, f)
