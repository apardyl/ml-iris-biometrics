import argparse
import os
import pathlib
import pickle
import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# P_C = 1.5
P_C = 1.2


def preview_detection(img, target, show=False):
    cornea_circle = int(target[0]), int(target[1]), int(target[2])
    pupil_circle = int(target[0]), int(target[1]), int(target[2] - target[3])
    output = img[:, :, None].repeat(3, axis=2)
    x, y, r = cornea_circle
    cv.circle(output, (x, y), r, (0, 1, 0), 2)
    x, y, r = pupil_circle
    cv.circle(output, (x, y), r, (1, 0, 0), 2)
    if show:
        plt.imshow(output)
        plt.show()
    return output


def find_iris(img_path):
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    img = (cv.imread(img_path, cv.IMREAD_COLOR) / 255).astype(np.float32)

    axs[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap='gray')
    axs[0, 0].set_title('original')

    img = img[..., 2]

    axs[0, 1].imshow(img, cmap='gray')
    axs[0, 1].set_title('normalised')

    blur = cv.GaussianBlur(img, (9, 9), cv.BORDER_DEFAULT)
    P = blur.sum() / (blur.shape[0] * blur.shape[1])

    axs[0, 2].imshow(blur, cmap='gray')
    axs[0, 2].set_title('blur')

    _, pupil = cv.threshold(blur, P / 4.5, 1, cv.THRESH_BINARY)
    _, cornea = cv.threshold(blur, P / P_C, 1, cv.THRESH_BINARY)

    pupil8 = (pupil * 255).astype(np.uint8)
    pupil_circles = cv.HoughCircles(pupil8, cv.HOUGH_GRADIENT, 1, 200,
                                    param1=100, param2=10, minRadius=5, maxRadius=120)
    pupil_circle = pupil_circles[0, 0, :]
    pupil_circle = np.round(pupil_circle).astype("int")

    output = pupil8[:, :, None].repeat(3, axis=2)
    x, y, r = pupil_circle
    cv.circle(output, (x, y), r, (255, 0, 0), 4)
    axs[1, 0].imshow(output, cmap='gray')
    axs[1, 0].set_title('pupil circle')

    cornea8 = (cornea * 255).astype(np.uint8)
    cornea_circles = cv.HoughCircles(cornea8, cv.HOUGH_GRADIENT, 1, 200,
                                     param1=100, param2=10, minRadius=20, maxRadius=200)
    cornea_circle = cornea_circles[0, 0, :]
    cornea_circle = np.round(cornea_circle).astype("int")

    output = cornea8[:, :, None].repeat(3, axis=2)
    x, y, r = cornea_circle
    cv.circle(output, (x, y), r, (255, 0, 0), 4)
    axs[1, 1].imshow(output, cmap='gray')
    axs[1, 1].set_title('cornea circle')

    # use center of the pupil
    cornea_circle = pupil_circle[0], pupil_circle[1], cornea_circle[2]

    x, y = pupil_circle[0:2]
    radius = cornea_circle[2]
    width = cornea_circle[2] - pupil_circle[2]
    status = None
    target = x, y, radius, width

    axs[1, 2].imshow(preview_detection(img, target))
    axs[1, 2].set_title('result')
    fig.tight_layout()

    def good(self):
        nonlocal status
        status = 1
        plt.close()

    def bad(self):
        nonlocal status
        status = 0
        plt.close()

    def key_press(ev):
        if ev.key == '1':
            good(ev)
        elif ev.key == '0':
            bad(ev)

    bt_ok = Button(axs[2, 2], 'OK')
    bt_ok.on_clicked(good)
    bt_skip = Button(axs[2, 0], 'SKIP')
    bt_skip.on_clicked(bad)

    axs[2, 1].axis('off')
    fig.canvas.mpl_connect('key_press_event', key_press)
    plt.show()

    if status is None:
        raise EOFError
    if status:
        return target
    return None


def build_target_list(dataset_path, skip=None):
    eye_files = list(str(p) for p in pathlib.Path(dataset_path).rglob('*.jpg'))
    random.shuffle(eye_files)
    targets = []
    for file in eye_files:
        if skip and file in skip:
            print('already known:', file)
            continue
        try:
            res = find_iris(file)
        except EOFError:
            return targets
        except Exception:
            print('invalid file:', file)
            plt.close()
            continue
        if res:
            targets.append((file, res))
            print('ok:', res, 'new:', len(targets) + (len(skip) if skip else 0))
        else:
            print('skip')
    return targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manually detect iris positions.')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('save_path', type=str)
    args = parser.parse_args()

    if os.path.exists(args.save_path):
        with open(args.save_path, 'rb') as f:
            targets = pickle.load(f)
            print('loaded', len(targets), 'entries')
    else:
        print('starting new session')
        targets = []

    targets += build_target_list(args.dataset_path, {k for k, v in targets})

    if targets:
        with open(args.save_path, 'wb') as f:
            pickle.dump(targets, f)
            print('saved', len(targets), 'entries to', args.save_path)
    else:
        print('nothing to save')
