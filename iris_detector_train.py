import pickle
import random
import sys

import numpy as np
import torch
import torch.nn.functional as fn
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torchinfo import summary
from torchvision.transforms.functional import resize, to_tensor

from iris_detector_manual import preview_detection
from models import IrisDetector
from utils import load_picture


class IrisDetectorDataset(Dataset):
    SIZE = [224, 298]
    SIZE_CROP = [224, 224]
    SPLIT = 4 / 5

    def __init__(self, meta_path, train=True):
        super().__init__()
        self.train = train

        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)

        rd = random.getstate()
        random.seed(3)
        random.shuffle(self.metadata)
        random.setstate(rd)

        if train:
            self.metadata = self.metadata[:int(len(self.metadata) * self.SPLIT)]
            random.shuffle(self.metadata)
        else:
            self.metadata = self.metadata[int(len(self.metadata) * self.SPLIT):]

    def __getitem__(self, index):
        path, target = self.metadata[index]
        img = load_picture(path)
        ratio = self.SIZE[1] / img.shape[1]
        img = to_tensor(img)
        img = resize(img, self.SIZE)

        if self.train:
            offset = int(((298 - 224) * random.uniform(0, 1)))
        else:
            offset = (298 - 224) // 2
        img = img[..., offset:offset + 224]

        target = target[0] * ratio - offset, target[1] * ratio, target[2] * ratio, target[3] * ratio
        return img, torch.Tensor(target)

    def __len__(self):
        return len(self.metadata)


train_data = IrisDetectorDataset('manual.pck', train=True)
test_data = IrisDetectorDataset('manual.pck', train=False)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=12,
                                           pin_memory=True,
                                           prefetch_factor=4)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=8,
                                          pin_memory=True,
                                          prefetch_factor=4)

detector = IrisDetector()
detector = detector.cuda()
summary(detector, (2, 1, 224, 224), col_names=("input_size", "output_size", "num_params"), depth=4)

LR = 0.01
optimizer = torch.optim.AdamW(detector.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
loss_fn = nn.MSELoss()
EPOCHS = 100

epoch = 0
while epoch < EPOCHS:
    epoch += 1
    print(f'Epoch: {epoch}')
    detector.train()
    epoch_losses = []
    epoch_diffs = []
    for step, (img, target) in enumerate(train_loader):
        img, target = img.cuda(), target.cuda()
        out = detector(img)
        loss_val = loss_fn(out, target)
        diff = fn.l1_loss(out, target)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        epoch_losses.append(loss_val.item())
        epoch_diffs.append(diff.item())
        print('    Batch {} of {} loss: {}, diff: {}, lr: {}'.format(step + 1, len(train_loader), loss_val.item(),
                                                                     diff.item(), optimizer.param_groups[0]["lr"]),

              file=sys.stderr)
    scheduler.step()
    print(f'Train loss: {np.mean(epoch_losses):.4f}, diff: {np.mean(epoch_diffs):.4f}')
    with torch.no_grad():
        detector.eval()
        epoch_losses = []
        epoch_diffs = []
        for step, (img, target) in enumerate(test_loader):
            img, target = img.cuda(), target.cuda()
            out = detector(img)
            loss_val = loss_fn(out, target)
            diff = fn.l1_loss(out, target)
            print('    Batch {} of {} loss: {}, diff: {}'.format(step + 1, len(train_loader), loss_val.item(),
                                                                 diff.item()), file=sys.stderr)
            epoch_losses.append(loss_val.item())
            epoch_diffs.append(diff.item())
    print(f'Test loss: {np.mean(epoch_losses):.4f}, diff: {np.mean(epoch_diffs):.4f}')

with torch.no_grad():
    detector.eval()
    fig, axs = plt.subplots(20, 2, figsize=(12, 60))
    for i in range(20):
        img, expected = test_data[i]
        tgt = detector(img.unsqueeze(0).cuda()).cpu()[0].numpy().astype(np.int32)
        axs[i, 0].imshow(preview_detection(img.squeeze(0).numpy(), tgt))
        axs[i, 0].set_title('result')
        axs[i, 1].imshow(preview_detection(img.squeeze(0).numpy(), expected.numpy().astype(np.int32)))
        axs[i, 1].set_title('expected')
    fig.tight_layout()
    plt.show()

torch.save(detector.state_dict(), 'iris.detector.pck')
