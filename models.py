import torch
from torch import nn, Tensor
from torchvision.models import resnet18


class IrisDetector(nn.Module):
    INPUT_SIZE = (224, 224)

    def __init__(self):
        super().__init__()
        self.detector = resnet18(pretrained=True)
        self.detector.fc = nn.Linear(self.detector.fc.in_features, 4)

    def forward(self, x: Tensor):
        x = x.repeat((1, 3, 1, 1))
        return self.detector(x)


class IrisEncoder(nn.Module):
    INPUT_SIZE = (224, 1000)
    CROP_SIZE = (224, 224)
    LEFT_OFFSETS = (175, 601)
    CONV_FEATURES_DIM = 128
    embedding_dim = CONV_FEATURES_DIM * 2

    def __init__(self):
        super().__init__()
        self.conv_features = resnet18(pretrained=True)
        self.conv_features_dim = self.conv_features.fc.in_features

        self.conv_features.fc = nn.Sequential(
            nn.Linear(self.conv_features.fc.in_features, self.CONV_FEATURES_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        left = x[..., self.LEFT_OFFSETS[0]:self.LEFT_OFFSETS[0] + 224]
        right = x[..., self.LEFT_OFFSETS[1]:self.LEFT_OFFSETS[1] + 224]
        left_embed = self.conv_features(left)
        right_embed = self.conv_features(right)
        return torch.cat([left_embed, right_embed], dim=-1)
