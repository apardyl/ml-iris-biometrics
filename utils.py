import cv2 as cv
import faiss
import numpy as np
import torch
from torch import nn


def load_picture(path):
    img = (cv.imread(path, cv.IMREAD_COLOR) / 255).astype(np.float32)
    img = img[..., 2]  # use red channel only
    return img


def save_train_state(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, best_score: float,
                     file_path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_score': best_score,
    }, file_path)


def load_train_state(file_path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    if 'scheduler' in data:
        scheduler.load_state_dict(data['scheduler'])
    return data['epoch'], data.get('best_score', 0)


def load_model_state(file_path, model: nn.Module):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])


# noinspection PyArgumentList
class Searcher:
    def __init__(self, index: faiss.Index):
        self.index = index

    def save(self, file_path: str):
        faiss.write_index(self.index, file_path)

    def search(self, x: torch.Tensor, k: int = 1):
        return self.index.search(x, k)

    def add(self, x: torch.Tensor):
        self.index.add(x)

    @classmethod
    def load(cls, file_path: str):
        index = faiss.read_index(file_path)
        return cls(index=index)

    @classmethod
    def get_simple_index(cls, embedding_dim):
        return cls(index=faiss.IndexFlatL2(embedding_dim))
