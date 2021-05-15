import argparse
import copy
import datetime
import os
import random
import shutil
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import IrisEncoder
from utils import load_train_state, save_train_state, Searcher

SAVED_MODELS_PATH = 'saved_models'
CHECKPOINT_FILE = 'checkpoint.pck'
LOGS_DIR = 'runs'


def evaluate_encoder(encoder: nn.Module, test_loader: torch.utils.data.DataLoader, writer: SummaryWriter = None,
                     epoch: int = -1, final=False):
    with torch.no_grad():
        encoder.eval()
        searcher = Searcher.get_simple_index(encoder.embedding_dim)
        embeddings = []
        targets = []
        for step, (x, target) in enumerate(test_loader):
            x = x.cuda()
            x_enc = encoder(x)
            embeddings.append(x_enc.cpu().numpy())
            targets.append(target.numpy())
            print('    Test batch {} of {}'.format(step + 1, len(test_loader)), file=sys.stderr)

        embeddings = np.concatenate(embeddings, axis=0)
        targets = np.concatenate(targets, axis=0)
        searcher.add(embeddings)
        lookup = searcher.search(embeddings, 100)[1]
        lookup = [[x for x in l if x != i] for i, l in enumerate(lookup)]
        correct_10 = sum(targets[y] in targets[x[:10]] for y, x in enumerate(lookup)) / len(lookup)
        correct_1 = sum(targets[y] == targets[x[0]] for y, x in enumerate(lookup)) / len(lookup)
        class_embeds = defaultdict(list)
        for embed, tgt in zip(embeddings, targets):
            class_embeds[tgt.item()].append(embed)
        class_means = {k: np.mean(v, axis=0) for k, v in class_embeds.items()}
        mean_correct_1 = 0
        for embed, tgt in zip(embeddings, targets):
            class_mean_without_x = np.mean([x for x in class_embeds[tgt.item()] if not np.array_equal(x, embed)],
                                           axis=0)
            best_other_val = 9999999999
            for k, v in class_means.items():
                if k == tgt.item():
                    continue
                dist = ((embed - v) ** 2).sum()
                if best_other_val > ((embed - v) ** 2).sum():
                    best_other_val = dist
            if ((embed - class_mean_without_x) ** 2).sum() < best_other_val:
                mean_correct_1 += 1
        mean_correct_1 /= len(embeddings)
        print(
            'Test accuracy:\n    top1 {}\n    top10 {}\n    mean1 {}'.format(
                correct_1, correct_10, mean_correct_1))
        if writer and not final:
            writer.add_scalar('Accuracy/Test_top1', correct_1, global_step=epoch)
            writer.add_scalar('Accuracy/Test_top10', correct_10, global_step=epoch)
            writer.add_scalar('Accuracy/Test_mean1', mean_correct_1, global_step=epoch)
            if epoch == -1 or epoch % 5 == 1:
                mat = embeddings[:1000]
                labels = targets[:1000]
                writer.add_embedding(mat, labels, tag='Embeddings', global_step=epoch)
        if writer and final:
            writer.add_text('FinalResults',
                            'top_1: {}, top_2: {}, mean1: {}'.format(correct_1, correct_10, mean_correct_1))
            mat = embeddings
            labels = targets
            writer.add_embedding(mat, labels, tag='EmbeddingsAll', global_step=epoch+1)
        return correct_1 * 100, correct_10 * 100


def train_encoder(train_loader, test_loader, all_loader, encoder, epochs=30):
    LR = 0.005

    summary(encoder, next(iter(train_loader))[0].shape, col_names=("input_size", "output_size", "num_params"),
            depth=4)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    epoch = 0
    best_model = copy.deepcopy(encoder)
    best_score = -1

    if os.path.exists(CHECKPOINT_FILE):
        print('Loading checkpoint')
        epoch, best_score = load_train_state(CHECKPOINT_FILE, encoder, optimizer, scheduler)

    log_file = os.path.join(LOGS_DIR, encoder.__class__.__name__)
    writer = SummaryWriter(log_file, purge_step=epoch, flush_secs=60)

    print("Learning started")

    AUG_ROUNDS = 5
    while epoch < epochs:
        epoch += 1
        print(f"Epoch: {epoch}")
        epoch_losses = []
        encoder.train()
        margin = np.sqrt(encoder.embedding_dim) / 4
        loss_fn = torch.nn.TripletMarginLoss(margin=margin)
        step = 0
        for round in range(AUG_ROUNDS):
            for idx, (x, y_pos, y_neg) in enumerate(train_loader):
                step += 1
                x, y_pos, y_neg = x.cuda(), y_pos.cuda(), y_neg.cuda()
                x_enc = encoder(x)
                y_pos_enc = encoder(y_pos)
                y_neg_enc = encoder(y_neg)
                loss_val = loss_fn(x_enc, y_pos_enc, y_neg_enc)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                epoch_losses.append(loss_val.item())
                print('    Batch {} of {} loss: {}, lr: {}'.format(step, len(train_loader) * AUG_ROUNDS,
                                                                   loss_val.item(), optimizer.param_groups[0]["lr"]),
                      file=sys.stderr)
        print(f'Train loss: {np.mean(epoch_losses):.4f}')
        writer.add_scalar('Loss/train', np.mean(epoch_losses), global_step=epoch)
        writer.add_scalar('Lr', optimizer.param_groups[0]["lr"], global_step=epoch)
        score = evaluate_encoder(encoder, test_loader, writer=writer, epoch=epoch)[0]
        if score > best_score:
            best_model = copy.deepcopy(encoder)
            best_score = score
            print('New best score')
            save_train_state(epoch, encoder, optimizer, scheduler, best_score, CHECKPOINT_FILE)
        scheduler.step()
    if best_score < 0:
        best_score = evaluate_encoder(encoder, test_loader, writer=writer)[0]

    writer.close()
    save_file_path = os.path.join(SAVED_MODELS_PATH, '{}.{}.{:.2f}.pck'.format(encoder.__class__.__name__,
                                                                               datetime.datetime.now().isoformat(),
                                                                               best_score))
    log_file_path = os.path.join(LOGS_DIR, '{}.{}.{:.2f}.pck'.format(encoder.__class__.__name__,
                                                                     datetime.datetime.now().isoformat(),
                                                                     best_score))
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    shutil.move(CHECKPOINT_FILE, save_file_path)
    shutil.move(log_file, log_file_path)

    evaluate_encoder(best_model, all_loader, writer, final=True)

    return best_model, best_score


class IrisEncoderDataset(ImageFolder):
    def __init__(self, *args, test=False, train_split=0.8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.samples_by_class = defaultdict(list)
        for sample, target in self.samples:
            self.samples_by_class[target].append((sample, target))
        self.samples_by_class = list(self.samples_by_class.items())
        split = int(len(self.samples_by_class) * train_split)
        if test:
            self.samples_by_class = self.samples_by_class[split:]
        else:
            self.samples_by_class = self.samples_by_class[:split]
        self.samples = []
        for y, x in self.samples_by_class:
            self.samples += x
        self.samples_by_class = {k: v for k, v in self.samples_by_class}


class IrisEncoderTripletDataset(IrisEncoderDataset):
    def __getitem__(self, index):
        path, target = self.samples[index]
        positive_path = random.choice(self.samples_by_class[target])[0]
        negative_class = random.choice(list(k for k in self.samples_by_class.keys() if k != target))
        negative_path = random.choice(self.samples_by_class[negative_class])[0]
        sample = self.loader(path)
        positive = self.loader(positive_path)
        negative = self.loader(negative_path)
        if self.transform is not None:
            sample = self.transform(sample)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return sample, positive, negative


def train(epochs=30):
    print('Training encoder for {} epochs'.format(epochs))
    encoder = IrisEncoder()
    encoder = encoder.cuda()

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(encoder.INPUT_SIZE),
        transforms.RandomCrop(size=encoder.INPUT_SIZE, padding=(20, 0))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(encoder.INPUT_SIZE)
    ])

    dataset_path = 'extracted/data/UBIRIS_800_600_Sessao_1'
    train_data = IrisEncoderTripletDataset(root=dataset_path, test=False,
                                           transform=transform_train)
    test_data = IrisEncoderDataset(root=dataset_path, test=True,
                                   transform=transform_test)
    all_data = IrisEncoderDataset(root=dataset_path, test=True, train_split=0,
                                  transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True,
                                               prefetch_factor=4)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=100,
                                              shuffle=False,
                                              num_workers=8,
                                              pin_memory=True,
                                              prefetch_factor=4)
    all_loader = torch.utils.data.DataLoader(all_data,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             prefetch_factor=4)
    train_encoder(train_loader, test_loader, all_loader, encoder, epochs)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Train models.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=10, help='How many epochs to train for')
    args = parser.parse_args()
    train(args.epochs)
