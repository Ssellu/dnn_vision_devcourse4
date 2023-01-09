import argparse
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader

import torch.optim as optim

from model.models import *
from loss.loss import *


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument('--mode', dest='mode',
                        help='train / eval / test', default=None, type=str)
    parser.add_argument('--download', dest='download',
                        help='Downlpad MNIST dataset', default=False, type=bool)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Output directory', default='./output', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='Checkpoint trained model', default=None, type=str)

    if len(sys.argv) == 1:
        sys.exit()
    args = parser.parse_args()
    return args


def get_mnist_dataset():
    download_root = './mnist_dataset'
    my_transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    train_dataset = MNIST(root=download_root,
                          transform=my_transform,
                          train=True,
                          download=args.download)
    eval_dataset = MNIST(root=download_root,
                         transform=my_transform,
                         train=True,
                         download=args.download)
    test_dataset = MNIST(root=download_root,
                         transform=my_transform,
                         train=False,
                         download=args.download)
    return train_dataset, eval_dataset, test_dataset


def main():
    print(torch.__version__)

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    # Get MNIST Dataset
    train_dataset, eval_dataset, test_dataset = get_mnist_dataset()

    # Create DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(train_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)
    test_loader = DataLoader(train_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    # LeNet5
    _model = get_model('lenet5')

    if args.mode == 'train':
        model = _model(batch=8, n_classes=10, in_channel=1,
                       in_width=32, in_height=32, is_train=True)
        model.to(device=device)
        model.train()

        # Set Optimizer and Sceduler
        # Stochastic Gradient Descent
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1)

        # Set Loss
        criterion = get_criterion(crit="mnist", device=device)

        epoch = 15
        iter = 0
        for e in range(epoch):
            total_loss = 0
            for i, batch in enumerate(train_loader):
                img = batch[0]
                gt = batch[1]

                # img = img.to(device)
                # gt = gt.to(device)
                out = model(img)
                # print('Trained Image : {}'.format(out))

                if out is None:
                    continue

                # Get loss value before back-propagation
                loss_val = criterion.forward(out, gt)

                # Back-propagation
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss_val.item()

                if iter % 100 == 0:
                    print('{} epoch {} iter loss : {}'.format(
                        e, iter, loss_val.item()))
                iter += 1

            mean_loss = total_loss / i
            scheduler.step()
            print('-> {} epoch mean loss : {}'.format(e, mean_loss))

            torch.save(model.state_dict(),
                       '{}/model_epoch/{}.pt'.format(args.output_dir, e))
        print('Training has ended.')


if __name__ == '__main__':
    args = parse_args()
    main()
