import torch
import torch.nn as nn
import sys


class MNISTLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')) -> None:
        super(MNISTLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device=device)

    def forward(self, out, gt):
        # Return loss value
        return self.loss(out, gt)


def get_criterion(crit="mnist", device = torch.device('cpu')):
    if crit is "mnist":
        return MNISTLoss(device=device)
    

    print("Unknown Criterion")
    sys.exit(1)
