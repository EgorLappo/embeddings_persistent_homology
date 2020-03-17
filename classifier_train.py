import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random

class SeqCNN(nn.Module):
    def __init__(self, n):
        '''n: length of input tensor'''
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.c2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.f1 = nn.Linear(n * 32, 50)
        self.drop = nn.Dropout(0.5)
        self.f2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.c1(x.unsqueeze(1)))
        x = F.relu(self.c2(x))
        x = x.flatten(1)
        x = F.relu(self.f1(x))
        x = self.drop(x)
        x = self.f2(x)
        return x, sum(x.pow(2).sum() for x in self.parameters())


parser = argparse.ArgumentParser(description="train on sequences")
parser.add_argument('data', type=str, help='data directory')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--test', action='store_true', help='run on test set')
args = parser.parse_args()

train = np.load(f'{args.data}/train.npy', allow_pickle=True)
val = np.load(f'{args.data}/val.npy', allow_pickle=True)
print(f"Found {train.shape[0]} training and {val.shape[0]} validation data points.")
model = SeqCNN(train[0][0].shape[0]).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
batch = 100
lam = 0.001
loss_fn = nn.CrossEntropyLoss().cuda()


def get_loss(data):
    X, Y = map(torch.tensor, zip(*data))
    Y_hat, l2 = model(X.cuda().float())
    loss = loss_fn(Y_hat, Y.cuda()) + lam * l2
    return loss


def evaluate(data):
    X, Y = map(torch.tensor, zip(*data))
    Y_hat, l2 = model(X.cuda().float())
    return np.round(100 * (torch.argmax(Y_hat, dim=1) == Y.cuda()).sum().item() / Y.shape[0], 2)


avg_short_val = 1.
avg_long_val = 1.
long_decay = 0.05
short_decay = 0.1

for i in range(args.epochs):
    model.train()
    feed = list(train) 
    random.shuffle(feed)
    while feed:
        mb, feed = feed[:batch], feed[batch:]
        loss = get_loss(mb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    val_loss = get_loss(val).item()
    avg_short_val = (1 - short_decay) * avg_short_val + short_decay * val_loss
    avg_long_val = (1 - long_decay) * avg_long_val + long_decay * val_loss
    print(f'Epoch {i} - Train Loss: {get_loss(train).item()}, Validation Loss: {val_loss}')
    if avg_long_val < avg_short_val:
        break

print()
print(f'Train Accuracy: {evaluate(train)}%')
print(f'Validation Accuracy: {evaluate(val)}%')
if args.test:
    print(f'Test Accuracy: {evaluate(val)}%')



