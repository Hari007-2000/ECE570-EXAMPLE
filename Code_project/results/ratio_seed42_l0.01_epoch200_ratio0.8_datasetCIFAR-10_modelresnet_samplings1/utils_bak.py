import sys
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
import os

# Set seed function for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Colored print function
def cprint(color: str, text: str, **kwargs) -> None:
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30', 'r': '31', 'g': '32', 'y': '33',
        'b': '34', 'p': '35', 'c': '36', 'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()

# Class to keep track of averages
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Accuracy calculation function
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Training function
def train(train_loader, model, optimizer, epoch, N, ratio=0, samplings=1, print_freq=50):
    assert 0 <= ratio <= 1
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, (input_data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Move data to CPU
        input_var, target_var = input_data, target

        # Unpack the model output if it's a tuple (for Bayesian models)
        output, _ = model(input_var)
        loss = F.cross_entropy(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure accuracy and record loss
        prec1 = accuracy(output, target_var)[0]
        losses.update(loss.item(), input_data.size(0))
        top1.update(prec1.item(), input_data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    return losses.avg, top1.avg

# Validation function
def validate(val_loader, model, criterion, print_freq=50):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            input_var, target_var = input_data, target

            # Unpack the model output if it's a tuple (for Bayesian models)
            output, _ = model(input_var)
            loss = criterion(output, target_var)

            # Measure accuracy and record loss
            prec1 = accuracy(output, target_var)[0]
            losses.update(loss.item(), input_data.size(0))
            top1.update(prec1.item(), input_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Prec@1 {top1.avg:.3f}')
    return losses.avg, top1.avg

# Function to save data as an image
def data2img(data, path, reshape=True):
    if reshape:
        data = np.einsum('ijk->jki', data)
    data = (data * 255).round().astype(np.uint8)
    img = Image.fromarray(data)
    img.save(path)

if __name__ == '__main__':
    for color in ['a', 'r', 'g', 'y', 'b', 'p', 'c', 'w']:
        cprint(color, color)
        cprint('*' + color, '*' + color)
