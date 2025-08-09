from typing import List
import torch
from torch import Tensor
from sklearn.metrics import f1_score

def topk_accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k / batch_size).item())
        return res

def macro_f1(output: Tensor, target: Tensor) -> float:
    with torch.no_grad():
        preds = output.argmax(dim=1).cpu().numpy()
        tgt = target.cpu().numpy()
        return f1_score(tgt, preds, average='macro')

class AverageMeter:
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
        self.avg = self.sum / self.count if self.count else 0
