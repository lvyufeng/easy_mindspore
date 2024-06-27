from .module import Module
from easy_mindspore.nn import functional as F

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        return F.softmax_cross_entropy(logits, labels)
