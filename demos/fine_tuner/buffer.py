import torch
from collections import deque

class ContrastiveBuffer:
    def __init__(self, positive_buffer_size, negative_buffer_size):
        self.positive_buffer_size = positive_buffer_size
        self.negative_buffer_size = negative_buffer_size
        self.negative_buffer = deque(maxlen=negative_buffer_size)
        self.positive_buffer = deque(maxlen=positive_buffer_size)
    
    def add_to_buffer(self, samples, positive=True):
        target_buffer = self.positive_buffer if positive else self.negative_buffer
        for sample in samples:
            target_buffer.append(sample)
    
    def get_negatives(self, num_negatives):
        num_negatives = min(num_negatives, len(self.negative_buffer))
        indices = torch.randperm(len(self.negative_buffer))[:num_negatives]
        negatives = [self.negative_buffer[idx] for idx in indices]
        return negatives

    def get_positives(self, num_positives):
        num_positives = min(num_positives, len(self.positive_buffer))
        indices = torch.randperm(len(self.positive_buffer))[:num_positives]
        positives = [self.positive_buffer[idx] for idx in indices]
        return positives
