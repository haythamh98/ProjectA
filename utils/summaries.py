import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
    '''The SummaryWriter class provides a high-level API to create an event file in a given directory and add summaries
     and events to it. The class updates the file contents asynchronously.  This allows a training program to call 
     methods to add data to the file directly from the training loop, without slowing down training.'''
    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer