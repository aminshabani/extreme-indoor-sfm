from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger('log')


class MyWriter():
    def __init__(self, args, subdir):
        self.writer = SummaryWriter('exps/{}/{}/'.format(subdir, args.exp))
        self.values = dict()
        self.cnt = dict()

    def add_scalar(self, name, val):
        if (name in self.values):
            self.values[name] += val
            self.cnt[name] += 1.0
        else:
            self.values[name] = val
            self.cnt[name] = 1.0

    def add_imgs(self, pred, target, epoch):
        pred = make_grid(pred, 4)
        target = make_grid(target, 4)
        self.writer.add_image('pred', pred, epoch)
        self.writer.add_image('target', target, epoch)

    def push(self, step):
        for key in self.values:
            self.values[key] = self.values[key] / float(self.cnt[key])
        for key in self.values:
            self.writer.add_scalar(key, self.values[key], step)
        self.values = dict()

    def get(self, name):
        if (name not in self.values):
            logger.error("name {} was not found in the writer".format(name))
            return None
        return self.values[name] / float(self.cnt[name])
