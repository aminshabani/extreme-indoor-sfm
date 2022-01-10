import numpy as np


class Tree:
    def __init__(self):
        self.rooms = []
        self.pairs_list = []
        self.num_pos = 0
        self.num_neg = 0

    def __len__(self):
        return len(self.rooms)

    def add_pair(self, pair, is_positive):
        self.pairs_list.append(pair)
        if (is_positive):
            self.num_pos += 1
        else:
            self.num_neg += 1

    def drop_last(self, is_positive):
        self.pairs_list = self.pairs_list[:-1]
        if (is_positive):
            self.num_pos -= 1
        else:
            self.num_neg -= 1
