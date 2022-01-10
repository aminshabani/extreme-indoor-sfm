import numpy as np

class BBox:
    def __init__(self, bbox=None, obj_type=None):
        self.bbox = bbox  #BoundingBox (2,3)
        self.type = obj_type
        if abs(self.bbox[0][0]-self.bbox[1][0])<1e-4:
            if self.bbox[0][0]>0:
                self.direction = 0
            else:
                self.direction = 2
        else:
            if self.bbox[0][2]>0:
                self.direction = 1
            else:
                self.direction = 3

