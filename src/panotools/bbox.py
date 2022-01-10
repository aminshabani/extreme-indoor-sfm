import shapely
import numpy as np

# thing_classes=["Door","Glass_door","Frame","Window","Kitchen_counter","closet"]


class BBox:
    def __init__(self, bbox=None, obj_type=None):
        # if type(obj_type).__module__ == np.__name__:
        #     obj_type = int(np.argmax(obj_type))
        self.bbox = np.round(bbox, 3)  # BoundingBox (2,3) xyz
        self.type = obj_type
        if abs(self.bbox[0][0]-self.bbox[1][0]) < abs(self.bbox[0][2]-self.bbox[1][2]):
            if self.bbox[0][0] > 0:
                self.direction = 0
            else:
                self.direction = 2
        else:
            if self.bbox[0][2] > 0:
                self.direction = 1
            else:
                self.direction = 3

    def get_type(self):
        if isinstance(self.type, np.ndarray):
            return np.argmax(self.type)
        return self.type

    def get_center(self):
        center = (self.bbox[0]+self.bbox[1])/2
        center = shapely.geometry.Point([center[0], center[2]])
        return center

    def get_line(self):
        line = shapely.geometry.LineString(
            [(self.bbox[0, 0], self.bbox[0, 2]), (self.bbox[1, 0], self.bbox[1, 2])])
        return line

    def length(self):
        return self.get_line().length
