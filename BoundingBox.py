class BoundingBox:
    def __init__(self, x: [int], y: [int], w: [int], h: [int], center_x: [int], center_y: [int]):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center_x = center_x
        self.center_y = center_y

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h, self.center_x, self.center_y))
