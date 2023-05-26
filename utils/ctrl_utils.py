"""
    Store information in this class to control the state
"""

class ControlInfo:
    """ Control information for visualization"""
    def __init__(self, vertex_x):
        self.calculate_pdf    = False
        self.calculate_sample = False
        self.dir_selected     = False
        self.use_tr           = True
        self.pos_x = -1
        """ pos_x and pos_y is in world frame (divided by scale already)"""
        self.pos_y = -1
        self.vertex_x = vertex_x
        self.length           = 0.

    def reset(self, vertex_x):
        self.calculate_pdf    = False
        self.calculate_sample = False
        self.dir_selected     = False
        self.use_tr           = True
        self.pos_x = -1
        self.pos_y = -1
        self.vertex_x = vertex_x
        self.length           = 0.