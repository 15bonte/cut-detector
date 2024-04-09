import matplotlib as mpl

class MbTrackColorManager:
    def __init__(self):
        self.index = 0
        self.color_list = [
            mpl.colormaps["tab10"](i)[:3] for i in range(10)
        ]
        self.id2color = {}

    def get_color_for_track(self, id: int):
        color = self.id2color.get(id)
        if color is None:
            new_color = self.color_list[self.index]
            self.id2color[id] = new_color
            color = new_color
            self.inc_index()
        return color
    
    def inc_index(self):
        self.index += 1
        if self.index >= len(self.color_list):
            self.index = 0


