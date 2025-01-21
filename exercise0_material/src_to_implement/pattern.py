import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        if self.resolution %(2* self.tile_size)!=0:
            return

        tile = self.tile_size
        white_box = np.ones((tile * 2, tile * 2))
        black_box = np.zeros((tile, tile))
        white_box[0:tile, 0:tile] = black_box
        white_box[tile:tile * 2, tile:tile * 2] = black_box
        no_of_boxes = self.resolution / self.tile_size
        axes = int(no_of_boxes / 2)
        self.chessq = np.tile(white_box, (axes, axes))
        self.output = self.chessq.copy()
        return self.chessq

    def show(self):
        self.draw()
        plt.imshow(self.chessq, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        x = np.linspace(0, self.resolution, self.resolution)
        y = np.linspace(0, self.resolution, self.resolution)
        x_centre = self.position[0]
        y_centre = self.position[1]
        self.x_axis, y_axis = np.meshgrid(x, y)
        self.points = (self.x_axis - x_centre) ** 2 + (y_axis - y_centre) ** 2 <= self.radius ** 2
        self.output = self.points.copy()
        return self.points

    def show(self):
        self.draw()
        final_output = np.zeros(self.x_axis.shape)
        final_output[self.points] = 1
        plt.imshow(final_output, cmap='gray')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.ones([self.resolution, self.resolution, 3])

    def draw(self):
        self.output[:,:,0] = np.linspace(0,1,self.resolution)
        self.output[:,:,1] = np.linspace(0,1,self.resolution).reshape(self.resolution, 1)
        self.output[:,:,2] = np.linspace(1,0,self.resolution)
        return self.output.copy()

    def show(self):
        self.draw()
        plt.imshow(self.output)
        plt.show()