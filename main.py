import pygame
import numpy as np
from time import sleep
from threading import Thread
from random import choice, shuffle, randint


class Velocity:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y


class Point:
    def __init__(self, row: int, column: int, direction: int):
        self.row, self.column, self.direction = row, column, direction
        self.value = game.map.matrix[row, column]


class Shape:
    def __init__(self, dimensions: tuple[int, int]):
        self.width, self.height = dimensions


class Screen:
    def __init__(self, resolution: tuple[int, int]):
        self.resolution = Shape(resolution)
        self.display = pygame.display.set_mode(resolution)


# This function is used to create a random color (r, g, b).
def random_color(start=100, end=200):
    r = randint(start, end)
    g = randint(start, end)
    b = randint(start, end)
    color = (r, g, b)
    return color


# direction map for pathfinding.
# +-----+-----+-----+
# |  1  |  2  |  3  |
# +-----+-----+-----+
# |  4  |  5  |  6  |
# +-----+-----+-----+
# |  7  |  8  |  9  |
# +-----+-----+-----+
# It shows the next possible direction values relative to the previous direction value.
direction_map = {1: [2, 4],
                 2: [2, 4, 6],
                 3: [2, 6, 9],
                 4: [2, 4, 8],
                 5: [2, 4, 6, 8],
                 6: [2, 6, 8],
                 7: [4, 8],
                 8: [4, 6, 8],
                 9: [6, 8]}

"""
# For diagonal movement: (not recommended)
direction_map = {1: [1, 2, 3, 4, 7],
                 2: [1, 2, 3, 4, 6],
                 3: [1, 2, 3, 6, 9],
                 4: [1, 2, 4, 7, 8],
                 5: [1, 2, 3, 4, 6, 7, 8, 9],
                 6: [2, 3, 6, 8, 9],
                 7: [1, 4, 7, 8, 9],
                 8: [4, 6, 7, 8, 9],
                 9: [3, 6, 7, 8, 9]}
"""

# velocity map for pathfinding.
velocity_map = {1: Velocity(-1, -1),
                2: Velocity(0, -1),
                3: Velocity(1, -1),
                4: Velocity(-1, 0),
                6: Velocity(1, 0),
                7: Velocity(-1, 1),
                8: Velocity(0, 1),
                9: Velocity(1, 1)}

# 0: empty, 1: wall, 2: start point, 3: end_point, 4: empty but reserved.
color_map = {0: (255, 255, 255),
             1: (0, 0, 0),
             2: (0, 0, 255),
             3: (255, 0, 0),
             4: (255, 255, 255)}


# a class for map.
class Map:
    def __init__(self, map_size):
        self.matrix = None
        self.shape = None
        self.create(map_size)
        self.start_points = []

    # This function is used to create the map with the given map size
    def create(self, map_size):
        self.shape = Shape(map_size)
        width, height = map_size
        self.matrix = np.ones((width, height))
        self.matrix[1: (width - 1), 1: (height - 1)] = 0

    # This function is used to check if at least one start point and end point are defined on the map.
    def check(self):
        number_of_start_points = (self.matrix == 2).sum()
        number_of_end_points = (self.matrix == 3).sum()
        if number_of_start_points > 0 and number_of_end_points > 0:
            return True
        else:
            return False


# a class for pathfinding process.
class Pathfinding:
    def __init__(self):
        self.path_color = None
        self.path_number = 5
        self.best_paths = []
        self.need_reset = False
        self.end_point = None
        self.dictionary = {}
        self.not_found = True
        self.paths = []
        self.path = []
        self.number = 10
        self.busy = False

    # This function is used to reset the variables.
    def reset(self, reset_best_paths=True):
        self.need_reset = False
        self.end_point = None
        self.dictionary = {}
        self.not_found = True
        self.paths = []
        self.path = []
        # The number of repetitions to find a much better path.
        self.number = 10
        self.busy = False
        if reset_best_paths is True:
            self.best_paths = []
        game.update()

    # This function is used to repeat the pathfinding process just to reach better results. (not guaranteed)
    def find_another_way(self, start_point):
        self.dictionary = {}
        self.not_found = True
        self.path = []
        self.flow(start_point)

    # This function is used to find new points by using the direction map.
    def move(self, point):
        new_points = []
        directions = direction_map[point.direction]
        shuffle(directions)
        for direction in directions:
            velocity = velocity_map[direction]
            new_point = Point(point.row + velocity.x, point.column + velocity.y, direction)
            if new_point.value == 0:
                new_points.append(new_point)
                game.map.matrix[new_point.row, new_point.column] = 4
            elif new_point.value == 3:
                new_points.append(new_point)
                self.end_point = new_point
                self.not_found = False
        shuffle(new_points)
        return new_points

    # This 'recursive' function is used to create a list of points by using the information from dictionary.
    def create_list(self, last_point):
        for point in self.dictionary:
            if last_point in self.dictionary[point]:
                self.path.append(last_point)
                self.create_list(point)

    # This function is used to find the shortest path (best path) by comparing their lengths.
    # Might be more useful when the diagonal movement is valid.
    def find_best_path(self):
        lengths = []
        for path in self.paths:
            lengths.append(len(path))
        minimum_length = min(lengths)
        index_value = lengths.index(minimum_length)
        best_path = self.paths[index_value]
        self.best_paths.append(best_path)
        return best_path

    # This function is used to apply pathfinding to every start point on the map.
    def flow_all(self):
        self.path_number = 5
        for start_point in game.map.start_points:
            self.path_number += 1
            self.path_color = random_color()
            color_map[self.path_number] = self.path_color
            self.flow(start_point)
            self.reset(reset_best_paths=False)
        self.show_best_paths()

    # This function is used to apply general pathfinding process.
    def flow(self, start_point):
        if not self.busy:
            reachable = True
            self.busy = True
            list_of_new_points = [start_point]
            while self.not_found:
                points = list_of_new_points.copy()
                list_of_new_points = []
                if len(points) > 0:
                    for point in points:
                        new_points = (self.move(point))
                        self.dictionary[point] = new_points
                        list_of_new_points.extend(new_points)
                else:
                    reachable = False
                    break
            if reachable:
                self.create_list(self.end_point)
                points = self.convert_path(self.path, True)
                self.show_path(points, self.path_number)
                game.update()
                self.clear_map()
                self.busy = False
                if self.number > 0:
                    self.number -= 1
                    self.find_another_way(start_point)
                else:
                    best_path = self.find_best_path()
                    self.show_path(best_path, self.path_number)

    # This functions is used to convert the list of points to list of rows and columns without their direction values.
    # For example, [Point(1, 1, 5), Point(2, 2, 9)] = [(1, 1), (2, 2)]
    def convert_path(self, path, append_to_paths=False):
        points = []
        # Remove the end_point from the path, so its value will remain 3 instead of 4.
        path.pop(0)
        for point in path:
            points.append((point.row, point.column))
        if append_to_paths is True:
            self.paths.append(points)
        return points

    # This functions is used to draw all the best paths found.
    def show_best_paths(self):
        number_of_best_paths = len(self.best_paths)
        for number in range(number_of_best_paths):
            best_path = self.best_paths[number]
            path_number = number + 6
            self.show_path(best_path, path_number)
        self.need_reset = True

    @staticmethod
    def show_path(path, path_number):
        # This functions is used to draw the path given.
        for point in path:
            row, column = point
            game.map.matrix[row, column] = path_number
        game.update()

    @staticmethod
    def clear_map():
        # It does not clear the endpoints, walls and start points!
        game.map.matrix[np.where(game.map.matrix >= 4)] = 0


# a class to make changes on the map.
class Painter:
    def __init__(self):
        # wall = 1, start_point = 2, end_point = 3
        self.block_type = 1

    # This functions is used to check if the left mouse button or the right mouse button is clicked.
    def check(self):
        if mouse.buttons.left:
            self.create()
        elif mouse.buttons.right:
            self.delete()

    # This functions is used to create a block on the map.
    def create(self):
        if not pathfinding.busy and not pathfinding.need_reset:
            row, column = self.convert_mouse_position()
            value = game.map.matrix[row, column]
            if value != self.block_type:
                game.map.matrix[row, column] = self.block_type
                game.draw_block(row, column)
                if self.block_type == 2:
                    game.map.start_points.append(Point(row, column, direction=5))
                if value == 2:
                    self.remove_start_point(row, column)

    # This functions is used to delete a block on the map.
    def delete(self):
        if not pathfinding.busy and not pathfinding.need_reset:
            row, column = self.convert_mouse_position()
            value = game.map.matrix[row, column]
            if value != 0:
                game.map.matrix[row, column] = 0
                game.draw_block(row, column)
                if value == 2:
                    self.remove_start_point(row, column)

    # Since We use the class Point(), we can't directly find its index value in a list by creating a new one.
    # So the following two functions need to be used.
    def remove_start_point(self, row, column):
        start_point = self.find_start_point(row, column)
        game.map.start_points.remove(start_point)

    @staticmethod
    def find_start_point(row, column):
        for start_point in game.map.start_points:
            if start_point.row == row and start_point.column == column:
                return start_point

    @staticmethod
    def convert_mouse_position():
        row = game.convert_position(value=mouse.position.x, axis="x", case=1)
        column = game.convert_position(value=mouse.position.y, axis="y", case=1)
        return row, column


class Game:
    def __init__(self, resolution: tuple[int, int], map_size: tuple[int, int], grid_size: int, grid_color: tuple):
        self.screen = Screen(resolution)
        self.map = Map(map_size)
        block_width = self.screen.resolution.width / self.map.shape.width
        block_height = self.screen.resolution.height / self.map.shape.height
        self.block = Shape((block_width, block_height))
        self.grid_size, self.grid_color = grid_size, grid_color

    # This functions is used to convert row and column data to x and y data. Or vice versa...
    def convert_position(self, value, axis: str, case: int):
        if case == 0:
            if axis == "x":
                result = value * self.block.width
            elif axis == "y":
                result = value * self.block.height
            else:
                raise ValueError(f"axis have to be 'x' or 'y' but {axis} is given instead.")
        elif case == 1:
            if axis == "x":
                result = int(value // self.block.width)
            elif axis == "y":
                result = int(value // self.block.height)
            else:
                raise ValueError(f"axis have to be 'x' or 'y' but {axis} is given instead.")
        else:
            raise ValueError(f"case have to be 0 or 1 but {case} is given instead.")
        return result

    # This functions is used to draw a block on specified position.
    def draw_block(self, row, column):
        value = self.map.matrix[row, column]
        color = color_map[value]
        x = self.convert_position(value=row, axis="x", case=0)
        y = self.convert_position(value=column, axis="y", case=0)
        geometry = (x, y, self.block.width, self.block.height)
        pygame.draw.rect(self.screen.display, color, geometry)
        # 2 grids need to be drawn again after drawing the block.
        self.draw_grid(value=row, axis="x")
        self.draw_grid(value=column, axis="y")
        pygame.display.flip()

    # This functions is used to draw a grid on specified axis.
    def draw_grid(self, value, axis):
        if axis == "x":
            x = self.convert_position(value, axis, case=0)
            geometry = (x, 0, self.grid_size, self.screen.resolution.height)
            pygame.draw.rect(self.screen.display, self.grid_color, geometry)
        elif axis == "y":
            y = self.convert_position(value, axis, case=0)
            geometry = (0, y, self.screen.resolution.width, self.grid_size)
            pygame.draw.rect(self.screen.display, self.grid_color, geometry)
        else:
            raise ValueError(f"axis have to be 'x' or 'y' but {axis} is given instead.")

    # This functions is used to draw block for all row and column values.
    def draw_blocks(self):
        for row in range(self.map.shape.width):
            for column in range(self.map.shape.height):
                value = self.map.matrix[row, column]
                color = color_map[value]
                x = self.convert_position(value=row, axis="x", case=0)
                y = self.convert_position(value=column, axis="y", case=0)
                geometry = (x, y, self.block.width, self.block.height)
                pygame.draw.rect(self.screen.display, color, geometry)

    # This functions is used to draw the horizontal and vertical grids.
    def draw_grids(self):
        for row in range(self.map.shape.width):
            x = self.convert_position(value=row, axis="x", case=0)
            geometry = (x, 0, self.grid_size, self.screen.resolution.height)
            pygame.draw.rect(self.screen.display, self.grid_color, geometry)
        for column in range(self.map.shape.height):
            y = self.convert_position(value=column, axis="y", case=0)
            geometry = (0, y, self.screen.resolution.width, self.grid_size)
            pygame.draw.rect(self.screen.display, self.grid_color, geometry)

    # This functions is used to update the screen after drawing blocks and grids.
    def update(self):
        self.draw_blocks()
        self.draw_grids()
        pygame.display.flip()


# a class for the mouse events and its position.
class Mouse:
    def __init__(self):
        self.position = self.Position()
        self.buttons = self.Buttons()

    class Position:
        def __init__(self):
            self.x = self.y = None

        # This function is used to update the mouse position (2 axes; x and y).
        def update(self):
            self.x, self.y = pygame.mouse.get_pos()

    class Buttons:
        def __init__(self):
            self.left = self.middle = self.right = None

        # This function is used to update the states of the mouse buttons (True if clicked, False otherwise).
        def update(self):
            self.left, self.middle, self.right = pygame.mouse.get_pressed(num_buttons=3)


class Keyboard:
    @staticmethod
    def update():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # Press '1' to change the block type to 'wall'.
                if event.key == pygame.K_1:
                    painter.block_type = 1
                # Press '2' to change the block type to 'start point'.
                elif event.key == pygame.K_2:
                    painter.block_type = 2
                # Press '3' to change the block type to 'end point'.
                elif event.key == pygame.K_3:
                    painter.block_type = 3
                # Press 'S' to start pathfinding process.
                elif event.key == pygame.K_s:
                    if game.map.check() is True and pathfinding.need_reset is False:
                        Thread(target=pathfinding.flow_all, args=()).start()
                # Press 'R' to reset (it will clear the start points, end points and walls).
                elif event.key == pygame.K_r:
                    # Create the map with the same size.
                    width = game.map.shape.width
                    height = game.map.shape.height
                    game.map = Map(map_size=(width, height))
                    pathfinding.reset(True)
                # Press 'C' to clear (it won't clear the start points, end points and walls).
                elif event.key == pygame.K_c:
                    pathfinding.clear_map()
                    pathfinding.reset(True)
                    game.update()
                # Press 'Q' to quit.
                elif event.key == pygame.K_q:
                    pathfinding.need_reset = True
                    game.map.start_points = []
                    pathfinding.busy = True
                    pygame.quit()
                    quit()


game = Game(resolution=(1000, 1000), map_size=(40, 40), grid_size=1, grid_color=(0, 0, 0))
pathfinding = Pathfinding()
painter = Painter()
mouse = Mouse()
keyboard = Keyboard()
game.update()
delay = 0.01

if __name__ == "__main__":
    while True:
        sleep(delay)
        mouse.position.update()
        mouse.buttons.update()
        keyboard.update()
        painter.check()
