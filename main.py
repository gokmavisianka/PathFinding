"""
MIT License

Copyright (c) 2023 Rasim Mert YILDIRIM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import heapq
import pygame
import numpy as np
from threading import Thread


# Define the string to be printed on the screen to inform the user.
string = """

Keys:
    'Q': quit.
    'S': start pathfinding.
    'C': clear the map.
    'E': erase the path (Walls/Obstacles won't be erased).
    
    To change the block type:
        '0': 'empty'
        '1': 'wall'
        '2': 'start point'
        '3': 'end point'
    
Use mouse buttons to place/erase blocks on the map.
    'Left Click' : Place
    'Right Click': Erase
    
    If you can't use the right mouse button, simply press the '0' key and use your left mouse button.
    
"""

# Print the string on the screen.
print(string)

# Define the directions.
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# For the diagonal movement:
# directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Define the colors for different block types.
# 0: empty, 1: wall, 2: start point, 3: end_point, 4: path.
color_map = {0: (255, 255, 255),
             1: (0, 0, 0),
             2: (0, 0, 255),
             3: (255, 0, 0),
             4: (0, 255, 0)}


class Shape:
    def __init__(self, dimensions: tuple[int, int]):
        self.width, self.height = dimensions


class Screen:
    def __init__(self, resolution: tuple[int, int]):
        self.resolution = Shape(resolution)
        self.display = pygame.display.set_mode(resolution)


class Map:
    def __init__(self, map_size):
        self.grid = None
        self.shape = None
        self.create(map_size)
        self.start_point = None
        self.end_point = None

    # This function is used to create the map with the given map size
    def create(self, map_size):
        self.shape = Shape(map_size)
        width, height = map_size
        self.grid = np.ones((width, height))
        self.grid[1: (width - 1), 1: (height - 1)] = 0

    # This function is used to check if the start point and the end point are defined.
    def check(self):
        if self.start_point is not None and self.end_point is not None:
            return True
        # Return True if the condition is met. Otherwise, return False.
        return False

    # This function is used to recreate the map with the same size.
    def clear(self):
        width = self.shape.width
        height = self.shape.height
        self.create((width, height))


# a class for pathfinding process.
class Pathfinding:
    def __init__(self):
        self.need_reset = False
        self.busy = False

    # This function is used to reset the variables.
    def reset(self):
        self.need_reset = False
        self.busy = False

    # This function is used to apply general process.
    def apply(self):
        start_point = game.map.start_point
        end_point = game.map.end_point
        path = self.find_path(start_point, end_point)
        self.show_path(path)
        # The path must be erased before drawing the same path (or different one).
        self.need_reset = True

    # Find the path to the end position from the start position.
    def find_path(self, start_position: tuple[int, int], end_position: tuple[int, int]):
        # The heap is a priority queue represented as a list.
        heap = []
        #  Push the start position onto the priority queue heap with a priority of 0.
        heapq.heappush(heap, (0, start_position))
        # Create a dictionary to keep track of how we reached each cell in the path.
        came_from = {}
        # 'cost_so_far' is the cost of the shortest path from the start node to the current position,
        cost_so_far = {start_position: 0}
        # 'estimated_total_cost' is the estimated total cost of the path from the start position to the
        # end position that goes through the current position.
        estimated_total_cost = {start_position: self.heuristic(start_position, end_position)}
        # Keep doing stuff if heap is not empty.
        while heap:
            # Remove and return the smallest item from the priority queue, then define it as the current position.
            current_position = heapq.heappop(heap)[1]
            # Check if the current position is the end position.
            if current_position == end_position:
                path = []
                # Create the path using the list came_from.
                while current_position != start_position:
                    path.append(current_position)
                    current_position = came_from[current_position]
                # Lastly, remove the end position from the path.
                path.remove(end_position)
                # You can choose to append the start position to the path and not remove the end position from the path.
                # path.append(start_position)
                return path[::-1]
            for direction in directions:
                # Define the next position using the row and column values of the current position and direction.
                next_position = (current_position[0] + direction[0], current_position[1] + direction[1])
                # Check if the next position is valid. If not, continue with the next direction value, if any.
                if not self.is_valid(next_position):
                    continue
                # Calculate the cost of the movement from the current position to the next position.
                cost = self.get_cost(current_position, next_position)
                # Check if the next position is explored.
                explored = next_position in cost_so_far
                # Define the cost of the path to the next position through the current position.
                new_cost = cost_so_far[current_position] + cost
                if explored:
                    # Check if the new cost is lower than the previously computed cost to the same position.
                    cost_is_lower = new_cost < cost_so_far[next_position]
                else:
                    # No need to define "cost_is_lower" as True or False, as "if not explored" will be True.
                    cost_is_lower = None
                if not explored or cost_is_lower:
                    # Set the "parent" of next_position to be current_position.
                    came_from[next_position] = current_position
                    # Update the cost of reaching next_position to be new_cost.
                    cost_so_far[next_position] = new_cost
                    # Calculate the estimated total cost.
                    estimated_total_cost[next_position] = cost_so_far[next_position] + self.heuristic(next_position,
                                                                                                      end_position)
                    # Push the next position onto the priority queue heap with a priority of estimated total cost.
                    heapq.heappush(heap, (estimated_total_cost[next_position], next_position))
        return []

    @staticmethod
    def heuristic(position: tuple[int, int], end_position: tuple[int, int]):
        # This function is used to calculate the distance between the given position and end position.
        row, column = position
        end_row, end_column = end_position
        return abs(row - end_row) + abs(column - end_column)

    @staticmethod
    def is_valid(position: tuple[int, int]):
        # This function is used to check if the given position is valid.
        row, column = position
        # Check if the position is within the map boundaries.
        if 0 <= row < game.map.shape.width and 0 <= column < game.map.shape.height:
            # Check if there is no obstacle in the given position.
            if game.map.grid[row][column] != 1:
                return True
        # If all conditions are satisfied, return True. Otherwise, return False.
        return False

    @staticmethod
    def get_cost(current_position: tuple[int, int], next_position: tuple[int, int]):
        # This function is used to calculate the cost of the movement.
        current_row, current_column = current_position
        next_row, next_column = next_position
        # If the movement is diagonal, the cost is 14. Otherwise, it is 10.
        if abs(current_row - next_row) == 1 and abs(current_column - next_column) == 1:
            return 14
        else:
            return 10

    @staticmethod
    def show_path(path: list):
        # This functions is used to draw the path given.
        for cell in path:
            row, column = cell
            game.map.grid[row, column] = 4
        game.update()

    @staticmethod
    def erase_path():
        # This function is used to erase the map (It doesn't erase the walls).
        game.map.grid[game.map.grid == 4] = 0


# a class for making changes on the map.
class Painter:
    def __init__(self):
        # empty = 0, wall = 1, start_point = 2, end_point = 3
        self.block_type = 1

    # This functions is used to check if the left mouse button or the right mouse button is clicked.
    def check(self):
        if mouse.buttons.left:
            self.create()
        elif mouse.buttons.right:
            self.erase()

    # This functions is used to create a block on the map.
    def create(self):
        if not pathfinding.need_reset:
            row, column = self.convert_mouse_position()
            value = game.map.grid[row, column]
            # Check if we are not trying to paint the cell same colour.
            if value != self.block_type:
                # Add the start point to the list of start points.
                if self.block_type == 2:
                    # Erase the previous start point, if exists.
                    if game.map.start_point is not None:
                        previous_row, previous_column = game.map.start_point
                        game.map.grid[previous_row, previous_column] = 0
                        game.draw_block(previous_row, previous_column)
                    game.map.start_point = (row, column)
                # Create the end point.
                if self.block_type == 3:
                    # Erase the previous end point, if exists.
                    if game.map.end_point is not None:
                        previous_row, previous_column = game.map.end_point
                        game.map.grid[previous_row, previous_column] = 0
                        game.draw_block(previous_row, previous_column)
                    game.map.end_point = (row, column)
                # Remove the start point if it is replaced by different block type.
                if value == 2:
                    game.map.start_point = None
                # Remove the end point if it is replaced by different block type.
                elif value == 3:
                    game.map.end_point = None
                game.map.grid[row, column] = self.block_type
                game.draw_block(row, column)

    # This functions is used to erase a block on the map.
    def erase(self):
        if not pathfinding.busy and not pathfinding.need_reset:
            row, column = self.convert_mouse_position()
            value = game.map.grid[row, column]
            # Check if the cell is not empty.
            if value != 0:
                game.map.grid[row, column] = 0
                game.draw_block(row, column)
                # Remove the start point if it is erased.
                if value == 2:
                    game.map.start_point = None
                # Remove the end point if it is erased.
                elif value == 3:
                    game.map.end_point = None

    @staticmethod
    def convert_mouse_position():
        # This function is used to convert x and y values to row and column values based on the block size.
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

    # This functions is used to convert row and column values to x and y values. Or vice versa...
    def convert_position(self, value: int, axis: str, case: int):
        # Case 0 is for converting row value to x value or column value to y value based on the block size.
        if case == 0:
            if axis == "x":
                result = value * self.block.width
            elif axis == "y":
                result = value * self.block.height
            else:
                raise ValueError(f"axis have to be 'x' or 'y' but {axis} is given instead.")
        # Case 1 is for converting x value to row value or y value to column value based on the block size.
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

    # This functions is used to draw a block on a specified position.
    def draw_block(self, row: int, column: int):
        value = self.map.grid[row, column]
        color = color_map[value]
        x = self.convert_position(value=row, axis="x", case=0)
        y = self.convert_position(value=column, axis="y", case=0)
        geometry = (x, y, self.block.width, self.block.height)
        pygame.draw.rect(self.screen.display, color, geometry)
        # Two grids need to be redrawn after the block is drawn.
        self.draw_grid(value=row, axis="x")
        self.draw_grid(value=column, axis="y")
        pygame.display.flip()

    # This functions is used to draw a grid on a specified axis.
    def draw_grid(self, value: int, axis: str):
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

    # This functions is used to draw blocks for all row and column values.
    def draw_blocks(self):
        for row in range(self.map.shape.width):
            for column in range(self.map.shape.height):
                value = self.map.grid[row, column]
                color = color_map[value]
                x = self.convert_position(value=row, axis="x", case=0)
                y = self.convert_position(value=column, axis="y", case=0)
                geometry = (x, y, self.block.width, self.block.height)
                pygame.draw.rect(self.screen.display, color, geometry)

    # This functions is used to draw all the horizontal and vertical grids.
    def draw_grids(self):
        for row in range(self.map.shape.width):
            x = self.convert_position(value=row, axis="x", case=0)
            geometry = (x, 0, self.grid_size, self.screen.resolution.height)
            pygame.draw.rect(self.screen.display, self.grid_color, geometry)
        for column in range(self.map.shape.height):
            y = self.convert_position(value=column, axis="y", case=0)
            geometry = (0, y, self.screen.resolution.width, self.grid_size)
            pygame.draw.rect(self.screen.display, self.grid_color, geometry)

    # This functions is used to update the screen after drawing all the blocks and grids.
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

        # This function is used to update the mouse position.
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
                # If you can't use the right mouse button, press '0' to change the block type to 'empty'/'air'.
                if event.key == pygame.K_0:
                    painter.block_type = 0
                # Press '1' to change the block type to 'wall'.
                elif event.key == pygame.K_1:
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
                        Thread(target=pathfinding.apply, args=()).start()
                # Press 'C' to clear the map (it will erase everything on the map).
                elif event.key == pygame.K_c:
                    game.map.clear()
                    pathfinding.reset()
                    # Update the screen after clearing the map.
                    game.update()
                # Press 'E' to erase the path (it won't erase the walls).
                elif event.key == pygame.K_e:
                    pathfinding.erase_path()
                    pathfinding.reset()
                    game.update()
                # Press 'Q' to quit.
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()


game = Game(resolution=(1000, 1000), map_size=(40, 40), grid_size=1, grid_color=(0, 0, 0))
pathfinding = Pathfinding()
painter = Painter()
mouse = Mouse()
keyboard = Keyboard()
game.update()
# Define the clock.
clock = pygame.time.Clock()

if __name__ == "__main__":
    while True:
        mouse.position.update()
        mouse.buttons.update()
        keyboard.update()
        painter.check()
        # Limit the number of iterations per second to 1000.
        clock.tick(1000)
  
